"""
client_pi.py — Runs on Raspberry Pi with camera + robot.

Usage:
    python client_pi.py --host 192.168.1.100 --server-port 8000 \
        --port /dev/ttyACM0 --prompt "pick up the cup"

    # For DROID-config server (pi05_droid / pi0_fast_droid):
    python client_pi.py --host 192.168.1.100 --config droid \
        --port /dev/ttyACM0 --prompt "pick up the cup"

    # For LIBERO-config server (pi0_libero / pi05_libero):
    python client_pi.py --host 192.168.1.100 --config libero \
        --port /dev/ttyACM0 --prompt "pick up the cup"
"""

import argparse
import time
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from openpi_client import image_tools
from openpi_client import websocket_client_policy


# ── Observation builders ─────────────────────────────────────────
# Each server config expects a different set of observation keys.
# See the root README "Observation Keys by Config" table.

def build_observation_droid(img, state, prompt, wrist_img=None):
    """Build observation dict for pi05_droid / pi0_fast_droid configs.

    Expected by the server:
        observation/exterior_image_1_left  (224, 224, 3) uint8
        observation/wrist_image_left       (224, 224, 3) uint8
        observation/joint_position         (7,) float
        observation/gripper_position       (1,) float
        prompt                             str
    """
    obs = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                wrist_img if wrist_img is not None else img, 224, 224
            )
        ),
        "observation/joint_position": state[:7].astype(np.float64),
        "observation/gripper_position": state[7:8].astype(np.float64) if len(state) > 7 else np.zeros(1),
        "prompt": prompt,
    }
    return obs


def build_observation_libero(img, state, prompt, wrist_img=None):
    """Build observation dict for pi0_libero / pi05_libero configs.

    Expected by the server:
        observation/image       (224, 224, 3) uint8
        observation/wrist_image (224, 224, 3) uint8  (optional)
        observation/state       (8,) float
        prompt                  str
    """
    obs = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/state": state.astype(np.float64),
        "prompt": prompt,
    }
    if wrist_img is not None:
        obs["observation/wrist_image"] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
    return obs


OBSERVATION_BUILDERS = {
    "droid": build_observation_droid,
    "libero": build_observation_libero,
}


# ── Robot Interface (SO-101 via LeRobot) ────────────────────────

class RobotInterface:
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(self, port: str, robot_id: str = "my_so101"):
        config = SO101FollowerConfig(port=port, id=robot_id)
        self.robot = SO101Follower(config)
        self.robot.connect()
        self.num_joints = len(self.JOINT_NAMES)
        print(f"Robot initialized ({self.num_joints} joints) on {port}")

    def get_state(self) -> np.ndarray:
        """Read current joint positions (degrees)."""
        obs = self.robot.get_observation()
        return np.array(
            [obs[f"{name}.pos"] for name in self.JOINT_NAMES],
            dtype=np.float32,
        )

    def send_action(self, action: np.ndarray):
        """Send joint commands (degrees). Slices to num_joints if action is longer."""
        clipped = action[:self.num_joints]
        action_dict = {
            f"{name}.pos": float(clipped[i])
            for i, name in enumerate(self.JOINT_NAMES)
        }
        self.robot.send_action(action_dict)

    def disconnect(self):
        self.robot.disconnect()


# ── Camera Interface ─────────────────────────────────────────────
# Supports both USB cameras (via OpenCV) and Raspberry Pi CSI cameras
# (via picamera2). Use --camera-type to select.

class CameraInterface:
    """OpenCV-based camera for USB webcams."""

    def __init__(self, camera_index=0, width=640, height=480):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        print(f"Camera {camera_index} ready ({width}x{height}) [opencv]")

    def capture(self) -> np.ndarray:
        """Capture a frame as a numpy array (H, W, 3) uint8 RGB."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        # Convert BGR -> RGB
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


class PiCameraInterface:
    """picamera2-based camera for Raspberry Pi CSI cameras (IMX477, OV5647, etc.)."""

    def __init__(self, camera_index=0, width=640, height=480):
        from picamera2 import Picamera2
        self.cam = Picamera2(camera_index)
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()
        # Let auto-exposure settle
        import time
        time.sleep(0.5)
        print(f"PiCamera {camera_index} ready ({width}x{height}) [picamera2]")

    def capture(self) -> np.ndarray:
        """Capture a frame as a numpy array (H, W, 3) uint8 RGB."""
        return self.cam.capture_array()

    def release(self):
        self.cam.stop()
        self.cam.close()


def make_camera(camera_type: str, camera_index: int = 0, width: int = 640, height: int = 480):
    """Factory to create the right camera interface."""
    if camera_type == "picamera2":
        return PiCameraInterface(camera_index, width, height)
    elif camera_type == "opencv":
        return CameraInterface(camera_index, width, height)
    elif camera_type == "auto":
        try:
            return PiCameraInterface(camera_index, width, height)
        except Exception:
            return CameraInterface(camera_index, width, height)
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")


# ── Control Loop ─────────────────────────────────────────────────

def control_loop(host: str, port: int, prompt: str, robot_port: str,
                  robot_id: str = "my_so101", config: str = "droid",
                  camera_type: str = "auto", hz: float = 5.0):
    build_obs = OBSERVATION_BUILDERS[config]
    robot = RobotInterface(port=robot_port, robot_id=robot_id)
    camera = make_camera(camera_type)

    # Connect to the GPU policy server via websocket
    client = websocket_client_policy.WebsocketClientPolicy(
        host=host,
        port=port,
    )
    print(f"Connected to policy server at {host}:{port} (config: {config})")

    period = 1.0 / hz
    step = 0

    print(f"\nControl loop running at {hz} Hz")
    print(f"Prompt: \"{prompt}\"")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            t_start = time.time()

            # 1. Capture image
            img = camera.capture()  # (H, W, 3) uint8

            # 2. Read robot state
            state = robot.get_state()

            # 3. Build observation in the format the server config expects
            observation = build_obs(img, state, prompt)

            # 4. Inference — returns action chunk
            result = client.infer(observation)
            action_chunk = result["actions"]  # shape: (horizon, action_dim)

            # 5. Execute first action (or loop through the chunk)
            robot.send_action(action_chunk[0])

            # 6. Timing
            elapsed = time.time() - t_start
            if step % 10 == 0:
                print(f"[Step {step}] latency={elapsed*1000:.0f}ms")

            time.sleep(max(0, period - elapsed))
            step += 1

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps")
    finally:
        camera.release()
        robot.disconnect()


# ── Action Chunking (optional, better performance) ───────────────

def control_loop_chunked(host: str, port: int, prompt: str, robot_port: str,
                         robot_id: str = "my_so101", config: str = "droid",
                         camera_type: str = "auto", hz: float = 10.0):
    """
    More efficient: query the server every N steps and execute the
    full action chunk open-loop in between. This reduces latency impact.
    """
    build_obs = OBSERVATION_BUILDERS[config]
    robot = RobotInterface(port=robot_port, robot_id=robot_id)
    camera = make_camera(camera_type)

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Connected to {host}:{port} (chunked mode, config: {config})")

    period = 1.0 / hz
    step = 0
    action_chunk = None
    chunk_idx = 0

    try:
        while True:
            t_start = time.time()

            # Only query server when we've exhausted the current chunk
            if action_chunk is None or chunk_idx >= len(action_chunk):
                img = camera.capture()
                state = robot.get_state()

                observation = build_obs(img, state, prompt)

                result = client.infer(observation)
                action_chunk = result["actions"]
                chunk_idx = 0

                if step % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"[Step {step}] inference latency={elapsed*1000:.0f}ms, "
                          f"chunk_size={len(action_chunk)}")

            # Execute next action from chunk
            robot.send_action(action_chunk[chunk_idx])
            chunk_idx += 1
            step += 1

            elapsed = time.time() - t_start
            time.sleep(max(0, period - elapsed))

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps")
    finally:
        camera.release()
        robot.disconnect()


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pi0.5 Robot Client")
    parser.add_argument("--host", default="localhost",
                        help="GPU server IP (default: localhost)")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Policy server port (default: 8000)")
    parser.add_argument("--port", required=True,
                        help="USB serial port for the SO-101 (e.g. /dev/ttyACM0)")
    parser.add_argument("--robot-id", default="my_so101",
                        help="Robot calibration identity (default: my_so101)")
    parser.add_argument("--config", default="droid",
                        choices=list(OBSERVATION_BUILDERS.keys()),
                        help="Server policy config — must match what the server is running "
                             "(default: droid)")
    parser.add_argument("--prompt", default="pick up the object",
                        help="Language instruction for the robot")
    parser.add_argument("--hz", type=float, default=5.0,
                        help="Control frequency in Hz")
    parser.add_argument("--camera-type", default="auto",
                        choices=["auto", "opencv", "picamera2"],
                        help="Camera backend: auto (try picamera2 first), opencv, "
                             "or picamera2 (default: auto)")
    parser.add_argument("--chunked", action="store_true",
                        help="Use action chunking for better throughput")
    args = parser.parse_args()

    if args.chunked:
        control_loop_chunked(args.host, args.server_port, args.prompt,
                             args.port, args.robot_id, args.config,
                             args.camera_type, args.hz)
    else:
        control_loop(args.host, args.server_port, args.prompt,
                     args.port, args.robot_id, args.config,
                     args.camera_type, args.hz)

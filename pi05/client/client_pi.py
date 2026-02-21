"""
client_pi.py — Runs on Raspberry Pi with camera + robot.

Usage:
    python client_pi.py --host 192.168.1.100 --server-port 8000 \
        --port /dev/ttyACM0 --prompt "pick up the cup"
"""

import argparse
import time
import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from openpi_client import image_tools
from openpi_client import websocket_client_policy


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
# Uses OpenCV directly. LeRobot also provides camera support via
# SO101FollowerConfig.cameras if you prefer a unified setup.

class CameraInterface:
    def __init__(self, camera_index=0, width=640, height=480):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        print(f"Camera {camera_index} ready ({width}x{height})")

    def capture(self) -> np.ndarray:
        """Capture a frame as a numpy array (H, W, 3) uint8 BGR."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        # Convert BGR → RGB
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


# ── Control Loop ─────────────────────────────────────────────────

def control_loop(host: str, port: int, prompt: str, robot_port: str,
                  robot_id: str = "my_so101", hz: float = 5.0):
    robot = RobotInterface(port=robot_port, robot_id=robot_id)
    camera = CameraInterface()

    # Connect to the GPU policy server via websocket
    client = websocket_client_policy.WebsocketClientPolicy(
        host=host,
        port=port,
    )
    print(f"Connected to policy server at {host}:{port}")

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

            # 3. Build observation in OpenPI format
            # - Resize + pad to 224x224 (standard for π₀ models)
            # - Server handles normalization of state
            observation = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, 224, 224)
                ),
                "observation/state": state,
                "prompt": prompt,
            }

            # If you have a wrist camera too:
            # observation["observation/wrist_image"] = image_tools.convert_to_uint8(
            #     image_tools.resize_with_pad(wrist_img, 224, 224)
            # )

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
                         robot_id: str = "my_so101", hz: float = 10.0):
    """
    More efficient: query the server every N steps and execute the
    full action chunk open-loop in between. This reduces latency impact.
    """
    robot = RobotInterface(port=robot_port, robot_id=robot_id)
    camera = CameraInterface()

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Connected to {host}:{port} (chunked mode)")

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

                observation = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, 224, 224)
                    ),
                    "observation/state": state,
                    "prompt": prompt,
                }

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
    parser = argparse.ArgumentParser(description="π₀.₅ Robot Client")
    parser.add_argument("--host", default="localhost",
                        help="GPU server IP (default: localhost)")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Policy server port (default: 8000)")
    parser.add_argument("--port", required=True,
                        help="USB serial port for the SO-101 (e.g. /dev/ttyACM0)")
    parser.add_argument("--robot-id", default="my_so101",
                        help="Robot calibration identity (default: my_so101)")
    parser.add_argument("--prompt", default="pick up the object",
                        help="Language instruction for the robot")
    parser.add_argument("--hz", type=float, default=5.0,
                        help="Control frequency in Hz")
    parser.add_argument("--chunked", action="store_true",
                        help="Use action chunking for better throughput")
    args = parser.parse_args()

    if args.chunked:
        control_loop_chunked(args.host, args.server_port, args.prompt,
                             args.port, args.robot_id, args.hz)
    else:
        control_loop(args.host, args.server_port, args.prompt,
                     args.port, args.robot_id, args.hz)

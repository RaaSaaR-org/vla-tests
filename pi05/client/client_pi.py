"""
client_pi.py — Runs on Raspberry Pi with camera + SO-101 robot.

Usage:
    python client_pi.py --host 192.168.1.100 --server-port 8000 \
        --port /dev/ttyACM0 --config droid --prompt "pick up the cup"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy


# ── Observation builders ─────────────────────────────────────────
# Each server config expects a different set of observation keys.
# See the root README "Observation Keys by Config" table.

def build_observation_droid(img, state, prompt, wrist_img=None):
    """Build observation dict for pi05_droid / pi0_fast_droid configs.

    Expected by the server (DROID = Franka Panda, 7-DOF arm + 1 gripper):
        observation/exterior_image_1_left  (224, 224, 3) uint8
        observation/wrist_image_left       (224, 224, 3) uint8
        observation/joint_position         (7,) float — radians
        observation/gripper_position       (1,) float — [0, 1]
        prompt                             str

    SO-101 state is 6 values: 5 arm joints (degrees) + 1 gripper (0-100).
    We convert degrees→radians, pad 5 joints→7 with zeros, and normalize
    the gripper to [0, 1].

    NOTE: The DROID model was trained on a 7-DOF Franka Panda. The SO-101
    has 5-DOF with a different kinematic chain. The pretrained checkpoint
    will NOT produce useful actions without fine-tuning on SO-101 data.
    """
    # SO-101: state[0:5] = arm joints (degrees), state[5] = gripper (0-100)
    arm_joints_deg = state[:5]
    gripper_raw = state[5] if len(state) > 5 else 0.0

    # Convert degrees to radians (DROID training data is in radians)
    arm_joints_rad = np.deg2rad(arm_joints_deg).astype(np.float64)

    # Pad 5 joints to 7 with zeros (Franka has 7 joints)
    joint_position = np.zeros(7, dtype=np.float64)
    joint_position[:5] = arm_joints_rad

    # Normalize gripper: SO-101 reports 0-100, DROID expects 0.0-1.0
    gripper_position = np.array([float(gripper_raw) / 100.0], dtype=np.float64)

    obs = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                wrist_img if wrist_img is not None else img, 224, 224
            )
        ),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
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
    # Full SO-101 joint list (6 motors)
    ALL_JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(self, port: str, robot_id: str = "my_so101",
                 calibration_file: str | None = None,
                 skip_motors: list[str] | None = None):
        """Initialize SO-101 robot.

        Args:
            skip_motors: list of joint names to skip (e.g. ["wrist_roll"] if
                         motor 5 has a bad cable). The robot will operate with
                         the remaining motors.
        """
        self.skip_motors = set(skip_motors or [])
        self.JOINT_NAMES = [n for n in self.ALL_JOINT_NAMES if n not in self.skip_motors]
        self._try_connect(port, robot_id)
        self.num_joints = len(self.JOINT_NAMES)

        # Load calibration so joint reads/writes are in degrees
        self._load_calibration(calibration_file)
        print(f"Robot initialized ({self.num_joints} joints) on {port}")
        if self.skip_motors:
            print(f"  Skipped motors: {', '.join(sorted(self.skip_motors))}")

    def _try_connect(self, port: str, robot_id: str, retries: int = 3):
        """Try connecting, auto-skipping motors that fail handshake."""
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        for attempt in range(1, retries + 1):
            try:
                config = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
                self.robot = SO101Follower(config)
                self.robot.connect(calibrate=False)
                return
            except RuntimeError as e:
                err = str(e)
                if "Missing motor IDs" in err and attempt < retries:
                    # Parse which motor IDs are missing
                    import re
                    missing_ids = {int(m) for m in re.findall(r"- (\d+)", err)}
                    id_to_name = {i + 1: n for i, n in enumerate(self.ALL_JOINT_NAMES)}
                    missing_names = {id_to_name.get(mid, f"unknown_{mid}") for mid in missing_ids}
                    print(f"Attempt {attempt}/{retries}: motors {missing_names} not responding, retrying...")
                    import time
                    time.sleep(1)
                elif "Missing motor IDs" in err:
                    # Final attempt failed — offer to skip the bad motors
                    import re
                    missing_ids = {int(m) for m in re.findall(r"- (\d+)", err)}
                    id_to_name = {i + 1: n for i, n in enumerate(self.ALL_JOINT_NAMES)}
                    missing_names = {id_to_name.get(mid, f"unknown_{mid}") for mid in missing_ids}
                    print(f"Motors {missing_names} unreachable after {retries} attempts — skipping them")
                    self.skip_motors |= missing_names
                    self.JOINT_NAMES = [n for n in self.ALL_JOINT_NAMES if n not in self.skip_motors]
                    # Build a custom motor bus without the bad motors
                    self._connect_partial(port, robot_id)
                    return
                else:
                    raise

    def _connect_partial(self, port: str, robot_id: str):
        """Connect with a subset of motors (excluding skip_motors)."""
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech.feetech import FeetechMotorsBus

        motor_ids = {
            "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
            "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
        }
        motors = {}
        for name in self.JOINT_NAMES:
            motors[name] = Motor(id=motor_ids[name], model="sts3215",
                                 norm_mode=MotorNormMode.RANGE_M100_100)
        self.bus = FeetechMotorsBus(port=port, motors=motors)
        self.bus.connect()
        self.robot = None  # no SO101Follower, using bus directly

    def _load_calibration(self, calibration_file: str | None):
        """Find and apply servo calibration data."""
        from lerobot.motors import MotorCalibration

        candidates = []
        if calibration_file:
            candidates.append(Path(calibration_file))
        candidates += [
            Path.cwd() / "robot-calibration-data.json",
            Path(__file__).parent / "robot-calibration-data.json",
            Path.home() / "develop" / "backup" / "robot-calibration-data.json",
        ]

        cal_path = None
        for p in candidates:
            if p.exists():
                cal_path = p
                break

        if cal_path is None:
            print("WARNING: no calibration file found, joint values will be raw ticks")
            return

        with open(cal_path) as f:
            cal_json = json.load(f)

        # Only filter motors when in partial mode (self.robot is None)
        cal_map = {}
        for name, entry in cal_json.items():
            if self.robot is None and name in self.skip_motors:
                continue
            cal_map[name] = MotorCalibration(
                id=entry["id"],
                drive_mode=entry.get("drive_mode", 0),
                homing_offset=entry["homing_offset"],
                range_min=entry["range_min"],
                range_max=entry["range_max"],
            )
        bus = self.robot.bus if self.robot is not None else self.bus
        bus.write_calibration(cal_map)
        print(f"Calibration loaded from {cal_path}")

    def get_state(self) -> np.ndarray:
        """Read current joint positions (degrees, gripper 0-100).

        Returns array with ALL_JOINT_NAMES ordering (6 values), with 0.0
        for any skipped motors.
        """
        if self.robot is not None:
            obs = self.robot.get_observation()
            return np.array(
                [obs[f"{name}.pos"] for name in self.ALL_JOINT_NAMES],
                dtype=np.float32,
            )
        else:
            # Partial mode: read from bus directly
            positions = self.bus.sync_read("Present_Position")
            result = np.zeros(len(self.ALL_JOINT_NAMES), dtype=np.float32)
            for name in self.JOINT_NAMES:
                idx = self.ALL_JOINT_NAMES.index(name)
                result[idx] = float(positions[name])
            return result

    def send_action(self, action: np.ndarray):
        """Send joint position commands (in degrees). Maps to available motors."""
        if self.robot is not None:
            clipped = action[:len(self.ALL_JOINT_NAMES)]
            action_dict = {
                f"{name}.pos": float(clipped[i])
                for i, name in enumerate(self.ALL_JOINT_NAMES)
            }
            self.robot.send_action(action_dict)
        else:
            # Partial mode: only send to motors we have
            goal = {}
            for i, name in enumerate(self.ALL_JOINT_NAMES):
                if name not in self.skip_motors and i < len(action):
                    goal[name] = int(action[i])
            self.bus.sync_write("Goal_Position", goal)

    def disconnect(self):
        if self.robot is not None:
            self.robot.disconnect()
        else:
            self.bus.disconnect()


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
        import time
        time.sleep(0.5)  # let auto-exposure settle
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


# ── DROID Action Conversion ───────────────────────────────────────
# DROID actions are (horizon, 8): 7 joint velocities (rad/s) + 1 gripper [0,1].
# SO-101 needs absolute joint positions in degrees. We integrate the velocity
# into the current state and convert units.

NUM_ARM_JOINTS_SO101 = 5   # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
NUM_ARM_JOINTS_DROID = 7   # Franka Panda 7-DOF

# Best-effort mapping: DROID Franka joints → SO-101 joints.
# Franka: [shoulder_rot, shoulder_pitch, elbow_rot, elbow_pitch, wrist_rot, wrist_pitch, wrist_roll]
# SO-101: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
# We map the 5 most-relevant Franka joints. This is approximate — fine-tuning
# is needed for real task performance.
DROID_TO_SO101_JOINT_MAP = [0, 1, 3, 5, 6]  # Franka indices → SO-101 joints


def droid_action_to_so101(action: np.ndarray, current_state: np.ndarray,
                          dt: float) -> np.ndarray:
    """Convert a single DROID action (8,) to SO-101 joint positions (6,).

    Args:
        action: (8,) — 7 joint velocities (rad/s) + 1 gripper position [0,1]
        current_state: (6,) — current SO-101 state [5 arm joints (deg) + gripper (0-100)]
        dt: time step in seconds (1/hz)

    Returns:
        (6,) — target joint positions for SO-101 [5 arm (deg) + gripper (0-100)]
    """
    joint_velocities_rad = action[:NUM_ARM_JOINTS_DROID]  # (7,) rad/s
    gripper_target_01 = action[7] if len(action) > 7 else 0.5  # [0, 1]

    # Pick the 5 relevant Franka joints for SO-101
    mapped_velocities = joint_velocities_rad[DROID_TO_SO101_JOINT_MAP]  # (5,) rad/s

    # Convert velocity (rad/s) to delta degrees: v_rad * dt * (180/pi)
    delta_deg = np.rad2deg(mapped_velocities * dt)

    # Integrate: new position = current position + delta
    arm_current_deg = current_state[:NUM_ARM_JOINTS_SO101]
    arm_target_deg = arm_current_deg + delta_deg

    # Gripper: DROID outputs [0,1], SO-101 expects 0-100
    gripper_target = float(gripper_target_01) * 100.0

    target = np.zeros(6, dtype=np.float32)
    target[:NUM_ARM_JOINTS_SO101] = arm_target_deg
    target[NUM_ARM_JOINTS_SO101] = gripper_target
    return target


# ── Control Loop ─────────────────────────────────────────────────

def control_loop(host: str, server_port: int, prompt: str, config: str,
                 robot_port: str, robot_id: str, camera_type: str,
                 calibration_file: str | None = None,
                 skip_motors: list[str] | None = None, hz: float = 5.0):
    build_obs = OBSERVATION_BUILDERS[config]
    is_droid = (config == "droid")
    robot = RobotInterface(port=robot_port, robot_id=robot_id,
                           calibration_file=calibration_file,
                           skip_motors=skip_motors)
    camera = make_camera(camera_type)

    client = websocket_client_policy.WebsocketClientPolicy(
        host=host, port=server_port,
    )
    print(f"Connected to policy server at {host}:{server_port} (config: {config})")

    period = 1.0 / hz
    step = 0

    print(f"\nControl loop running at {hz} Hz")
    print(f"Prompt: \"{prompt}\"")
    if is_droid:
        print("NOTE: DROID actions are joint velocities — integrating into positions")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            t_start = time.time()

            img = camera.capture()
            state = robot.get_state()
            observation = build_obs(img, state, prompt)

            result = client.infer(observation)
            action_chunk = result["actions"]

            if is_droid:
                target = droid_action_to_so101(action_chunk[0], state, period)
            else:
                target = action_chunk[0]
            robot.send_action(target)

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

def control_loop_chunked(host: str, server_port: int, prompt: str, config: str,
                         robot_port: str, robot_id: str, camera_type: str,
                         calibration_file: str | None = None,
                         skip_motors: list[str] | None = None, hz: float = 10.0):
    """
    More efficient: query the server every N steps and execute the
    full action chunk open-loop in between. This reduces latency impact.
    """
    build_obs = OBSERVATION_BUILDERS[config]
    is_droid = (config == "droid")
    robot = RobotInterface(port=robot_port, robot_id=robot_id,
                           calibration_file=calibration_file,
                           skip_motors=skip_motors)
    camera = make_camera(camera_type)

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=server_port)
    print(f"Connected to {host}:{server_port} (chunked mode, config: {config})")

    period = 1.0 / hz
    step = 0
    action_chunk = None
    chunk_idx = 0

    try:
        while True:
            t_start = time.time()

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

            # For DROID: integrate velocities from current state each step
            if is_droid:
                current_state = robot.get_state()
                target = droid_action_to_so101(action_chunk[chunk_idx], current_state, period)
            else:
                target = action_chunk[chunk_idx]
            robot.send_action(target)
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
    parser.add_argument("--calibration-file", default=None,
                        help="Path to robot-calibration-data.json (auto-detected if omitted)")
    parser.add_argument("--skip-motors", nargs="*", default=None,
                        help="Motor names to skip (e.g. wrist_roll if cable is loose)")
    parser.add_argument("--config", default="droid",
                        choices=list(OBSERVATION_BUILDERS.keys()),
                        help="Server policy config -- must match what the server is running "
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

    kwargs = dict(
        host=args.host, server_port=args.server_port, prompt=args.prompt,
        config=args.config, robot_port=args.port, robot_id=args.robot_id,
        camera_type=args.camera_type, calibration_file=args.calibration_file,
        skip_motors=args.skip_motors, hz=args.hz,
    )

    if args.chunked:
        control_loop_chunked(**kwargs)
    else:
        control_loop(**kwargs)

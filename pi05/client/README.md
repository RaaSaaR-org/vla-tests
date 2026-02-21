# VLA Client (Raspberry Pi)

Runs on the robot side — captures camera frames, reads joint state, sends observations to the GPU policy server over WebSocket, and executes the returned actions on the robot.

## Prerequisites

- Raspberry Pi (or any Linux machine) with a USB camera
- A robot arm (e.g. [SO-100](https://wiki.seeedstudio.com/lerobot_so100m_new/)) — or just run in stub mode to test communication
- Python 3.10+
- Network access to the GPU server (direct or via [Tailscale](https://tailscale.com/))
- `cmake` (required by `dm-tree`, a transitive dependency)

```bash
# On Raspberry Pi (Debian/Ubuntu)
sudo apt update
sudo apt install cmake python3-dev
```

## Install

```bash
# Install uv (Python package manager, replaces pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd pi05/client
uv sync
```

## Usage

```bash
uv run client_pi.py --host <GPU_SERVER_IP> --port 8000 --prompt "pick up the cup"
```

The client will:
1. Open the USB camera
2. Connect to the GPU server via WebSocket
3. Start the control loop: capture image, read joints, send observation, execute returned action

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | GPU server IP address (use Tailscale IP if remote) |
| `--port` | `8000` | Policy server port |
| `--prompt` | `pick up the object` | Natural language instruction for the robot |
| `--hz` | `5.0` | Control loop frequency in Hz |
| `--chunked` | off | Execute the full action chunk between server queries |

### Action Chunking

By default the client queries the server every loop iteration and executes only the first action from the returned chunk. With `--chunked`, it executes the entire action chunk open-loop before querying again. This reduces the impact of network latency:

```bash
uv run client_pi.py --host 192.168.1.100 --prompt "pick up the cup" --chunked --hz 10
```

## Customization

The client ships with **stub implementations** for `RobotInterface` and `CameraInterface`. Out of the box it will open a USB camera and print actions to the console. To actually move a robot, you need to replace the stubs.

### Robot Interface

Edit the `RobotInterface` class in `client_pi.py`. Example for a LeRobot SO-100 arm:

```python
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

class RobotInterface:
    def __init__(self):
        self.robot = ManipulatorRobot(...)
        self.robot.connect()

    def get_state(self) -> np.ndarray:
        return self.robot.get_observation()["observation.state"]

    def send_action(self, action: np.ndarray):
        self.robot.send_action(action)

    def disconnect(self):
        self.robot.disconnect()
```

See the [SO-100 setup guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) for hardware assembly and LeRobot configuration.

### Camera Interface

The default `CameraInterface` uses OpenCV with a USB camera (`/dev/video0`). Adjust `camera_index`, `width`, and `height` as needed. If your robot has a wrist camera, add a second `CameraInterface` instance and include it in the observation dict.

### Observation Keys

The observation dictionary keys **must match** the policy config running on the server. The default client uses LIBERO-style keys. If you're running a different checkpoint, update the keys:

**LIBERO** (default in `client_pi.py`):
```python
observation = {
    "observation/image": ...,       # 224x224 uint8 RGB
    "observation/state": state,     # 8-dim
    "prompt": prompt,
}
```

**DROID** (for `pi05_droid` / `pi0_fast_droid`):
```python
observation = {
    "observation/exterior_image_1_left": ...,   # 224x224 uint8 RGB
    "observation/wrist_image_left": ...,        # 224x224 uint8 RGB (wrist cam)
    "observation/joint_position": state[:7],    # 7 joints
    "observation/gripper_position": state[7:8], # 1 gripper
    "prompt": prompt,
}
```

See the [observation keys table](../../README.md#observation-keys-by-config) in the main README for all supported configs.

## Troubleshooting

- **"Cannot open camera 0":** Check that your USB camera is connected and shows up as `/dev/video0`. Try `ls /dev/video*`.
- **Connection refused:** Make sure the GPU server is running and reachable. Test with `curl http://<SERVER_IP>:8000` or ping.
- **Action shape mismatch:** Your `RobotInterface.send_action()` expects a different number of joints than the model outputs. Check which config is running on the server.
- **Slow control loop:** Network latency dominates. Use `--chunked` mode and/or run both machines on the same LAN.

## Further Reading

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi) — server framework and model details
- [pi0.5 in LeRobot docs](https://huggingface.co/docs/lerobot/pi05) — model overview, fine-tuning instructions
- [SO-100 robot arm guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) — hardware assembly, motor calibration, LeRobot setup

# VLA Client (Raspberry Pi)

Runs on the robot side — captures camera frames, reads joint state from an SO-101 arm, sends observations to the GPU policy server over WebSocket, and executes the returned actions on the robot.

## Prerequisites

- Raspberry Pi (or any Linux machine) with a USB camera
- An [SO-101 robot arm](https://wiki.seeedstudio.com/lerobot_so100m_new/) with Feetech servos, connected via USB
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

# Install dependencies (includes LeRobot with Feetech support)
cd pi05/client
uv sync
```

## First-Time Robot Setup

Before running the client, you need to set up and calibrate the SO-101 arm using LeRobot's CLI tools:

```bash
# 1. Find your robot's USB serial port
uv run lerobot-find-port

# 2. Set up motors (sets IDs, baud rates, etc.)
uv run lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0

# 3. Calibrate joint ranges (follow the interactive prompts)
uv run lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101
```

The `--robot.id` you use during calibration must match the `--robot-id` you pass to `client_pi.py` (default: `my_so101`). Calibration data is stored in `~/.cache/lerobot/calibration/`.

See the [SO-101 setup guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) for hardware assembly details.

## Usage

```bash
uv run client_pi.py --port /dev/ttyACM0 --host <GPU_SERVER_IP> --prompt "pick up the cup"
```

The client will:
1. Connect to the SO-101 arm on the specified USB port
2. Open the USB camera
3. Connect to the GPU server via WebSocket
4. Start the control loop: capture image, read joints, send observation, execute returned action

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | *(required)* | USB serial port for the SO-101 (e.g. `/dev/ttyACM0`) |
| `--robot-id` | `my_so101` | Robot calibration identity (must match calibration) |
| `--host` | `localhost` | GPU server IP address (use Tailscale IP if remote) |
| `--server-port` | `8000` | Policy server port |
| `--prompt` | `pick up the object` | Natural language instruction for the robot |
| `--hz` | `5.0` | Control loop frequency in Hz |
| `--chunked` | off | Execute the full action chunk between server queries |

### Action Chunking

By default the client queries the server every loop iteration and executes only the first action from the returned chunk. With `--chunked`, it executes the entire action chunk open-loop before querying again. This reduces the impact of network latency:

```bash
uv run client_pi.py --port /dev/ttyACM0 --host 192.168.1.100 --prompt "pick up the cup" --chunked --hz 10
```

## Camera

The client uses OpenCV with a USB camera (`/dev/video0`) by default. Adjust `camera_index`, `width`, and `height` in the `CameraInterface` class as needed.

LeRobot also provides built-in camera support via `SO101FollowerConfig.cameras` — see the [LeRobot docs](https://huggingface.co/docs/lerobot/pi05) if you prefer a unified setup.

If your robot has a wrist camera, add a second `CameraInterface` instance and include it in the observation dict.

### Observation Keys

The observation dictionary keys **must match** the policy config running on the server. The default client uses LIBERO-style keys. If you're running a different checkpoint, update the keys:

**LIBERO** (default in `client_pi.py`):
```python
observation = {
    "observation/image": ...,       # 224x224 uint8 RGB
    "observation/state": state,     # 6-dim (joint positions in degrees)
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
- **Action shape mismatch:** The model may output more action dimensions than the SO-101's 6 joints — the client automatically slices to 6. If you see shape errors, check which config is running on the server.
- **Slow control loop:** Network latency dominates. Use `--chunked` mode and/or run both machines on the same LAN.
- **Serial port not found:** Run `uv run lerobot-find-port` to discover the correct port. Make sure your user has permission (`sudo usermod -aG dialout $USER`).
- **Calibration errors:** Re-run `uv run lerobot-calibrate` with the correct `--robot.id`. Delete stale calibration from `~/.cache/lerobot/calibration/` if needed.

## Further Reading

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi) — server framework and model details
- [pi0.5 in LeRobot docs](https://huggingface.co/docs/lerobot/pi05) — model overview, fine-tuning instructions
- [SO-101 robot arm guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) — hardware assembly, motor calibration, LeRobot setup

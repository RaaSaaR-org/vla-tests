# VLA Client (Raspberry Pi)

Runs on the robot side — captures camera frames, reads joint state from an SO-101 arm, sends observations to the GPU policy server over WebSocket, and executes the returned actions on the robot.

## Prerequisites

- Raspberry Pi (or any Linux machine) with a CSI or USB camera
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

# On Raspberry Pi with a CSI camera: use --system-site-packages so
# the venv can access the system picamera2 + libcamera bindings
uv venv --system-site-packages
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
# For a pi05_droid server (the recommended config):
uv run client_pi.py --port /dev/ttyACM0 --host <GPU_SERVER_IP> --config droid --prompt "pick up the cup"

# For a pi0_libero server:
uv run client_pi.py --port /dev/ttyACM0 --host <GPU_SERVER_IP> --config libero --prompt "pick up the cup"
```

The `--config` flag controls which observation keys the client sends. It **must match** the policy config running on the server — sending the wrong keys will crash the server.

The client will:
1. Connect to the SO-101 arm on the specified USB port
2. Open the camera (CSI via picamera2, or USB via OpenCV)
3. Connect to the GPU server via WebSocket
4. Start the control loop: capture image, read joints, send observation, execute returned action

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | *(required)* | USB serial port for the SO-101 (e.g. `/dev/ttyACM0`) |
| `--robot-id` | `my_so101` | Robot calibration identity (must match calibration) |
| `--calibration-file` | *(auto-detected)* | Path to `robot-calibration-data.json` |
| `--host` | `localhost` | GPU server IP address (use Tailscale IP if remote) |
| `--server-port` | `8000` | Policy server port |
| `--config` | `droid` | Server policy config — must match the server (`droid` or `libero`) |
| `--prompt` | `pick up the object` | Natural language instruction for the robot |
| `--hz` | `5.0` | Control loop frequency in Hz |
| `--camera-type` | `auto` | Camera backend: `auto` (try picamera2 first), `opencv`, or `picamera2` |
| `--chunked` | off | Execute the full action chunk between server queries |

### Action Chunking

By default the client queries the server every loop iteration and executes only the first action from the returned chunk. With `--chunked`, it executes the entire action chunk open-loop before querying again. This reduces the impact of network latency:

```bash
uv run client_pi.py --port /dev/ttyACM0 --host 192.168.1.100 --prompt "pick up the cup" --chunked --hz 10
```

## Camera

Two camera backends are supported:

- **picamera2** — for Raspberry Pi CSI cameras (IMX477, OV5647, etc.). Requires `picamera2` installed system-wide (pre-installed on Raspberry Pi OS). Create the venv with `uv venv --system-site-packages` so it can access the system picamera2.
- **opencv** — for USB webcams via OpenCV `VideoCapture`.

The `--camera-type auto` default tries picamera2 first, then falls back to opencv.

LeRobot also provides built-in camera support via `SO101FollowerConfig.cameras` — see the [LeRobot docs](https://huggingface.co/docs/lerobot/pi05) if you prefer a unified setup.

If your robot has a wrist camera, add a second camera instance and pass the frames to the observation builder.

### Observation Keys

The `--config` flag selects the observation builder, so you no longer need to edit the Python code to switch between configs. The client handles the key mapping automatically:

| `--config` | Server configs | Image keys sent | State keys sent |
|------------|----------------|----------------|-----------------|
| `droid` | `pi05_droid`, `pi0_fast_droid` | `exterior_image_1_left`, `wrist_image_left` | `joint_position` (7), `gripper_position` (1) |
| `libero` | `pi0_libero`, `pi05_libero` | `image`, `wrist_image` (optional) | `state` (8) |

See the [observation keys table](../../README.md#observation-keys-by-config) in the main README for all supported configs.

## Troubleshooting

- **"Cannot open camera 0":** Check that your USB camera is connected and shows up as `/dev/video0`. Try `ls /dev/video*`. For CSI cameras, use `--camera-type picamera2`.
- **Connection refused:** Make sure the GPU server is running and reachable. Test with `curl http://<SERVER_IP>:8000` or ping.
- **Action shape mismatch:** The model may output more action dimensions than the SO-101's 6 joints — the client automatically slices to 6. If you see shape errors, check which config is running on the server.
- **Slow control loop:** Network latency dominates. Use `--chunked` mode and/or run both machines on the same LAN.
- **Serial port not found:** Run `uv run lerobot-find-port` to discover the correct port. Make sure your user has permission (`sudo usermod -aG dialout $USER`).
- **Calibration errors:** Re-run `uv run lerobot-calibrate` with the correct `--robot.id`. Delete stale calibration from `~/.cache/lerobot/calibration/` if needed. Alternatively, pass `--calibration-file` to point at a `robot-calibration-data.json`.

## Further Reading

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi) — server framework and model details
- [pi0.5 in LeRobot docs](https://huggingface.co/docs/lerobot/pi05) — model overview, fine-tuning instructions
- [SO-101 robot arm guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) — hardware assembly, motor calibration, LeRobot setup

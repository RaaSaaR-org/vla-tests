# VLA Client (Raspberry Pi)

Runs on the robot side — captures camera frames, reads joint state from an SO-101 arm, sends observations to the GPU policy server, and executes the returned actions on the robot.

Supports two server backends:

| Backend | Protocol | Server port | Framework | Use case |
|---------|----------|-------------|-----------|----------|
| **lerobot** (recommended) | gRPC | 8080 | PyTorch / LeRobot | SO-101-native checkpoints (pi0.5, SmolVLA, ACT) |
| **openpi** (legacy) | WebSocket | 8000 | JAX / OpenPI | DROID/Libero checkpoints (Franka-trained) |

## Prerequisites

- Raspberry Pi (or any Linux machine) with a CSI or USB camera
- An [SO-101 robot arm](https://wiki.seeedstudio.com/lerobot_so100m_new/) with Feetech servos, connected via USB
- Python 3.11+
- Network access to the GPU server (direct or via [Tailscale](https://tailscale.com/))

```bash
# On Raspberry Pi (Debian/Ubuntu)
sudo apt update
sudo apt install cmake python3-dev
```

## Install

Two venvs exist on the Pi — use the one matching your backend:

```bash
# ── LeRobot backend (recommended) ──
# Venv at ~/repos/vla-tests/.venv-lerobot/
# Already installed with: lerobot[feetech,async] + system picamera2
source ~/repos/vla-tests/.venv-lerobot/bin/activate

# ── OpenPI backend (legacy) ──
# Venv at ~/repos/vla-tests/pi05/client/.venv/
cd pi05/client
uv venv --system-site-packages
uv sync
```

## First-Time Robot Setup

Before running the client, you need to set up and calibrate the SO-101 arm using LeRobot's CLI tools:

```bash
# 1. Find your robot's USB serial port
python -m lerobot.find_port

# 2. Set up motors (sets IDs, baud rates, etc.)
python -m lerobot.setup_motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0

# 3. Calibrate joint ranges (follow the interactive prompts)
python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101
```

The `--robot.id` you use during calibration must match the `--robot-id` you pass to `client_pi.py` (default: `my_so101`). Calibration data is stored in `~/.cache/lerobot/calibration/`.

See the [SO-101 setup guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) for hardware assembly details.

## Usage

### LeRobot backend (recommended)

Uses a community checkpoint fine-tuned on SO-101 pick-and-place. Actions are absolute joint positions — no conversion needed.

```bash
# Activate the lerobot venv
source ~/repos/vla-tests/.venv-lerobot/bin/activate

# Run with the SO-101 pick-and-place checkpoint
python client_pi.py --backend lerobot --host <GPU_SERVER_IP> \
    --port /dev/ttyACM0 --prompt "pick up the green object"

# Custom model / options
python client_pi.py --backend lerobot --host <GPU_SERVER_IP> \
    --port /dev/ttyACM0 \
    --model Elvinky/pi05_so101_pick_place_bottle \
    --policy-type pi05 --device cuda \
    --actions-per-chunk 50 --hz 5 \
    --prompt "pick up the bottle"
```

The client will:
1. Connect to the SO-101 arm on the specified USB port
2. Open the camera (CSI via picamera2, or USB via OpenCV)
3. Connect to the LeRobot gRPC server and send the model config (server loads the checkpoint)
4. Run the control loop: capture image + joint state, send observation via gRPC, execute returned action chunk

### OpenPI backend (legacy)

Uses the original OpenPI WebSocket server with DROID/Libero checkpoints (Franka-trained, needs conversion for SO-101).

```bash
# Activate the openpi venv
source ~/repos/vla-tests/pi05/client/.venv/bin/activate

# DROID config (default)
python client_pi.py --backend openpi --host <GPU_SERVER_IP> \
    --port /dev/ttyACM0 --config droid --prompt "pick up the cup"

# Libero config
python client_pi.py --backend openpi --host <GPU_SERVER_IP> \
    --port /dev/ttyACM0 --config libero --prompt "pick up the cup"
```

The `--config` flag controls which observation keys the client sends. It **must match** the policy config running on the server.

### Options

#### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `lerobot` | Server backend: `lerobot` (gRPC) or `openpi` (WebSocket) |
| `--port` | *(required)* | USB serial port for the SO-101 (e.g. `/dev/ttyACM0`) |
| `--host` | `localhost` | GPU server IP address (use Tailscale IP if remote) |
| `--server-port` | `8080`/`8000` | Policy server port (auto-set by backend) |
| `--robot-id` | `my_so101` | Robot calibration identity (must match calibration) |
| `--calibration-file` | *(auto-detected)* | Path to `robot-calibration-data.json` |
| `--skip-motors` | *(none)* | Motor names to skip (e.g. `wrist_roll` if cable is loose) |
| `--prompt` | `pick up the object` | Natural language instruction for the robot |
| `--hz` | `5.0` | Control loop frequency in Hz |
| `--camera-type` | `auto` | Camera backend: `auto`, `opencv`, or `picamera2` |

#### LeRobot-only flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Elvinky/pi05_so101_pick_place_bottle` | HuggingFace checkpoint or local path |
| `--policy-type` | `pi05` | Policy architecture (`pi05`, `smolvla`, `act`, etc.) |
| `--device` | `cuda` | Inference device on the GPU server |
| `--actions-per-chunk` | `50` | Actions returned per inference call |

#### OpenPI-only flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `droid` | Server policy config (`droid` or `libero`) |
| `--chunked` | off | Execute full action chunk between server queries |

## Camera

Two camera backends are supported:

- **picamera2** — for Raspberry Pi CSI cameras (IMX477, OV5647, etc.). Requires `picamera2` installed system-wide (pre-installed on Raspberry Pi OS). The venv must have `--system-site-packages`.
- **opencv** — for USB webcams via OpenCV `VideoCapture`.

The `--camera-type auto` default tries picamera2 first, then falls back to opencv.

## Troubleshooting

- **"Cannot open camera 0":** For CSI cameras, use `--camera-type picamera2`. For USB cameras, check `ls /dev/video*`.
- **Connection refused:** Make sure the GPU server is running. Test: `curl http://<SERVER_IP>:8080` (lerobot) or `:8000` (openpi).
- **"Missing motor IDs: - 5":** Motor 5 (wrist_roll) has a loose cable. Use `--skip-motors wrist_roll` to skip it.
- **Slow control loop:** The LeRobot backend uses action chunking by default (50 actions per inference). Network latency only matters once per chunk.
- **Serial port not found:** Run `python -m lerobot.find_port`. Make sure your user has permission (`sudo usermod -aG dialout $USER`).
- **Calibration errors:** Re-run calibration or pass `--calibration-file` to point at a backup `robot-calibration-data.json`.

## Further Reading

- [LeRobot pi0.5 docs](https://huggingface.co/docs/lerobot/pi05) — model overview, fine-tuning, async inference
- [OpenPI repository](https://github.com/Physical-Intelligence/openpi) — JAX server framework
- [SO-101 robot arm guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) — hardware assembly, motor calibration

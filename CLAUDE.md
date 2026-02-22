# CLAUDE.md

## Project Overview

VLA (Vision-Language-Action) test project for running open-source VLA models on a GPU server and controlling an SO-101 robot arm remotely. Supports pi0/pi0.5 via both OpenPI (JAX, WebSocket) and LeRobot-native (PyTorch, gRPC). GR00T N1 and SmolVLA docs also included.

**Architecture:** GPU server (policy inference) <--Tailscale--> Raspberry Pi 5 (CSI camera + SO-101 arm)

### Two pi0.5 paths

| Path | Framework | Protocol | Checkpoint | Notes |
|------|-----------|----------|------------|-------|
| **OpenPI** (legacy) | JAX | WebSocket :8000 | `pi05_droid` (Franka-trained) | Custom `client_pi.py`. DROID actions are velocities for 7-DOF Franka — requires conversion for SO-101. |
| **LeRobot** (recommended) | PyTorch | gRPC :8080 | `lerobot/pi05_base` (cross-embodiment) | Built-in `policy_server` + `robot_client`. Native SO-101 support. Needs fine-tuning on SO-101 demos. |

## Directory Structure

```
pi05/                  pi0/pi0.5 via OpenPI (WebSocket, port 8000)
  client/
    client_pi.py       Main client (SO-101 RobotInterface, CameraInterface, control loop)
    pyproject.toml     Python project config (uses uv)
  server/              GPU server setup (OpenPI, checkpoints)
                       LeRobot pi0.5 also installed here (micromamba env `lerobot`)

groot/                 NVIDIA GR00T N1.6 via Isaac-GR00T / LeRobot
  client/              PolicyClient or LeRobot in-process setup
  server/              run_gr00t_server.py setup (port 5555) + fine-tuning

smolvla/               HuggingFace SmolVLA via LeRobot async (gRPC, port 8080)
  client/              robot_client setup + Python API example
  server/              policy_server setup + fine-tuning

docs/
  vla-models.md        Comparison of all VLA models (pi0, SmolVLA, GR00T, ACT, etc.)
agents.md              How to SSH to Pi, run commands from agents
```

## Key Technical Details

### pi0/pi0.5 (currently implemented and tested end-to-end)
- **Package manager:** uv (not pip). Use `uv sync` and `uv run`.
- **Python:** >=3.11 (OpenPI requires 3.11+, not 3.10)
- **Client dependencies:** `openpi-client`, `lerobot[feetech]`
- **Image processing:** Camera frames resized+padded to 224x224 via `openpi_client.image_tools`
- **Policy server:** OpenPI `serve_policy.py`, default port 8000, WebSocket
- **Supported configs:** `pi05_droid`, `pi0_fast_droid`, `pi0_aloha`, `pi0_libero`
- **Default config:** `droid` (client sends DROID observation keys by default)
- **GPU requirements:** >8 GB VRAM inference, >22.5 GB LoRA, >70 GB full fine-tuning
- **Checkpoint size:** 11.6 GB (pi05_droid), cached at `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid`

### GR00T N1.6 (setup docs ready)
- **Server:** Isaac-GR00T `run_gr00t_server.py`, port 5555
- **Client:** `gr00t.policy.server_client.PolicyClient` or LeRobot in-process
- **Images:** 224x224 (N1.5) or native resolution (N1.6)
- **Requires:** flash-attn (CUDA only), modality.json for state/action mapping
- **GPU:** ~8-10 GB inference, ~25 GB fine-tuning
- **License:** NVIDIA non-commercial

### SmolVLA (setup docs ready)
- **Server:** `lerobot.async_inference.policy_server`, port 8080, gRPC
- **Client:** `lerobot.async_inference.robot_client`
- **Images:** 256x256 (internally 512x512 via PixelShuffle)
- **GPU:** ~2 GB inference (runs on consumer GPUs, MPS, even CPU)
- **Key params:** `actions_per_chunk`, `chunk_size_threshold`

## Working with the Client (pi0.5)

- `client_pi.py` uses `SO101Follower` from LeRobot for real hardware control
- Import path: `lerobot.robots.so_follower` (not `so101_follower`)
- Config class: `lerobot.robots.so_follower.config_so_follower.SO101FollowerConfig`
- CLI args: `--port` (required, USB serial), `--robot-id` (default `my_so101`), `--server-port` (default 8000)
- Default config is `droid` — sends DROID observation keys (see table below)
- Two control loop modes: standard (query every step) and chunked (`--chunked`)
- Observation dict keys must match the server-side policy config exactly
- Camera: uses picamera2 for CSI cameras (Pi), falls back to OpenCV for USB cameras
- Calibration: auto-loaded from several candidate paths (see `_load_calibration` in client_pi.py)

### Observation Keys

| `--config` | Server configs | Image keys | State keys |
|------------|----------------|-----------|------------|
| `droid` (default) | `pi05_droid`, `pi0_fast_droid` | `observation/exterior_image_1_left`, `observation/wrist_image_left` | `observation/joint_position` (7), `observation/gripper_position` (1) |
| `libero` | `pi0_libero`, `pi05_libero` | `observation/image`, `observation/wrist_image` | `observation/state` (8) |

## Hardware Setup

### GPU Server (peter-ubuntu)
- **Tailscale IP:** `100.125.78.40`
- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB VRAM)
- **CUDA:** 13 (driver), 12.x (runtime via pip packages — backward compatible)
- **OpenPI repo:** `/home/peter/repos/openpi/`
- **Environments:**
  - `openpi` (Python 3.11 + uv) — JAX, for OpenPI `serve_policy.py`
  - `lerobot` (Python 3.11 + pip) — PyTorch, for LeRobot `policy_server` + `lerobot-train`
- **RTX 5090 note:** JAX inference works via PTX compilation. Do NOT use PyTorch-format checkpoints (`.safetensors`) with OpenPI — stick to default JAX/Orbax checkpoints. LeRobot uses PyTorch natively and works fine.

### Raspberry Pi 5
- **Tailscale IP:** `100.84.200.83`
- **SSH:** `ssh mindcube@100.84.200.83` (key-based, no password)
- **Client code:** `~/repos/vla-tests/pi05/client/`
- **Robot port:** `/dev/ttyACM0` (SO-101 via USB)
- **Cameras:** CSI — IMX477 (index 0, HQ camera) + OV5647 (index 1)
- **Camera library:** picamera2 (system-installed, venv needs `--system-site-packages`)
- **Calibration:** `~/develop/backup/robot-calibration-data.json`

See [agents.md](agents.md) for detailed instructions on running commands on the Pi.

## Known Issues

- **Motor 5 (wrist_roll) flaky:** Sometimes fails handshake ("Missing motor IDs: - 5"). Retrying usually works — likely a loose servo cable.
- **numpy version conflict:** `openpi-client` pins `numpy<2.0` while `lerobot` requires `numpy>=2` via `rerun-sdk`. Both work fine at runtime with numpy 2.x. Workaround: install lerobot separately with `uv pip install` after `uv sync`.
- **picamera2 cannot be pip-installed:** It depends on system libcamera bindings. The venv must be created with `uv venv --system-site-packages`.
- **Partial checkpoint downloads corrupt:** If the 11.6 GB checkpoint download is interrupted, delete `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid` before retrying.

## Tested Performance

| Metric | Value |
|--------|-------|
| Server-only inference (RTX 5090) | ~47.5 ms (p50) |
| Full Pi-to-server round-trip (Tailscale) | ~132 ms steady state |
| Control loop frequency | 5 Hz |
| Longest test run | 570+ steps ("lift the green ship") |

## External References

- OpenPI repo: https://github.com/Physical-Intelligence/openpi
- LeRobot repo: https://github.com/huggingface/lerobot
- Isaac-GR00T repo: https://github.com/NVIDIA/Isaac-GR00T
- SmolVLA model: https://huggingface.co/lerobot/smolvla_base
- pi0.5 LeRobot docs: https://huggingface.co/docs/lerobot/pi05
- GR00T LeRobot docs: https://huggingface.co/docs/lerobot/groot
- SO-100 robot arm setup: https://wiki.seeedstudio.com/lerobot_so100m_new/

## Code Style

- Keep it simple and script-oriented — this is a prototype/test project
- Use numpy for array operations
- Follow existing patterns in `client_pi.py` for any new code

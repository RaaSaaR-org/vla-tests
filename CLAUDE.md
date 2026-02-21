# CLAUDE.md

## Project Overview

VLA (Vision-Language-Action) test project for running open-source VLA models on a GPU server and controlling an SO-101 robot arm remotely. Currently supports pi0/pi0.5 (OpenPI), with GR00T N1 and SmolVLA in progress.

**Architecture:** GPU server (policy inference) <--network--> Raspberry Pi (USB camera + SO-101 arm)

## Directory Structure

```
pi05/                  pi0/pi0.5 via OpenPI (WebSocket, port 8000)
  client/
    client_pi.py       Main client (SO-101 RobotInterface, CameraInterface, control loop)
    pyproject.toml     Python project config (uses uv)
  server/              GPU server setup (OpenPI, checkpoints)

groot/                 NVIDIA GR00T N1.6 via Isaac-GR00T / LeRobot
  client/              PolicyClient or LeRobot in-process setup
  server/              run_gr00t_server.py setup (port 5555) + fine-tuning

smolvla/               HuggingFace SmolVLA via LeRobot async (gRPC, port 8080)
  client/              robot_client setup + Python API example
  server/              policy_server setup + fine-tuning

docs/
  vla-models.md        Comparison of all VLA models (pi0, SmolVLA, GR00T, ACT, etc.)
```

## Key Technical Details

### pi0/pi0.5 (currently implemented)
- **Package manager:** uv (not pip). Use `uv sync` and `uv run`.
- **Python:** >=3.10
- **Client dependencies:** `openpi-client`, `lerobot[feetech]`
- **Image processing:** Camera frames resized+padded to 224x224 via `openpi_client.image_tools`
- **Policy server:** OpenPI `serve_policy.py`, default port 8000, WebSocket
- **Supported configs:** `pi05_droid`, `pi0_fast_droid`, `pi0_aloha`, `pi0_libero`
- **GPU requirements:** >8 GB VRAM inference, >22.5 GB LoRA, >70 GB full fine-tuning

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
- CLI args: `--port` (required, USB serial), `--robot-id` (default `my_so101`), `--server-port` (default 8000)
- Two control loop modes: standard (query every step) and chunked (`--chunked`)
- Observation dict keys must match the server-side policy config exactly
- Default observation keys are LIBERO-style (`observation/image`, `observation/state`)

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

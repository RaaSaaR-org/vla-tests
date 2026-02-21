# CLAUDE.md

## Project Overview

VLA (Vision-Language-Action) test project for running Physical Intelligence pi0/pi0.5 models on a GPU server and controlling a robot arm remotely over WebSocket.

**Architecture:** GPU server (OpenPI policy inference on port 8000) <--WebSocket over Tailscale--> Raspberry Pi (USB camera + robot arm like SO-100)

## Directory Structure

```
pi05/
  client/          Pi-side client: camera capture, robot control, WebSocket comms
    client_pi.py   Main client script (RobotInterface, CameraInterface, control loop)
    pyproject.toml Python project config (uses uv)
  server/          GPU server setup instructions (OpenPI, checkpoints)
```

## Key Technical Details

- **Package manager:** uv (not pip). Use `uv sync` and `uv run`.
- **Python:** >=3.10
- **Client dependency:** `openpi-client` installed from the openpi GitHub repo (`packages/openpi-client` subdirectory)
- **Image processing:** All camera frames must be resized+padded to 224x224 via `openpi_client.image_tools`
- **Policy server:** OpenPI `serve_policy.py`, default port 8000, serves inference over WebSocket
- **Supported configs:** `pi05_droid`, `pi0_fast_droid`, `pi0_aloha`, `pi0_libero` — each expects different observation keys (see root README table)
- **GPU requirements:** >8 GB VRAM for inference, >22.5 GB for LoRA fine-tuning, >70 GB for full fine-tuning
- **Checkpoints:** Auto-downloaded and cached in `~/.cache/openpi`

## Working with the Client

- `client_pi.py` has stub `RobotInterface` and `CameraInterface` classes — replace with real hardware (e.g. LeRobot ManipulatorRobot for SO-100 arms)
- Two control loop modes: standard (query every step) and chunked (`--chunked`, execute full action chunk between queries)
- Observation dict keys must match the server-side policy config exactly
- Default observation keys are LIBERO-style (`observation/image`, `observation/state`)

## External References

- OpenPI repo: https://github.com/Physical-Intelligence/openpi
- pi0.5 LeRobot docs: https://huggingface.co/docs/lerobot/pi05
- SO-100 robot arm setup: https://wiki.seeedstudio.com/lerobot_so100m_new/

## Code Style

- Keep it simple and script-oriented — this is a prototype/test project
- Use numpy for array operations
- Follow existing patterns in `client_pi.py` for any new code

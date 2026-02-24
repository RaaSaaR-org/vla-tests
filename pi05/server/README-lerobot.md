# GPU Server Setup — LeRobot Backend (gRPC)

This is the **recommended** backend for SO-101 testing. It uses the LeRobot async inference server over gRPC (port 8080) instead of the OpenPI WebSocket server (port 8000).

For the OpenPI/WebSocket backend (pi0.5 DROID/Libero checkpoints), see [`README.md`](./README.md).

## Prerequisites

- Linux (Ubuntu 22.04+)
- GPU with ≥8 GB VRAM — your RTX 5090 is more than fine
- CUDA 12+ driver
- Python 3.11+
- Network access to the Pi via Tailscale

## Install

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env   # or restart shell

# 2. Create a venv with LeRobot
uv venv --python 3.11 .venv-lerobot
source .venv-lerobot/bin/activate

# 3. Install LeRobot with async inference support
uv pip install "lerobot[async] @ git+https://github.com/huggingface/lerobot.git"

# Note: this pulls PyTorch + CUDA. First install may take a few minutes.
```

## Start the Policy Server

The server loads the model checkpoint on first request from the Pi (not at startup).

```bash
# Start with auto model loading (Pi tells server which model to load):
python -m lerobot.scripts.server.policy_server \
    --port 8080 \
    --host 0.0.0.0 \
    --device cuda

# Or pre-load a specific checkpoint:
python -m lerobot.scripts.server.policy_server \
    --port 8080 --host 0.0.0.0 --device cuda \
    --pretrained Elvinky/pi05_so101_pick_place_bottle \
    --policy-type pi05
```

The server is ready when you see something like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on 0.0.0.0:8080
```

## Verify (from any machine)

```bash
# Simple TCP port check:
curl http://<SERVER_IP>:8080
# Should get an HTTP response (even an error page means the port is open)

# Or from the Pi via Tailscale:
curl http://100.125.78.40:8080
```

## Performance (expected with RTX 5090)

| Metric | Expected |
|--------|----------|
| Inference latency | ~20–60 ms |
| Round-trip Pi→GPU→Pi | ~50–150 ms (Tailscale adds ~20ms) |
| Control freq (5 Hz, 50-action chunks) | ~1 inference per 10s |

## Checkpoint Options

| Checkpoint | HuggingFace ID | Policy type | Notes |
|-----------|---------------|-------------|-------|
| pi0.5 SO-101 (recommended) | `Elvinky/pi05_so101_pick_place_bottle` | `pi05` | Fine-tuned on SO-101 pick-and-place |
| SmolVLA | (search HF for SO-101 SmolVLA) | `smolvla` | Lighter, faster |
| ACT | (any ACT SO-101 checkpoint) | `act` | If you have custom data |

The Pi client sends the checkpoint ID and policy type — the server loads it. So you don't need to specify it at startup unless you want to pre-warm the GPU.

## Troubleshooting

- **Port 8080 blocked**: Add a firewall rule, or just use Tailscale (which handles routing automatically between our devices).
- **CUDA out of memory**: Shouldn't happen with RTX 5090 + pi0.5 (model is ~3B params, needs ~6 GB).
- **"Module not found: lerobot.scripts.server"**: The async server is a newer LeRobot feature — make sure you installed from `git+https://` (not the PyPI release which may be older).
- **Connection refused from Pi**: Check the server is listening on `0.0.0.0` (not just `localhost`).

## What the Pi Does

Once the server is running, the Pi (me / Igor) runs:

```bash
source ~/repos/vla-tests/.venv-lerobot/bin/activate
python ~/repos/vla-tests/pi05/client/client_pi.py \
    --backend lerobot \
    --host 100.125.78.40 \       # your Tailscale IP
    --port /dev/ttyACM0 \        # robot arm USB port
    --prompt "pick up the green object" \
    --hz 5
```

The client:
1. Connects to your gRPC server
2. Sends the model config (which checkpoint to load)
3. Starts the control loop: captures image + joint state → sends to server → executes returned actions on arm

## Further Reading

- [LeRobot pi0.5 docs](https://huggingface.co/docs/lerobot/pi05)
- [SO-101 pick-and-place checkpoint](https://huggingface.co/Elvinky/pi05_so101_pick_place_bottle)
- [LeRobot async inference docs](https://github.com/huggingface/lerobot/tree/main/lerobot/scripts/server)

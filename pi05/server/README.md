# GPU Server Setup

This runs on your GPU machine. It loads a VLA model checkpoint and serves inference over WebSocket so the robot client can query it remotely.

## Prerequisites

- Linux (Ubuntu 22.04 recommended)
- GPU with >8 GB VRAM (RTX 4090, A100, H100, etc.)
- CUDA 12+
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
# Clone OpenPI
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (skip large Git LFS files, checkpoints are downloaded on demand)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Serve a Policy

Pick a checkpoint and start the server. Checkpoints are auto-downloaded and cached in `~/.cache/openpi`.

**pi0.5 DROID** (recommended — latest model, best generalization):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

**pi0-FAST DROID** (autoregressive variant, lighter):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

The server listens on **port 8000** by default. You can change it with `--port=<PORT>`.

### serve_policy.py Options

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.config` | — | Policy config name (`pi05_droid`, `pi0_fast_droid`, `pi0_aloha`, `pi0_libero`) |
| `--policy.dir` | — | Path or GCS URI to the checkpoint directory |
| `--port` | `8000` | WebSocket server port |
| `--default_prompt` | — | Fallback prompt if the client doesn't send one |
| `--record` | `false` | Record policy behavior for debugging |

## Verify

From any machine that can reach the server:

```bash
# From the openpi repo
uv run examples/simple_client/main.py --host <SERVER_IP> --env DROID
```

This sends a dummy observation and prints the returned action chunk, confirming the server is running.

## GPU Memory Reference

| Task | VRAM needed | Example GPU |
|------|-------------|-------------|
| Inference only | >8 GB | RTX 4090 |
| LoRA fine-tuning | >22.5 GB | RTX 4090 |
| Full fine-tuning | >70 GB | A100 / H100 |

## Troubleshooting

- **Checkpoint download hangs:** Make sure you have `gcloud` configured or try downloading checkpoints manually.
- **CUDA out of memory:** Use a smaller config (e.g. `pi0_fast_droid` instead of `pi05_droid`) or a GPU with more VRAM.
- **Port already in use:** Another process is on port 8000. Use `--port=8001` or kill the existing process.

## Further Reading

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi)
- [pi0.5 model details (LeRobot docs)](https://huggingface.co/docs/lerobot/pi05)
- [Physical Intelligence blog post on pi0.5](https://www.physicalintelligence.company/blog/pi05)

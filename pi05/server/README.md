# GPU Server Setup

This runs on your GPU machine. It loads a VLA model checkpoint and serves inference over WebSocket so the robot client can query it remotely.

## Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- GPU with >8 GB VRAM (RTX 4090, RTX 5090, A100, H100, etc.)
- CUDA 12+ (CUDA 13 works — see [Blackwell / RTX 5090 notes](#blackwell--rtx-5090-notes) below)
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (recommended) or [conda](https://docs.conda.io/)

## Install

```bash
# Create a micromamba environment with Python 3.11 and uv
micromamba create -n openpi python=3.11 uv -c conda-forge -y

# Clone OpenPI
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install dependencies (skip large Git LFS files, checkpoints are downloaded on demand)
GIT_LFS_SKIP_SMUDGE=1 micromamba run -n openpi uv sync
```

## Serve a Policy

Activate the environment and start the server. Checkpoints are auto-downloaded and cached in `~/.cache/openpi`.

**pi0.5 DROID** (recommended — latest model, best generalization):

```bash
micromamba run -n openpi uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

**pi0-FAST DROID** (autoregressive variant, lighter):

```bash
micromamba run -n openpi uv run scripts/serve_policy.py policy:checkpoint \
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
micromamba run -n openpi uv run examples/simple_client/main.py --host <SERVER_IP> --env DROID
```

This sends a dummy observation and prints the returned action chunk, confirming the server is running.

## GPU Memory Reference

| Task | VRAM needed | Example GPU |
|------|-------------|-------------|
| Inference only | >8 GB | RTX 4090, RTX 5090 |
| LoRA fine-tuning | >22.5 GB | RTX 4090, RTX 5090 |
| Full fine-tuning | >70 GB | A100 / H100 |

## Blackwell / RTX 5090 Notes

The RTX 5090 (Blackwell, sm_120) works with OpenPI but there are things to be aware of:

- **JAX inference works out of the box.** The default JAX/Flax checkpoint path (used by all standard GCS checkpoints like `pi05_droid`) runs entirely on JAX, which supports Blackwell via CUDA driver PTX compilation. You will see harmless `ptxas` fallback warnings in the logs — these are expected.
- **PyTorch sm_120 warning is harmless for default checkpoints.** The bundled `torch` package warns that sm_120 is not in its supported architectures. This does not affect serving because the default JAX checkpoints never run PyTorch CUDA kernels — torch is only imported for type definitions and conditional branches that are not taken.
- **Do not use PyTorch-format checkpoints on Blackwell.** If a checkpoint directory contains `model.safetensors` (PyTorch format), the code path switches to full torch GPU inference, which will fail on sm_120 until PyTorch ships Blackwell support. Stick to the default JAX/Orbax checkpoints (the `gs://openpi-assets/` ones).
- **CUDA 13 driver is backward compatible.** The `jax[cuda12]` and `torch+cu126` pip packages bundle CUDA 12 runtime libraries, which run fine on a CUDA 13 driver.

### Tested performance (RTX 5090, 32 GB VRAM, pi05_droid)

| Metric | Value |
|--------|-------|
| Policy inference | ~47.5 ms (p50) |
| Client round-trip | ~60.9 ms (p50) |
| Throughput | ~16.4 Hz |

## Troubleshooting

- **Checkpoint download hangs:** OpenPI uses `gcsfs` to download checkpoints. No `gcloud` CLI is required, but a stable network connection is. If the download is interrupted, delete the partial cache (`rm -rf ~/.cache/openpi/openpi-assets/checkpoints/<config>`) and retry — partial downloads will corrupt the checkpoint.
- **CUDA out of memory:** Use a smaller config (e.g. `pi0_fast_droid` instead of `pi05_droid`) or a GPU with more VRAM.
- **Port already in use:** Another process is on port 8000. Use `--port=8001` or kill the existing process.
- **PyTorch sm_120 warning on RTX 5090:** This is expected and harmless when using JAX checkpoints. See [Blackwell notes](#blackwell--rtx-5090-notes) above.

## Further Reading

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi)
- [pi0.5 model details (LeRobot docs)](https://huggingface.co/docs/lerobot/pi05)
- [Physical Intelligence blog post on pi0.5](https://www.physicalintelligence.company/blog/pi05)

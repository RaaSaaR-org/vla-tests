# SmolVLA — GPU Server Setup

Serves Hugging Face's SmolVLA model for inference. SmolVLA uses LeRobot's built-in **async inference** system with gRPC (not WebSocket like OpenPI).

## Prerequisites

- Linux, macOS, or any machine with a GPU
- GPU with >2 GB VRAM (SmolVLA is tiny — runs on consumer GPUs, Apple Silicon MPS, even CPU)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) or pip

## Install

```bash
# Option 1: pip
pip install "lerobot[smolvla,async]"

# Option 2: from source
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla,async]"
```

## Start the Policy Server

The server starts as an empty container — the robot client tells it which model to load during the initial handshake.

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080
```

That's it. The server listens on **port 8080** and waits for a client connection.

When the client connects, it specifies:
- Which policy to load (e.g. `lerobot/smolvla_base` or your fine-tuned model)
- Which device to use (`cuda`, `mps`, `cpu`)
- How many actions per chunk

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address (use `0.0.0.0` for remote access) |
| `--port` | `8080` | gRPC server port |

## Fine-Tuning

SmolVLA works out of the box for SO-100/101 tasks if you fine-tune on your own data.

### Collect data

```bash
# Teleoperate your robot and record episodes
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so101 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --dataset.repo_id=your_user/so101_pickplace \
  --dataset.single_task="pick up the red block"
```

Aim for 50+ episodes with variation in object positions.

### Train

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your_user/so101_pickplace \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=./outputs/smolvla-so101 \
  --policy.device=cuda \
  --policy.dtype=bfloat16
```

Training takes ~4 hours on a single A100 for 20k steps. Works on smaller GPUs with reduced batch size (batch_size=16 for 6 GB GPUs).

A [Google Colab notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb) is available for free-tier training.

### Push to Hub

```bash
huggingface-cli upload your_user/smolvla_so101 ./outputs/smolvla-so101/checkpoints/last/pretrained_model
```

## GPU Memory Reference

| Task | VRAM needed |
|------|-------------|
| Inference | ~2 GB |
| Fine-tune (batch_size=16) | ~6 GB |
| Fine-tune (batch_size=64) | ~16 GB |

## Pre-trained Checkpoints

| Checkpoint | Description |
|-----------|-------------|
| [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | Base model, pre-trained on 487 community datasets (~10M frames) |
| [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero) | Fine-tuned on LIBERO simulation |

4,000+ community fine-tunes are available on the [HuggingFace Hub](https://huggingface.co/models?search=smolvla).

## Further Reading

- [SmolVLA paper](https://arxiv.org/abs/2506.01844)
- [SmolVLA blog post](https://huggingface.co/blog/smolvla)
- [SmolVLA LeRobot docs](https://huggingface.co/docs/lerobot/smolvla)
- [Async inference docs](https://huggingface.co/docs/lerobot/en/async)
- [Training notebook (Colab)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)

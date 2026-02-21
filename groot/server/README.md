# GR00T N1 — GPU Server Setup

Serves NVIDIA's GR00T N1.6 VLA model for inference. The robot client connects and sends observations to get action chunks back.

Two options: **Isaac-GR00T native server** (recommended) or **LeRobot in-process** (simpler but no remote inference).

## Prerequisites

- Linux (Ubuntu 22.04 recommended)
- GPU with >8 GB VRAM (RTX 4090, A100, H100)
- CUDA 12.4 (11.8 also works)
- [uv](https://docs.astral.sh/uv/) package manager
- `flash-attn` (required, CUDA only)

## Option A: Isaac-GR00T Server (remote inference)

This runs a dedicated inference server that clients connect to over the network.

```bash
# Clone Isaac-GR00T
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --python 3.10
```

### Serve a checkpoint

```bash
# Base model (not fine-tuned — use for testing connectivity)
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555

# Fine-tuned SO-101 checkpoint (after you fine-tune)
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path ./outputs/groot-so101/checkpoints/best \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555 \
  --denoising-steps 16
```

The server listens on **port 5555** by default.

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | — | HuggingFace model ID or local checkpoint path |
| `--embodiment-tag` | — | Robot embodiment tag (use `NEW_EMBODIMENT` for custom robots) |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `5555` | Server port |
| `--device` | `cuda:0` | GPU device |
| `--denoising-steps` | `4` | More steps = smoother motion (try 16 for real hardware) |

### Verify

```python
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="localhost", port=5555)
# Send a test observation to confirm the server responds
```

## Option B: LeRobot In-Process (local inference)

If the GPU and robot are on the same machine, you can skip the server and run inference in-process via LeRobot. See `groot/client/README.md` for the LeRobot pathway.

## Fine-Tuning

GR00T requires fine-tuning on your SO-101 data before it will produce useful actions (the base model was trained on humanoid data).

### Prepare data

Your dataset must be in GR00T-flavored LeRobot v2 format with a `modality.json`:

```json
{
  "state": {
    "single_arm": {"start": 0, "end": 6},
    "gripper": {"start": 6, "end": 7}
  },
  "action": {
    "single_arm": {"start": 0, "end": 6},
    "gripper": {"start": 6, "end": 7}
  },
  "video": {
    "front": {"original_key": "observation.images.front"}
  },
  "annotation": {
    "annotation.human.action.task_description": {}
  }
}
```

### Train via LeRobot

```bash
lerobot-train \
  --policy.type=groot \
  --dataset.repo_id=your_user/your_so101_dataset \
  --batch_size=32 \
  --steps=20000 \
  --policy.tune_diffusion_model=false \
  --output_dir=./outputs/groot-so101
```

Set `--policy.tune_diffusion_model=true` for smoother motion (uses more VRAM).

### Train via Isaac-GR00T

```bash
uv run python scripts/train.py \
  --dataset-path your_dataset/ \
  --modality-config-path your_dataset/meta/modality.json \
  --embodiment-tag NEW_EMBODIMENT \
  --num-epochs 50
```

See the [SO-101 fine-tuning blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) for a complete walkthrough.

## GPU Memory Reference

| Task | VRAM needed |
|------|-------------|
| Inference (4 denoising steps) | ~8-10 GB |
| Fine-tune (DiT frozen) | ~15 GB |
| Fine-tune (DiT unfrozen) | ~25 GB |

## Available Checkpoints

| Checkpoint | Use case |
|-----------|----------|
| [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | Base model (latest) |
| [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) | Previous version |
| [nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge) | WidowX manipulation |
| [nvidia/GR00T-N1.6-DROID](https://huggingface.co/nvidia/GR00T-N1.6-DROID) | DROID robot tasks |

## Further Reading

- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 paper](https://arxiv.org/abs/2503.14734)
- [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/groot)
- [SO-101 fine-tuning blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Seeed Studio GR00T + SO-101 wiki](https://wiki.seeedstudio.com/control_robotic_arm_via_gr00t/)

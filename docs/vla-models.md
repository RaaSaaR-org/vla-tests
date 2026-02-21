# Open-Source VLA Models for Robot Manipulation

A comparison of Vision-Language-Action models you can run with an SO-100/SO-101 arm. All of these take camera images + a text prompt and output robot joint actions.

## Quick Comparison

| Model | Params | Org | LeRobot native | GPU VRAM (inference) | Inference speed | SO-101 ready |
|-------|--------|-----|----------------|----------------------|-----------------|--------------|
| **[ACT](#act)** | ~80M | Stanford | Yes | 4-6 GB | ~100 Hz | Yes |
| **[Diffusion Policy](#diffusion-policy)** | ~50-100M | Columbia | Yes | 6-12 GB | ~10 Hz | Yes |
| **[SmolVLA](#smolvla)** | 450M | Hugging Face | Yes | ~2 GB | ~10-50 Hz (GPU) | Yes |
| **[pi0 / pi0.5](#pi0--pi05)** | ~3B | Physical Intelligence | Via OpenPI | ~14 GB | ~5-10 Hz | Yes (this repo) |
| **[GR00T N1.6](#groot-n1)** | 3B | NVIDIA | Yes | ~8-10 GB | ~23-27 Hz | Yes (fine-tune) |
| **[OpenVLA](#openvla)** | 7B | Stanford | No | ~16-18 GB | ~5 Hz | Needs IK layer |
| **[Octo](#octo)** | 27-93M | UC Berkeley | No (JAX) | ~8 GB | ~13-17 Hz | Needs fine-tune |

"LeRobot native" means you can use `lerobot-train` / `lerobot-record` / `lerobot-eval` directly. Models without native support need their own codebases.

---

## LeRobot-Native Models

These integrate directly with LeRobot's toolchain: data collection, training, evaluation, and deployment all use the same CLI.

### ACT

**Action Chunking with Transformers** -- the default starting point for LeRobot users.

- **Architecture:** CVAE with ResNet-18 vision backbone + transformer encoder/decoder. Predicts a chunk of 100 future actions at once.
- **Training:** Imitation learning from teleoperated demonstrations. ~30 min on RTX 3080 with <30 demos.
- **Strengths:** Fast, lightweight, well-documented for SO-100/101. Best "first model to try."
- **Weaknesses:** No language conditioning (task-specific only). Needs fine-tuning per task.

```bash
# Train ACT on your own data
lerobot-train --policy.type=act --dataset.repo_id=your_user/your_dataset

# Deploy on the robot
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --policy.path=outputs/train/act_run/checkpoints/last/pretrained_model
```

Links: [Paper](https://tonyzhaozh.github.io/aloha/) | [LeRobot docs](https://huggingface.co/docs/lerobot/act)

---

### Diffusion Policy

**Denoising diffusion for action prediction** -- stronger than ACT for multi-modal action distributions.

- **Architecture:** 1D temporal U-Net (CNN variant) or 8-layer transformer, with ResNet-18 image encoder. Iteratively denoises Gaussian noise into actions over 100 steps.
- **Training:** Imitation learning. ~5 hours on H100 for PushT benchmark (200k steps). ~11.5 GB VRAM at batch_size=44.
- **Strengths:** Handles multi-modal actions well (when multiple valid actions exist). Native LeRobot support.
- **Weaknesses:** Slower inference (100 denoising steps). No language conditioning.

```bash
lerobot-train --policy.type=diffusion --dataset.repo_id=your_user/your_dataset
```

Links: [Project page](https://diffusion-policy.cs.columbia.edu/) | [Pre-trained model](https://huggingface.co/lerobot/diffusion_pusht)

---

### SmolVLA

**Hugging Face's lightweight VLA** -- the smallest model with language understanding.

- **Architecture:** SmolVLM2-500M backbone (SigLIP vision + SmolLM2 language), with a ~100M flow matching action expert. Only uses the first N/2 VLM layers for efficiency. Each camera frame compressed to 64 visual tokens via PixelShuffle.
- **Input:** Up to 3 camera views (256x256) + 6-dim joint state + text instruction
- **Output:** 50-step action chunk (continuous joint positions)
- **Training:** ~4 hours on A100 for 20k steps. Can fine-tune on Google Colab.
- **Strengths:** Only ~2 GB VRAM for inference. Language-conditioned. Built-in async inference (gRPC server/client). Designed for SO-100/101.
- **Weaknesses:** Smaller model = less generalization than pi0 or GR00T. Newer, less battle-tested.

```bash
# Install
pip install "lerobot[smolvla]"

# Fine-tune
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your_user/your_dataset --steps=20000

# Deploy (local inference)
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --policy.path=your_user/my_smolvla \
  --dataset.single_task="pick up the red block"
```

#### Async inference (remote GPU)

SmolVLA's recommended production setup decouples inference from the robot. Uses gRPC instead of WebSocket.

```bash
# GPU server
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

# Robot side
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --robot.cameras="{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="pick up the block" \
    --policy_type=smolvla \
    --pretrained_name_or_path=lerobot/smolvla_base \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

Key tuning parameters:
- `actions_per_chunk` (10-50): larger = fewer idle frames but more compounding error
- `chunk_size_threshold` (0.0-1.0): when queue drops below this fraction, request new chunk. 0.5-0.6 recommended.

Links: [Paper](https://arxiv.org/abs/2506.01844) | [Model card](https://huggingface.co/lerobot/smolvla_base) | [Docs](https://huggingface.co/docs/lerobot/smolvla) | [Async docs](https://huggingface.co/docs/lerobot/en/async) | [Blog](https://huggingface.co/blog/smolvla)

---

### GR00T N1

**NVIDIA's foundation model for generalist robots.**

- **Architecture:** Dual-system design. System 2 (slow): frozen SigLIP2 + T5 VLM for perception. System 1 (fast): Diffusion Transformer (DiT) with flow matching for action generation. Cross-attention connects the two.
- **Versions:** N1 (2B, March 2025), N1.5 (3B), N1.6 (3B, 32-layer DiT + world-modeling objective)
- **Input:** Variable RGB camera views + joint state vector + text instruction. N1.5: 224x224 images. N1.6: native aspect ratio.
- **Output:** 8-16 step action chunk (shorter than pi0's 50)
- **Strengths:** Fast inference (27 Hz on RTX 5090). Good LIBERO benchmarks. Native LeRobot integration. Extensive fine-tuning docs for SO-101.
- **Weaknesses:** Non-commercial license. Requires flash-attn (CUDA only). Originally designed for humanoids, needs fine-tuning for tabletop arms.

```bash
# Install LeRobot with GR00T support
pip install -e ".[groot]"

# Fine-tune on your data
lerobot-train --policy.type=groot \
  --dataset.repo_id=your_user/your_dataset \
  --batch_size=32 --steps=20000

# Deploy
lerobot-record --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=your_user/groot_so101
```

#### Isaac-GR00T native server (alternative to LeRobot)

GR00T also has its own inference server, separate from OpenPI:

```bash
# Clone Isaac-GR00T
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T && uv sync --python 3.10

# Start server
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 --port 5555
```

GR00T requires a `modality.json` to define the state/action mapping:

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

Links: [GitHub](https://github.com/NVIDIA/Isaac-GR00T) | [N1.6-3B checkpoint](https://huggingface.co/nvidia/GR00T-N1.6-3B) | [LeRobot docs](https://huggingface.co/docs/lerobot/groot) | [SO-101 fine-tuning blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) | [Paper](https://arxiv.org/abs/2503.14734)

---

### pi0 / pi0.5

**Physical Intelligence's VLA** -- what this repo currently uses.

- **Architecture:** Single-model design. Vision + language + action processed in one forward pass with MoE-like attention. Flow matching for action generation.
- **Input:** Camera images (224x224) + joint state + text prompt. Observation keys depend on config (DROID, ALOHA, LIBERO).
- **Output:** 50-step action chunk
- **Serving:** OpenPI framework, WebSocket on port 8000
- **Strengths:** Strong generalization (pi0.5 trained on 10k+ hours). Multiple pre-trained configs.
- **Weaknesses:** ~14 GB VRAM. Not natively in LeRobot (uses separate OpenPI client).

See [`pi05/server/README.md`](../pi05/server/README.md) and [`pi05/client/README.md`](../pi05/client/README.md) for setup.

Links: [OpenPI repo](https://github.com/Physical-Intelligence/openpi) | [LeRobot pi0.5 docs](https://huggingface.co/docs/lerobot/pi05) | [Blog](https://www.physicalintelligence.company/blog/pi05)

---

## Non-LeRobot Models

These have their own codebases. Using them with SO-101 requires more integration work.

### OpenVLA

**Stanford's 7B VLA built on a Prismatic VLM backbone.**

- **Architecture:** Fused SigLIP + DinoV2 vision encoder, Llama 2 7B language model. Autoregressive action prediction (discrete tokens).
- **Input:** Single 224x224 RGB image + text instruction. OpenVLA-OFT adds multi-image + proprioceptive state.
- **Output:** 7-DoF end-effector deltas (x, y, z, roll, pitch, yaw, gripper)
- **Strengths:** OpenVLA-OFT achieves 97.1% on LIBERO (state-of-the-art). Outperforms pi0 on ALOHA tasks.
- **Weaknesses:** Outputs EEF deltas, not joint positions -- needs an inverse kinematics layer for SO-101. Large model (16-18 GB VRAM). Slow base inference (~5 Hz, OFT variant ~50 Hz).

Links: [Project](https://openvla.github.io/) | [GitHub](https://github.com/openvla/openvla) | [Model](https://huggingface.co/openvla/openvla-7b) | [OFT paper](https://arxiv.org/abs/2502.19645)

### MiniVLA

**OpenVLA compressed to 1B parameters** (Stanford ILIAD).

- 7x smaller, 2.5x faster inference (12.5 Hz vs 5 Hz on L40s). 82% on LIBERO-90.
- Uses VQ-BeT's Residual VQ for action chunking.

Links: [Blog](https://ai.stanford.edu/blog/minivla/) | [GitHub](https://github.com/Stanford-ILIAD/openvla-mini)

### Octo

**UC Berkeley's transformer-based generalist policy.**

- **Architecture:** ViT backbone (27M Small, 93M Base) + T5 language encoder + diffusion action head. Predicts 4-step action chunks.
- **Framework:** JAX (not PyTorch), so no LeRobot integration.
- **Input:** Primary image (256x256) + optional wrist image (128x128) + optional text.
- **Output:** 7-DoF actions, 4-step horizon.
- Fine-tunable to new embodiments, but requires RLDS data format conversion.

Links: [Project](https://octo-models.github.io/) | [GitHub](https://github.com/octo-models/octo) | [Model](https://huggingface.co/rail-berkeley/octo-base)

---

## Choosing a Model

### For getting started (first robot, first task)

Use **ACT**. It's the simplest, fastest to train, and has the most SO-101 tutorials. Collect 30-50 demonstrations of a single task, train for ~30 minutes, deploy.

### For language-conditioned tasks on a budget

Use **SmolVLA**. At ~2 GB VRAM it runs on consumer GPUs and even Apple Silicon. It understands text prompts, so one model can handle multiple tasks. The async inference architecture works well over a network.

### For maximum generalization

Use **pi0.5** (this repo's current setup) or **GR00T N1.6**. Both are ~3B parameter models trained on massive datasets. pi0.5 has the edge on real-world manipulation data; GR00T has better simulation coverage and faster inference.

### Decision flowchart

```
Do you need language conditioning (multi-task)?
├── No  → ACT (simplest) or Diffusion Policy (handles multi-modal actions)
└── Yes
    ├── Budget GPU (<8 GB)?  → SmolVLA (~2 GB)
    ├── Mid-range GPU (8-16 GB)? → GR00T N1.6 (~8-10 GB)
    └── Large GPU (16+ GB)?  → pi0.5 (~14 GB) or OpenVLA-OFT (~16 GB)
```

---

## Server Architecture Comparison

Each VLA model family has its own serving approach:

| Model | Server | Protocol | Port | Client library |
|-------|--------|----------|------|----------------|
| pi0/pi0.5 | OpenPI `serve_policy.py` | WebSocket | 8000 | `openpi_client` |
| GR00T N1.x | `run_gr00t_server.py` | Custom | 5555 | `gr00t.policy.server_client` |
| SmolVLA | `lerobot.async_inference.policy_server` | gRPC | 8080 | `lerobot.async_inference.robot_client` |
| ACT / Diffusion | In-process (no server) | N/A | N/A | `lerobot-record --policy.path=...` |

For remote GPU inference with SO-101 over a network:
- **pi0/pi0.5:** Use this repo's `client_pi.py` (WebSocket over Tailscale)
- **SmolVLA:** Use LeRobot's built-in async inference (gRPC)
- **GR00T:** Use Isaac-GR00T's `PolicyClient` or LeRobot's in-process approach
- **ACT/Diffusion:** These are small enough to run on the robot controller itself (no server needed)

---

## Observation Format Comparison

| Model | Image size | Image key format | State format | Prompt |
|-------|-----------|-----------------|--------------|--------|
| pi0/pi0.5 | 224x224 | `observation/image` | `observation/state` (flat array) | `prompt` |
| SmolVLA | 256x256 | `observation.images.camera1` | `observation.state` (6-dim) | Text string |
| GR00T N1 | 224x224 (N1.5) / native (N1.6) | `observation.images.front` | Per `modality.json` | Text string |
| ACT | Varies | Per dataset config | Per dataset config | N/A (task-specific) |
| OpenVLA | 224x224 | Single image tensor | N/A (image-only base) | Text string |

---

## GPU VRAM Requirements

| Model | Inference | LoRA fine-tune | Full fine-tune |
|-------|-----------|---------------|----------------|
| ACT | 4-6 GB | N/A | 6-8 GB (single GPU) |
| Diffusion Policy | 6-12 GB | N/A | 11+ GB |
| SmolVLA | ~2 GB | 8+ GB | ~16 GB |
| pi0 / pi0.5 | ~14 GB | >22.5 GB | >70 GB |
| GR00T N1.6 | ~8-10 GB | ~25 GB | Multi-GPU |
| OpenVLA | ~16-18 GB | ~26 GB | 8x A100 |

---

## Further Reading

- [Vision-Language-Action Models (Wikipedia)](https://en.wikipedia.org/wiki/Vision-language-action_model)
- [VLA overview (LearnOpenCV)](https://learnopencv.com/vision-language-action-models-lerobot-policy/)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [OpenPI GitHub](https://github.com/Physical-Intelligence/openpi)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [SO-101 setup guide (Seeed Studio)](https://wiki.seeedstudio.com/lerobot_so100m_new/)

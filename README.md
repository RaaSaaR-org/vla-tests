# VLA Test

Test open-source **Vision-Language-Action (VLA) models** on a GPU server and control an SO-101 robot arm remotely.

## What is this?

This project lets you run **VLA models** (neural networks that see camera images, read a text instruction, and output robot joint actions) on a powerful GPU machine, while the actual robot runs on a low-power device like a Raspberry Pi. The two communicate over the network so inference and control are fully decoupled.

We're testing multiple open-source VLA models -- see [docs/vla-models.md](docs/vla-models.md) for a full comparison. Currently set up for pi0/pi0.5 via OpenPI, with plans to add GR00T N1, SmolVLA, and others.

**Key terms:**

- **VLA (Vision-Language-Action):** A model that takes camera images + a language prompt (e.g. "pick up the cup") and predicts robot joint actions.
- **[OpenPI](https://github.com/Physical-Intelligence/openpi):** Physical Intelligence's open-source framework for serving pi0/pi0.5 models. We use it on the GPU server side.
- **[LeRobot](https://github.com/huggingface/lerobot):** Hugging Face's robotics framework. Provides drivers for SO-100/SO-101 arms and native support for SmolVLA, GR00T, ACT, and other policies.
- **[pi0.5](https://www.physicalintelligence.company/blog/pi05):** Physical Intelligence's VLA (~3B params), trained on 10k+ hours of robot data.
- **[SmolVLA](https://huggingface.co/lerobot/smolvla_base):** Hugging Face's lightweight VLA (450M params, ~2 GB VRAM). Built for LeRobot + SO-100/101.
- **[GR00T N1](https://github.com/NVIDIA/Isaac-GR00T):** NVIDIA's VLA (3B params). Dual-system architecture with fast DiT action head.

## Architecture

```
┌─────────────────────┐        WebSocket         ┌──────────────────────┐
│   GPU Server        │◄────────────────────────►│   Raspberry Pi       │
│                     │       (port 8000)         │                      │
│  openpi policy      │                           │  camera + robot arm  │
│  (pi0.5 / DROID)    │      ── Tailscale ──      │  client_pi.py        │
└─────────────────────┘                           └──────────────────────┘
```

1. The **GPU server** loads a VLA checkpoint via OpenPI and serves inference over WebSocket (port 8000).
2. The **Pi client** captures camera frames, reads joint state from the robot, bundles them into an observation dict, and sends it to the server.
3. The server returns an **action chunk** (a sequence of joint commands). The client executes them on the robot.
4. **[Tailscale](https://tailscale.com/)** (a peer-to-peer VPN) connects the two machines over a private network so you don't need to worry about port forwarding or firewalls.

## Hardware Requirements

| Component | What you need |
|-----------|---------------|
| **GPU server** | Linux machine with a GPU that has >8 GB VRAM (e.g. RTX 4090, RTX 5090, A100). Ubuntu 22.04+ recommended. |
| **Robot side** | Raspberry Pi (or any Linux machine) + camera (CSI or USB) + robot arm (e.g. [SO-101](https://wiki.seeedstudio.com/lerobot_so100m_new/)) |
| **Network** | Both machines on the same network, or connected via [Tailscale](https://tailscale.com/) |

## Quick Start

### 1. GPU Server

Set up OpenPI and serve a policy checkpoint. Full instructions in [`pi05/server/README.md`](pi05/server/README.md).

```bash
# Create environment with micromamba
micromamba create -n openpi python=3.11 uv -c conda-forge -y

# Clone OpenPI
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 micromamba run -n openpi uv sync

# Serve pi0.5 DROID checkpoint
micromamba run -n openpi uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

The server starts on port 8000. Checkpoints are auto-downloaded and cached in `~/.cache/openpi`.

### 2. Raspberry Pi (Robot Client)

Set up the client that talks to the GPU server. Full instructions in [`pi05/client/README.md`](pi05/client/README.md).

```bash
# Install Tailscale (if connecting over the internet)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install client dependencies
cd pi05/client
uv sync

# Run the client (--config must match the server policy)
uv run client_pi.py --host <GPU_SERVER_IP> --config droid --prompt "pick up the cup"
```

### 3. Verify Connection

On any machine that can reach the GPU server, you can test with OpenPI's built-in example client:

```bash
# From the openpi repo
micromamba run -n openpi uv run examples/simple_client/main.py --host <SERVER_IP> --env DROID
```

## Project Structure

```
vla-test/
├── README.md              ← You are here
├── CLAUDE.md              ← Context file for Claude Code
├── agents.md              ← How to SSH to the Pi and run commands from agents
├── docs/
│   └── vla-models.md      ← Comparison of all VLA models
├── pi05/                  ← pi0/pi0.5 via OpenPI (WebSocket)
│   ├── client/
│   │   ├── client_pi.py   ← Main client (SO-101 + camera + control loop)
│   │   ├── pyproject.toml
│   │   └── README.md
│   └── server/
│       └── README.md
├── groot/                 ← NVIDIA GR00T N1.6 via Isaac-GR00T / LeRobot
│   ├── client/
│   │   └── README.md      ← PolicyClient or LeRobot in-process setup
│   └── server/
│       └── README.md      ← run_gr00t_server.py setup + fine-tuning
└── smolvla/               ← HuggingFace SmolVLA via LeRobot async (gRPC)
    ├── client/
    │   └── README.md      ← robot_client setup + Python API example
    └── server/
        └── README.md      ← policy_server setup + fine-tuning
```

## How It Works (Step by Step)

1. **Server starts** — OpenPI loads the VLA model weights into GPU memory and listens for WebSocket connections.
2. **Client connects** — `client_pi.py` opens a WebSocket to the server.
3. **Each control loop iteration:**
   - Capture an image from the USB camera
   - Read the current joint positions from the robot
   - Resize the image to 224x224 (what the model expects)
   - Pack everything into an observation dict with a text prompt
   - Send it to the server
   - Receive back an **action chunk** — a sequence of predicted joint positions
   - Execute the first action (or all actions in `--chunked` mode)
4. **Repeat** at the configured frequency (default 5 Hz).

## Observation Keys by Config

Different policy configs expect different observation keys. The client's observation dict must match what the server config expects:

| Config | Image keys | State key | Notes |
|--------|-----------|-----------|-------|
| `pi05_droid` / `pi0_fast_droid` | `observation/exterior_image_1_left`, `observation/wrist_image_left` | `observation/joint_position` + `observation/gripper_position` | 7 joints + 1 gripper |
| `pi0_aloha` | `images/cam_high`, `images/cam_low`, `images/cam_left_wrist`, `images/cam_right_wrist` | `state` | 14 joints |
| `pi0_libero` | `observation/image`, `observation/wrist_image` | `observation/state` | 8-dim state |

All configs also accept a `prompt` key (string) for the language instruction.

## VLA Model Comparison

See [docs/vla-models.md](docs/vla-models.md) for a detailed comparison of all models, including setup instructions, GPU requirements, and a decision flowchart for choosing the right one.

| Model | Params | VRAM | Language | LeRobot native |
|-------|--------|------|----------|----------------|
| ACT | ~80M | 4-6 GB | No | Yes |
| SmolVLA | 450M | ~2 GB | Yes | Yes |
| GR00T N1.6 | 3B | ~8-10 GB | Yes | Yes |
| pi0.5 | ~3B | ~14 GB | Yes | Via OpenPI |
| OpenVLA | 7B | ~16 GB | Yes | No |

## Useful Links

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi) — pi0/pi0.5 server framework
- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face robotics framework (SmolVLA, GR00T, ACT)
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA's GR00T model and tools
- [SmolVLA model card](https://huggingface.co/lerobot/smolvla_base) — Hugging Face's lightweight VLA
- [pi0.5 in LeRobot docs](https://huggingface.co/docs/lerobot/pi05) — model details, fine-tuning, benchmarks
- [SO-100 robot arm guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) — hardware assembly & LeRobot setup
- [Physical Intelligence blog](https://www.physicalintelligence.company/blog/pi05) — pi0.5 research paper & motivation
- [Agents guide](agents.md) — how to SSH to the Pi and run commands from AI agents

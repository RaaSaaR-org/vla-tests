# VLA Test

Run a [Physical Intelligence](https://www.physicalintelligence.company/) **pi0 / pi0.5** vision-language-action (VLA) model on a GPU server and control a robot arm remotely over WebSocket.

## What is this?

This project lets you run a **VLA model** (a neural network that sees camera images, reads a text instruction, and outputs robot joint actions) on a powerful GPU machine, while the actual robot runs on a low-power device like a Raspberry Pi. The two communicate over WebSocket so inference and control are fully decoupled.

**Key terms:**

- **VLA (Vision-Language-Action):** A model that takes camera images + a language prompt (e.g. "pick up the cup") and predicts robot joint actions.
- **[OpenPI](https://github.com/Physical-Intelligence/openpi):** Physical Intelligence's open-source framework for serving pi0/pi0.5 models. We use it on the GPU server side.
- **[LeRobot](https://github.com/huggingface/lerobot):** Hugging Face's robotics framework. Compatible robot arms (like the [SO-100](https://wiki.seeedstudio.com/lerobot_so100m_new/)) can be controlled through it.
- **pi0.5:** The latest model in the family, trained on 10k+ hours of robot data with improved open-world generalization across physical, semantic, and environmental levels.

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
| **GPU server** | Linux machine with a GPU that has >8 GB VRAM (e.g. RTX 4090, A100). Ubuntu 22.04 recommended. |
| **Robot side** | Raspberry Pi (or any Linux machine) + USB camera + robot arm (e.g. [SO-100](https://wiki.seeedstudio.com/lerobot_so100m_new/)) |
| **Network** | Both machines on the same network, or connected via [Tailscale](https://tailscale.com/) |

## Quick Start

### 1. GPU Server

Set up OpenPI and serve a policy checkpoint. Full instructions in [`pi05/server/README.md`](pi05/server/README.md).

```bash
# Clone OpenPI
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Serve pi0.5 DROID checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
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

# Run the client
uv run client_pi.py --host <GPU_SERVER_IP> --prompt "pick up the cup"
```

### 3. Verify Connection

On any machine that can reach the GPU server, you can test with OpenPI's built-in example client:

```bash
# From the openpi repo
uv run examples/simple_client/main.py --host <SERVER_IP> --env DROID
```

## Project Structure

```
vla-test/
├── README.md              ← You are here
├── CLAUDE.md              ← Context file for Claude Code
└── pi05/
    ├── client/            ← Runs on the Raspberry Pi
    │   ├── client_pi.py   ← Main client (camera, robot, control loop)
    │   ├── pyproject.toml ← Python dependencies (uses uv)
    │   └── README.md      ← Detailed client setup & customization guide
    └── server/
        └── README.md      ← GPU server setup instructions
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

## Useful Links

- [OpenPI repository](https://github.com/Physical-Intelligence/openpi) — the server-side framework
- [pi0.5 in LeRobot docs](https://huggingface.co/docs/lerobot/pi05) — model details, fine-tuning, benchmarks
- [SO-100 robot arm guide](https://wiki.seeedstudio.com/lerobot_so100m_new/) — hardware assembly & LeRobot setup
- [Physical Intelligence blog](https://www.physicalintelligence.company/blog/pi05) — pi0.5 research paper & motivation

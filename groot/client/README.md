# GR00T N1 — Robot Client (SO-101)

Runs on the robot side. Connects to the GR00T inference server, sends camera frames + joint state, executes returned actions on the SO-101 arm.

Two pathways: **Isaac-GR00T PolicyClient** (connects to `run_gr00t_server.py`) or **LeRobot in-process** (runs inference locally, no server needed).

## Prerequisites

- Raspberry Pi (or any Linux machine) with USB camera
- SO-101 robot arm connected via USB
- Python 3.10+
- Network access to GPU server (direct or via Tailscale)

## Option A: LeRobot In-Process (simplest)

If your GPU machine is the same machine or you want LeRobot to handle everything:

```bash
pip install "lerobot[groot]"

# Deploy a fine-tuned model directly
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so101 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --dataset.single_task="pick up the red block" \
  --policy.path=your_user/groot_so101
```

## Option B: Isaac-GR00T PolicyClient (remote GPU)

For the split architecture (GPU server + Pi client over network):

```bash
pip install gr00t-client  # or install from Isaac-GR00T repo
```

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

# Connect to the GPU server
client = PolicyClient(host="<GPU_SERVER_IP>", port=5555)

# Build observation dict (keys must match modality.json)
obs = {
    "video.front": image_array,          # (H, W, 3) uint8 RGB
    "state.single_arm": joint_positions,  # (6,) float32
    "state.gripper": gripper_position,    # (1,) float32
    "annotation": "pick up the cup",
}

# Get action chunk
action, info = client.get_action(obs)
# action shape: (horizon, action_dim) — e.g. (16, 7)
```

## Observation Format

GR00T expects observations matching the `modality.json` used during fine-tuning:

| Key | Shape | Description |
|-----|-------|-------------|
| `video.front` | (H, W, 3) uint8 | RGB camera frame. N1.5: resize to 224x224. N1.6: native resolution OK. |
| `state.single_arm` | (6,) float32 | Joint positions (degrees) for 6 arm joints |
| `state.gripper` | (1,) float32 | Gripper position |
| `annotation` | string | Natural language task instruction |

## Action Format

| Key | Shape | Description |
|-----|-------|-------------|
| `action.single_arm` | (horizon, 6) | Arm joint position commands |
| `action.gripper` | (horizon, 1) | Gripper commands |

Action horizon is typically 8 or 16 steps (configurable via `--denoising-steps` on the server).

## Key Differences from pi0.5 Client

| Aspect | pi0.5 (`client_pi.py`) | GR00T |
|--------|------------------------|-------|
| Server port | 8000 (WebSocket) | 5555 (custom protocol) |
| Client library | `openpi_client` | `gr00t.policy.server_client` |
| Image resize | Always 224x224 with padding | N1.5: 224x224, N1.6: native |
| Observation keys | `observation/image`, `observation/state` | `video.front`, `state.single_arm` |
| Action chunk size | 50 steps | 8-16 steps |

## Further Reading

- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/groot)
- [SO-101 fine-tuning blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)

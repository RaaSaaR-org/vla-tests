# SmolVLA — Robot Client (SO-101)

Runs on the robot side. Connects to the SmolVLA policy server via gRPC, sends camera frames + joint state, executes returned actions on the SO-101 arm.

SmolVLA uses LeRobot's built-in async inference — no custom client code needed.

## Prerequisites

- Raspberry Pi (or any Linux machine) with USB camera
- SO-101 robot arm connected via USB
- Python 3.10+
- Network access to GPU server (direct or via Tailscale)

## Install

```bash
pip install "lerobot[feetech,async]"
```

## First-Time Robot Setup

Same as pi0.5 — calibrate your SO-101 with LeRobot's CLI tools:

```bash
# Find USB port
lerobot-find-port

# Set up motors
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0

# Calibrate
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_so101
```

## Usage: Async Inference (remote GPU)

Make sure the policy server is running on your GPU machine (see `smolvla/server/README.md`).

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_SERVER_IP>:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
    --task="pick up the red block" \
    --policy_type=smolvla \
    --pretrained_name_or_path=lerobot/smolvla_base \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

### Client options

| Flag | Default | Description |
|------|---------|-------------|
| `--server_address` | — | GPU server `host:port` (e.g. `192.168.1.100:8080`) |
| `--robot.type` | — | Robot type (`so101_follower`) |
| `--robot.port` | — | USB serial port (e.g. `/dev/ttyACM0`) |
| `--robot.id` | — | Calibration identity (must match calibration) |
| `--robot.cameras` | — | Camera config (YAML dict, keys must match model training) |
| `--task` | — | Natural language instruction |
| `--policy_type` | — | Policy type (`smolvla`, `act`, `groot`, etc.) |
| `--pretrained_name_or_path` | — | Model ID on Hub or local path |
| `--policy_device` | `cuda` | Device for inference on the server (`cuda`, `mps`, `cpu`) |
| `--actions_per_chunk` | `50` | Actions per inference call (10-50) |
| `--chunk_size_threshold` | `0.7` | Request new chunk when queue drops below this fraction (0.5-0.6 recommended) |
| `--aggregate_fn_name` | `weighted_average` | How to blend overlapping action chunks |

### Tuning tips

- **Robot stops moving between chunks?** Lower `chunk_size_threshold` (e.g. 0.3) or increase `actions_per_chunk`
- **Actions feel stale/laggy?** Raise `chunk_size_threshold` (e.g. 0.7) to request fresh chunks more often
- **Debug:** Add `--debug_visualize_queue_size=True` to plot the action queue in real-time

## Usage: Local Inference (no server)

If the GPU is on the same machine as the robot:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so101 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --dataset.single_task="pick up the red block" \
  --policy.path=your_user/smolvla_so101
```

## Usage: Python API

```python
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

device = torch.device("cuda")  # or "mps" or "cpu"
model_id = "lerobot/smolvla_base"

# Load model
model = SmolVLAPolicy.from_pretrained(model_id)
preprocess, postprocess = make_pre_post_processors(
    model.config, model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Connect robot
camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
}
robot_cfg = SO101FollowerConfig(port="/dev/ttyACM0", id="my_so101", cameras=camera_config)
robot = SO101Follower(robot_cfg)
robot.connect()

# Map features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Control loop
task = "pick up the red block"
while True:
    obs = robot.get_observation()
    obs_frame = build_inference_frame(
        observation=obs, ds_features=dataset_features,
        device=device, task=task, robot_type="so101_follower",
    )
    batch = preprocess(obs_frame)
    action = model.select_action(batch)
    action = postprocess(action)
    action = make_robot_action(action, dataset_features)
    robot.send_action(action)
```

## Observation Format

SmolVLA expects:

| Key | Shape | Description |
|-----|-------|-------------|
| `observation.images.camera1` | (3, 256, 256) float32 | RGB, normalized 0-1. Camera key must match model config. |
| `observation.state` | (6,) float32 | Joint positions (6 joints for SO-101) |
| Task string | text | Natural language instruction |

Camera keys in the model config (e.g. `camera1`, `camera2`) **must match** your `--robot.cameras` keys. Check the model's `config.json` on HuggingFace to see expected keys.

## Key Differences from pi0.5 Client

| Aspect | pi0.5 (`client_pi.py`) | SmolVLA |
|--------|------------------------|---------|
| Protocol | WebSocket | gRPC |
| Server port | 8000 | 8080 |
| Client code | Custom `client_pi.py` | Built-in `lerobot.async_inference.robot_client` |
| Image size | 224x224 | 256x256 (internally resized to 512x512) |
| State dims | Config-dependent | 6 (SO-101 joints) |
| Action chunk | 50 steps | 50 steps (configurable) |
| VRAM needed | ~14 GB | ~2 GB |

## Further Reading

- [SmolVLA paper](https://arxiv.org/abs/2506.01844)
- [SmolVLA LeRobot docs](https://huggingface.co/docs/lerobot/smolvla)
- [Async inference docs](https://huggingface.co/docs/lerobot/en/async)
- [SmolVLA blog](https://huggingface.co/blog/smolvla)

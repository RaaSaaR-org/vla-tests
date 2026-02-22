# Agents Guide

How to interact with the hardware in this project from an AI agent (Claude Code, etc.) or a remote development machine.

## Machines

| Machine | Role | Tailscale IP | Hostname | OS |
|---------|------|-------------|----------|------|
| GPU server | Policy inference (RTX 5090) | `100.125.78.40` | `peter-ubuntu` | Ubuntu (Linux) |
| Raspberry Pi 5 | Robot arm + camera | `100.84.200.83` | — | Raspberry Pi OS |

Both machines are on a [Tailscale](https://tailscale.com/) mesh network. You can reach either from any machine on the same Tailnet.

## SSH Access

### Raspberry Pi

```bash
ssh mindcube@100.84.200.83
```

No password required (key-based auth). The user is `mindcube`.

### GPU Server

The GPU server is the local machine where Claude Code runs (`peter-ubuntu`). No SSH needed — just run commands directly.

## Raspberry Pi Layout

| Path | Description |
|------|-------------|
| `~/repos/vla-tests/pi05/client/` | Deployed client code + venv |
| `~/develop/backup/robot-calibration-data.json` | SO-101 servo calibration (6 joints) |
| `~/develop/backup/server.py` | Old telemetry server (reference for LeRobot API patterns) |
| `/dev/ttyACM0` | SO-101 robot USB serial port |

### Cameras on the Pi

The Pi has **CSI cameras** (not USB). OpenCV `VideoCapture` does not work with them — you must use `picamera2`.

| Index | Sensor | Notes |
|-------|--------|-------|
| 0 | IMX477 | HQ camera (main, used for inference) |
| 1 | OV5647 | Standard camera module |

To test cameras:

```bash
ssh mindcube@100.84.200.83
# Quick test — capture a JPEG
python3 -c "
from picamera2 import Picamera2
cam = Picamera2(0)
cam.start()
import time; time.sleep(1)
cam.capture_file('/tmp/test.jpg')
cam.close()
print('Saved /tmp/test.jpg')
"
```

## Running the Client on the Pi

```bash
ssh mindcube@100.84.200.83

cd ~/repos/vla-tests/pi05/client

# Activate the venv (created with --system-site-packages for picamera2 access)
source .venv/bin/activate

# Run the client (server must already be running on GPU machine)
python client_pi.py \
    --port /dev/ttyACM0 \
    --host 100.125.78.40 \
    --config droid \
    --prompt "pick up the green ship"
```

Or without activating the venv:

```bash
cd ~/repos/vla-tests/pi05/client
uv run client_pi.py --port /dev/ttyACM0 --host 100.125.78.40 --config droid --prompt "pick up the green ship"
```

## Running the Server on the GPU Machine

```bash
# From the GPU server (peter-ubuntu), in the openpi repo
cd /home/peter/repos/openpi

micromamba run -n openpi uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

The server listens on port 8000. Checkpoints are cached in `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid` (11.6 GB).

To run in background:

```bash
micromamba run -n openpi uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_drod \
    > /tmp/openpi-server.log 2>&1 &
echo "Server PID: $!"
```

## Running Commands on the Pi from an Agent

When Claude Code (or another agent) needs to run something on the Pi, use SSH:

```bash
ssh mindcube@100.84.200.83 "command here"
```

Examples:

```bash
# Check if the robot serial port is available
ssh mindcube@100.84.200.83 "ls -la /dev/ttyACM*"

# Check running Python processes
ssh mindcube@100.84.200.83 "ps aux | grep python"

# Pull latest code on the Pi
ssh mindcube@100.84.200.83 "cd ~/repos/vla-tests && git pull"

# Check camera status
ssh mindcube@100.84.200.83 "libcamera-list"

# Install/sync client dependencies
ssh mindcube@100.84.200.83 "cd ~/repos/vla-tests/pi05/client && uv sync"
```

For interactive or long-running tasks, SSH in with a TTY:

```bash
ssh -t mindcube@100.84.200.83 "cd ~/repos/vla-tests/pi05/client && uv run client_pi.py --port /dev/ttyACM0 --host 100.125.78.40 --config droid --prompt 'pick up the cup'"
```

## Troubleshooting

### Motor 5 (wrist_roll) fails handshake

The wrist_roll servo (ID 5) sometimes fails to respond during connection. This is likely a loose cable. Retry connecting — it usually works on the second attempt.

### "Missing calibration" error

The client auto-searches for calibration in several locations (see `client_pi.py` `_load_calibration`). On the Pi, the file is at `~/develop/backup/robot-calibration-data.json`. You can also pass `--calibration-file` explicitly.

### picamera2 import fails in venv

The venv must be created with `uv venv --system-site-packages` so it can access the system `picamera2` and `libcamera` bindings. These cannot be pip-installed.

### numpy ABI mismatch with simplejpeg

The system `simplejpeg` is compiled against numpy 1.x. If the venv has numpy 2.x, you get ABI errors. Fix: `pip install simplejpeg` inside the venv to get a version compiled against numpy 2.x.

### Server not reachable from Pi

Check Tailscale is up on both machines:

```bash
# On Pi
tailscale status

# Test connectivity
ping 100.125.78.40
curl http://100.125.78.40:8000
```

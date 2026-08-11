# Voicute Wyoming Service

Wyoming protocol server for [Voicute](https://github.com/voicute/onnx-wakeword) wake word models.
Compatible with **Home Assistant** Wyoming integration.

## Quick Start

```bash
# Install
pip install onnxruntime numpy

# Run
python wyoming_voicute.py \
    --model-info models/model_info.json \
    --mel models/melspectrogram.onnx \
    --preload "Hey Friday"
```

## Home Assistant Setup

1. Start the service:
   ```bash
   python wyoming_voicute.py \
       --uri tcp://0.0.0.0:10400 \
       --model-info /path/to/model_info.json \
       --mel /path/to/melspectrogram.onnx \
       --threshold 0.4
   ```

2. In Home Assistant → Settings → Voice Assistants → Add Wyoming:
   - Host: IP of this machine
   - Port: 10400

3. Select "Voicute" as the wake word engine in your voice pipeline.

## Options

| Flag | Default | Description |
|------|:------:|-------------|
| `--uri` | `tcp://0.0.0.0:10400` | Listen address |
| `--model-info` | (required) | Path to `model_info.json` |
| `--mel` | (required) | Path to `melspectrogram.onnx` |
| `--preload` | — | Wake words to advertise to HA |
| `--threshold` | 0.40 | Detection threshold (0–1) |
| `--cooldown` | 1500 | Cooldown in ms |
| `--L1 / --L3 / --L5` | 1 / 1 / 0 | Anti-false-trigger layers |
| `--debug` | off | Verbose logging |

## Docker

```bash
docker run --rm --network host \
    -v $(pwd)/models:/models \
    voicute/wyoming \
    --model-info /models/model_info.json \
    --mel /models/melspectrogram.onnx \
    --preload "Hey Friday"
```

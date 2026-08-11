# Voicute Wake Word — Home Assistant Add-on

Custom wake word detection for Home Assistant, powered by Voicute.

## Features

- **Multi-keyword** — detect multiple wake words with a single model
- **Multi-language** — Chinese, English, and 150+ languages supported
- **Small models** — ~128KB per keyword, runs on any hardware
- **5-layer anti-false-trigger** — consecutive frames, cooldown, energy jump, and more
- **Wyoming protocol** — native Home Assistant integration, auto-discovered

## Setup

1. Install this add-on
2. Upload your `model_info.json` and `.onnx` model files to `/data/` (or `/share/voicute/`)
3. Configure the paths in the add-on options
4. Start the add-on
5. In Home Assistant → Settings → Voice assistants → Add Wyoming, it should auto-discover

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `model_info` | `/data/model_info.json` | Path to model configuration |
| `mel` | `/data/melspectrogram.onnx` | Path to mel spectrogram ONNX |
| `threshold` | `0.4` | Detection threshold (0–1) |
| `cooldown` | `1500` | Cooldown in ms between triggers |
| `L1` | `true` | Consecutive frames filter |
| `L3` | `true` | Cooldown filter |
| `L5` | `false` | Energy jump filter |
| `debug` | `false` | Verbose logging |

## Getting Models

Generate custom wake word models at [voicute.com](https://www.voicute.com).

## Support

- GitHub: [github.com/voicute/onnx-wakeword](https://github.com/voicute/onnx-wakeword)
- Docker Hub: [voicute/voicute-wyoming](https://hub.docker.com/r/voicute/voicute-wyoming)

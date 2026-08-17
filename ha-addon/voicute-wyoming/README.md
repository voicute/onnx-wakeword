# Voicute Wake Word — Home Assistant Add-on

Custom wake word detection for Home Assistant, powered by Voicute.

## Features

- **Multi-keyword** — detect multiple wake words with a single model
- **Multi-language** — Chinese, English, and 150+ languages supported
- **Small models** — ~128KB per keyword, runs on any hardware
- **5-layer anti-false-trigger** — consecutive frames, cooldown, energy jump, and more
- **Wyoming protocol** — native Home Assistant integration, auto-discovered

## Setup

1. Add this repository to Home Assistant (Supervisor → Add-on store → ⋮ → Repositories)
2. Install the "Voicute Wake Word" add-on and start it
3. It ships with a demo keyword (`hey limi`) and the universal mel model, so it works out of the box
4. In Home Assistant → Settings → Devices & services → Add Integration → **Wyoming Protocol**, enter the HA host IP and port `10400`
5. Pick "Voicute" as the wake word engine in your voice assistant's wake-word configuration

> **Custom keyword**: upload your `model_info.json` + `.onnx` model to `/data/` (via the Samba/SSH/File editor add-ons), then set the `model_info` option to `/data/model_info.json`.

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `model_info` | `/app/models/model_info.json` | Bundled demo model; point to `/data/...` for custom |
| `mel` | `/app/models/melspectrogram.onnx` | Universal mel model (bundled, no upload needed) |
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

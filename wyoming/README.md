# Voicute Wyoming Service

Wyoming protocol server for [Voicute](https://github.com/voicute/onnx-wakeword) wake word models.
Compatible with **Home Assistant** Wyoming integration.

## Demo models

The repo ships demo models under [`../models/`](../models/) — **hey limi** (zh), **Hey Friday** (en), **曼波** (zh), **Apfelstrudel** (de), **Monsieur Sadin / Croissant** (fr). Full list + changelog: [`../models/README.md`](../models/README.md). Get your own keyword model at [voicute.com](https://www.voicute.com).

### Multi-keyword models

Two multi-keyword configs are also provided — a **single model** that detects several wake words at once (`model_type: "multi_keyword"`):

| Config | Wake words | Model |
| ------ | ---------- | ----- |
| `models/model_info_multi.json` | 小娜 · 你好小娜 · 小娜小娜 | `zh/multi_xiaona.onnx` |
| `models/model_info_multi martina.json` | Martina · Tina · Hey Tina | `de/multi_martina.onnx` |

Start one by pointing `--model-info` at it:

```bash
# Chinese multi-keyword: 小娜 / 你好小娜 / 小娜小娜
python wyoming/wyoming_voicute.py \
    --model-info models/model_info_multi.json \
    --mel models/melspectrogram.onnx

# German multi-keyword: Martina / Tina / Hey Tina
# (note: quote the path — the config filename contains a space)
python wyoming/wyoming_voicute.py \
    --model-info "models/model_info_multi martina.json" \
    --mel models/melspectrogram.onnx
```

## Quick Start

From the repo root:

```bash
pip install onnxruntime numpy

python wyoming/wyoming_voicute.py \
    --model-info models/model_info.json \
    --mel models/melspectrogram.onnx
```

This starts the service with the bundled demo keyword (**hey limi**) on `tcp://0.0.0.0:10400`. For your own model, point `--model-info` at its `model_info.json` and keep the `.onnx` files beside it.

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

> **Note:** the service does not advertise itself via mDNS yet, so Home Assistant will **not** auto-discover it — add it manually with the host IP and port above.

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

### Pull from Docker Hub (recommended)

```bash
docker pull voicute/voicute-wyoming:latest

docker run -d --name voicute-wakeword --restart unless-stopped --network host \
    -v /path/to/your/models:/models \
    voicute/voicute-wyoming:latest \
    --model-info /models/model_info.json --mel /app/models/melspectrogram.onnx
```

The image bundles `melspectrogram.onnx` (and a demo `model_info.json`) under `/app/models/`, so you only need to mount your own `model_info.json` + keyword `.onnx`.

### Or build locally

```bash
cd onnx-wakeword
docker build -t voicute/voicute-wyoming .
```

> **Note for users in China:** If `docker build` fails with network errors, disable BuildKit:
> ```bash
> DOCKER_BUILDKIT=0 docker build -t voicute/voicute-wyoming .
> ```

### docker-compose

```bash
docker compose up -d
```

Edit `docker-compose.yml` to set your model path, then in Home Assistant add a Wyoming service at `host:10400`.

## Home Assistant Add-on

A Supervisor add-on is provided under [`../ha-addon/`](../ha-addon/) for HAOS / Supervised installs. Add the repository `https://github.com/voicute/ha-addons` to the add-on store, install **"Voicute Wake Word"**, and start it — it ships with the demo keyword and the universal mel model, so it works out of the box. See [`../ha-addon/voicute-wyoming/README.md`](../ha-addon/voicute-wyoming/README.md).

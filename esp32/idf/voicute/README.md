# Voicute ESP-IDF component

English | [中文](README_CN.md)

Reusable offline wake-word / keyword-spotting runtime for ESP-IDF, published to the [Espressif Component Registry](https://components.espressif.com) as **`voicute/voicute`**:

```yaml
# your project's main/idf_component.yml
dependencies:
  voicute/voicute: "^1.0.0"
```

Or copy this directory into your project's `components/`, or add it through `EXTRA_COMPONENT_DIRS`. A copyable starter project lives in [`examples/minimal/`](examples/minimal/); a full board-integrated example in [`../examples/esp32s3_hmi_devkit/`](../examples/).

## What it provides

16 kHz PCM-to-Mel processing, INT8 TensorFlow Lite Micro inference (ESP-NN accelerated), a configurable model-head callback, optional L1–L5 detection logic, and multi-model loading from SPIFFS. See [API.md](API.md).

## What it deliberately does NOT include

- **No models** — bring your own `.tflite` (from [voicute.com](https://www.voicute.com) or your own export), either flashed to SPIFFS or compiled into firmware as a C array
- **No microphone driver** — you feed 16 kHz / mono / int16 PCM yourself, one `MEL_AUDIO_LEN` (16080-sample) window per inference tick, plus its RMS
- **No board BSP** — no codecs (ES7210 etc.), LEDs, or buttons
- **No model head weights** — the `postprocess` callback is caller-supplied; it converts the INT8 backbone output into a probability and must match your model's trained head (see `kws_postprocess.h` + `head_stub.h` in [`examples/minimal/`](examples/minimal/))

## Verified targets

- **ESP32-S3**: fully hardware-verified (all on-board validation in this repo is S3)
- Other targets (incl. ESP32-P4): code path available, **not hardware-verified** — no support claim

## ESP-IDF version

Built and verified with **ESP-IDF v6.0** (`CMake`-based, component-manager dependencies).

## L1–L5 detection defaults on ESP32

Inference on MCU is much slower than on phone/desktop, so the defaults are leaner than the mobile/web profile:

| Layer | Purpose | ESP32 default |
|:-----:|---------|:-------------:|
| L1 | consecutive frames above threshold | **off** (frames arrive too slowly) |
| L2 | peak ≫ background | on |
| L3 | 1.5 s cooldown | on |
| L4 | burst suppression | off |
| L5 | energy jump (music/video filter, post-speech confirm) | on, `l5_delta = 200` |

Tune via `recognizer_config_t`.

## License

MIT.

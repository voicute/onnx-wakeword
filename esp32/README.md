# Voicute for ESP32

English | [中文](README_CN.md)

Voicute is an offline custom wake-word SDK for ESP32. The reusable inference component is separated from the board-specific runnable example.

## Repository layout

```text
esp32/
├── sdk/voicute/                    Reusable ESP-IDF component
└── examples/esp32s3_hmi_devkit/    Complete tested application
```

The SDK accepts 16 kHz PCM and performs Mel extraction, INT8 TFLite inference, model-head post-processing, and wake-word detection. Microphone, codec, LED, ESP-SR AFE, and MultiNet handling belong to the example.

## Tested platform

| Item | Tested configuration |
|---|---|
| SoC | ESP32-S3, dual core, 240 MHz |
| Board | ESP32-S3-HMI-DevKit |
| Memory | 8 MB octal PSRAM, 16 MB flash |
| Audio | 4 MEMS microphones through ES7210; ES8311 output; 16 kHz PCM |
| Framework | ESP-IDF 6.0.1 |
| Runtime | esp-tflite-micro 1.4.1 + ESP-NN 1.4.1; ESP-SR MultiNet5 English |

Other ESP32-S3 boards can use the SDK after adapting their microphone/codec BSP and memory configuration. Other ESP32 chip families are not yet validated by this example.

## Included English demo

The matched model and head detect **Hey Robot**. After wake-up, say:

- Turn the light red
- Turn the light blue
- Turn the light green
- Turn the light white

See the [example guide](examples/esp32s3_hmi_devkit/README.md) or the [SDK guide](sdk/voicute/README.md).

## Measured performance

With the matched 98-frame model on the tested ESP32-S3 at 240 MHz (esp-tflite-micro 1.4.1 + ESP-NN 1.4.1), Mel extraction is approximately 29 ms. TFLite `Invoke()` measures about 39.5–40.2 ms in isolation; in the live demo app (ESP-SR AFE feed and MultiNet running concurrently) the wall-clock `Invoke()` is about 51 ms. Both bundled 98-frame models (the reference `hey_robot` and `doubaodoubao`) were verified on-board with the same results. Results vary by model, clock, and target.

## Important notes

- **Always swap `head.h` together with the model**: the classifier weights and calibration constants (`KWS_ABS_TEMP` / `KWS_FC_B`) in `main/head.h` are matched to the `.tflite` in `spiffs_content/`; replacing only one side produces wrong scores.
- **Enlarge the tensor arena when raising the window length**: `TFLITE_ARENA_SIZE` in `sdk/voicute/model_loader.h` is fixed at 64 KB, while the bundled 98-frame model uses about 43 KB. `KWS_MEL_TIME` in `main/head.h` is configurable (defaults to 98 when undefined); if you raise it (e.g. 200 frames for long wake words), scale the arena accordingly, otherwise model loading fails for lack of memory.
- **A mel_time mismatch refuses to load**: if the model's input frame count differs from `KWS_MEL_TIME`, startup fails fast with an error — there is no silent fallback. This is intentional, so model/firmware mismatches surface immediately.
- **Component downloads need the network**: `idf.py set-target` re-solves component versions; `dependencies.lock` pins them for reproducibility. On machines that reach the internet through a system proxy, do not set `no_proxy=*` or clear proxy variables in build scripts — the registry lookup then fails silently ("Component not found" / "no versions match").
- **Run `patch_led_strip.py` after `set-target`**: set-target re-downloads `managed_components/` and wipes the patch. `build.bat` already does this in the right order; don't skip the step in manual builds.

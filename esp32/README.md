# Voicute for ESP32

English | [中文](README_CN.md)

Voicute is an offline custom wake-word SDK for ESP32. The reusable inference component is separated from the board-specific runnable example.

## Repository layout

```text
esp32/
├── sdk/voicute/                    Reusable ESP-IDF component
└── examples/esp32s3_hmi_devkit/    Complete tested application
```

The SDK accepts 16 kHz PCM and performs Mel extraction, INT8 TFLite inference, model-head post-processing, and wake-word detection. Microphone, codec, LED, and ESP-SR AFE handling belong to the example.

## Tested platform

| Item | Tested configuration |
|---|---|
| SoC | ESP32-S3, dual core, 240 MHz |
| Board | ESP32-S3-HMI-DevKit |
| Memory | 8 MB octal PSRAM, 16 MB flash |
| Audio | 4 MEMS microphones through ES7210; ES8311 output; 16 kHz PCM |
| Framework | ESP-IDF 6.0.1 |
| Runtime | esp-tflite-micro 1.4.1 + ESP-NN 1.4.1; ESP-SR AFE (MultiNet removed — commands now run on our own KWS model) |

Other ESP32-S3 boards can use the SDK after adapting their microphone/codec BSP and memory configuration. Other ESP32 chip families are not yet validated by this example.

## Included demo (German, two own models)

The example runs two of our own KWS models time-divided on one board
(MultiNet fully removed):

1. Say **Hallo Lumo** to wake (single-head MultiProto, K=5).
2. Within 6 s say **Licht rot / Licht blau / Licht grün / Licht weiß**
   (multi-head, N=4 × K=3).
3. The LED shows the colour and the app returns to wake-word mode.

Both models are mel_time=150 exports (1.532 s window, 32 mels) and verified
on-board: 20+ wakes (prob 0.73–1.000) in noisy (rms 150–190) and quiet
(rms 24–36) rooms, all four colour commands hit with zero confusion.

See the [example guide](examples/esp32s3_hmi_devkit/README.md) — including the
model directory layout and how the two `head.h` files are merged — or the
[SDK guide](sdk/voicute/README.md).

## Model packages (`onnx-wakeword/models/<lang>/`)

Each shipped model is a zip containing the complete pair (backbone + head), so
a package alone is enough to integrate:

```text
models/de/
├── hallolumo_r1_1_x654af9_tflite.zip   "Hallo Lumo" wake model
│     ├── hallo_lumo.tflite              int8 backbone (filename stem = model id)
│     └── head.h                         float MultiProto head (KWS_MEL_TIME 150, K=5)
├── licht_multi_v10.0_de_tflite.zip     "Licht rot/blau/grün/weiß" command model
│     ├── licht_multi.tflite             int8 backbone
│     ├── head.h                         multi head (KWS_HEAD_N 4, KWS_HEAD_K 3)
│     ├── model_info.json                mel config, words, per-word thresholds
│     └── _report.txt                    training/eval report
├── hallolumo_r1_1_x654af9.zip          ONNX variant (for Android/Web/Python; no head.h)
└── ...                                 older ONNX builds kept for reference
```

For ESP32 use the `*_tflite.zip` packages. The two `head.h` files cannot be
compiled together as-is (clashing symbol names) — the example's
`main/_merge_heads.py` merges them into one `main/head.h` with `KWS_WAKE_*` /
`KWS_CMD_*` prefixes. All models merged this way must share the same mel
configuration. Details in the
[example guide](examples/esp32s3_hmi_devkit/README.md#how-the-heads-are-merged-main_merge_headspy).

## Measured performance

With the bundled mel_time=150 models (1.532 s window) on the tested ESP32-S3 at 240 MHz (esp-tflite-micro 1.4.1 + ESP-NN 1.4.1), Mel extraction measures about 47 ms and TFLite `Invoke()` about 76 ms — 121 ms per frame against a 160 ms hop, i.e. real time with margin. This was measured in the live two-model demo (ESP-SR AFE running on the other core; a gated-off model skips Invoke and costs nothing). Results vary by model, clock, and target.

For reference, the older 98-frame models measured ≈ 29 ms Mel + ≈ 40 ms Invoke (≈ 51 ms wall-clock with the former MultiNet also running).

## Important notes

- **Always swap `head.h` together with the model**: the classifier weights and calibration constants (`KWS_ABS_TEMP` / `KWS_FC_B`) in `main/head.h` are matched to the `.tflite` in `spiffs_content/`; replacing only one side produces wrong scores.
- **Enlarge the tensor arena when raising the window length**: `TFLITE_ARENA_SIZE` in `sdk/voicute/model_loader.h` is fixed at 64 KB, while the bundled 98-frame model uses about 43 KB. `KWS_MEL_TIME` in `main/head.h` is configurable (defaults to 98 when undefined); if you raise it (e.g. 200 frames for long wake words), scale the arena accordingly, otherwise model loading fails for lack of memory.
- **A mel_time mismatch refuses to load**: if the model's input frame count differs from `KWS_MEL_TIME`, startup fails fast with an error — there is no silent fallback. This is intentional, so model/firmware mismatches surface immediately.
- **Component downloads need the network**: `idf.py set-target` re-solves component versions; `dependencies.lock` pins them for reproducibility. On machines that reach the internet through a system proxy, do not set `no_proxy=*` or clear proxy variables in build scripts — the registry lookup then fails silently ("Component not found" / "no versions match").
- **Run `patch_led_strip.py` after `set-target`**: set-target re-downloads `managed_components/` and wipes the patch. `build.bat` already does this in the right order; don't skip the step in manual builds.

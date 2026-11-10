# minimal_wakeword — 最小 voicute ESP-IDF 组件示例

English below | 中文

## 这个示例做什么

用 **组件依赖**（而不是复制源码）的方式引入 voicute 组件，编译出一个最小的
唤醒词工程：

- 不含模型、不含麦克风驱动、不含板级 BSP —— 组件本身也刻意不含这些；
- 编译即可通过；空跑时喂静音帧，串口会提示缺什么；
- 你需要自己提供的三样东西（见下），这正是组件的设计边界。

## 快速开始

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p PORT flash monitor
```

第一次构建时 Component Manager 会自动拉取依赖
（`voicute/voicute`、`espressif/esp-tflite-micro`、`espressif/esp-dsp`）。
在仓库树内直接构建时，`main/idf_component.yml` 里的相对路径会把组件解析
到本仓库；用 `idf.py create-project-from-example` 导出的工程则自动改用
Registry 版本。

## 让它真正唤醒（三件事，都由调用方提供）

1. **模型**：把训练好的 `.tflite` 烧到 `storage` 分区（文件名主干即唤醒词）。
   本示例 `partitions.csv` 已含 1MB storage 分区。
   模型从 voicute.com 导出，或用 `keywords/*/tflite/` 正式产物。
2. **模型头**：把 `main/head_stub.h` 换成随模型一起生成的真实头
   （`kws_postprocess.h` + `head.h` 模式，参考
   `esp32/idf/examples/esp32s3_hmi_devkit/main/main.cpp` 的 per-model 分发写法）。
3. **音频**：把 `main.cpp` 里的静音 buffer 换成你的麦克风 PCM
   （16 kHz / int16 / 单声道，每 160 ms 喂 `MEL_AUDIO_LEN` = 16080 采样，
   RMS 由调用方算好传入）。

> 注意：`voicute.h` 会传递包含 TFLite Micro 的 C++ 头文件，所以你的应用主文件必须是 C++（`.cpp`）。

## 串口输出预期

```text
I (xxx) minimal: SPIFFS mount ok / or WARNING: SPIFFS mount failed
W (xxx) minimal: No models on SPIFFS. Flash a trained .tflite ...
```

## 目标芯片

- **ESP32-S3**：完整实测（本仓库的板级验证都在 S3 上）。
- 其他目标（含 ESP32-P4）：代码路径可用，未经硬件实测，不做支持承诺。

# ESP32-S3-HMI-DevKit 唤醒 + 命令词演示（自有 KWS 模型，MultiNet 已移除）

[English](README.md) | 中文

这是使用 Voicute SDK 的完整板级应用。板子上同时跑 **两个我们自己的 KWS 模型**（Causal DS-TCN），完全替换了乐鑫 MultiNet 命令模型：

- **唤醒模型** — `hallo_lumo.tflite`（"Hallo Lumo"，单头 MultiProto，K=5）
- **命令模型** — `licht_multi.tflite`（"Licht rot/blau/grün/weiß"，多头，N=4 × K=3）

两个模型都是 mel_time=150 导出（1.532s 窗口 / 24512 采样 @ 16kHz，32 mel）。板上实测：噪声环境（rms 150–190）和安静房间（rms 24–36）共 20+ 次唤醒（prob 0.73–1.000）全命中，四色命令零混淆；mel ≈ 47ms + invoke ≈ 76ms ≈ 121ms/帧（160ms 一帧，实时）。

## 运行平台

当前实测平台为 **ESP32-S3-HMI-DevKit**：ESP32-S3 运行在 240 MHz，8 MB Octal PSRAM、16 MB Flash、4 颗经 ES7210 接入的 MEMS 麦克风、ES8311 音频输出、WS2812 灯链和 16 kHz 音频。验证框架为 **ESP-IDF 6.0.1**。

移植到其他开发板时，需要修改 `main/bsp_board.c`、GPIO、Codec 配置，并按硬件调整分区和 PSRAM 设置。

## 架构：一条推理链路，两个模型，分时复用

tflite 文件里**只有 backbone**；MultiProto 头是 float C 数组，编译进固件（`main/head.h`）。两个模型共用 音频 → mel → invoke 推理链路，由状态机门控当前活跃的模型：

```text
IDLE    → 只跑唤醒模型   (hallo_lumo, 滚动窗口)
COMMAND → 只跑命令模型   (licht_multi, 同一窗口), 6s 超时回 IDLE
```

门控是按模型的 `recognizer_set_active()` —— 被禁用的模型直接跳过 Invoke，所以命令模式没有任何额外开销。`main/main.cpp` 的关键部件：

| 部件 | 作用 |
|---|---|
| `resolve_model_indices()` | 按**文件名 stem**（`hallo_lumo` / `licht_multi`）把角色映射到注册表索引 —— SPIFFS readdir 顺序不保证，绝不要硬编码索引 |
| `dispatch_postprocess()` | 带 `model_idx` 的 postprocess 回调：cmd 模型 → `kws_postprocess_cmd()` + **调用方自己做 argmax**（它返回的是头*数量*，不是赢家）；wake 模型 → `kws_postprocess_wake()` |
| `on_wake` / `on_cmd` / `leave_cmd` | 状态切换。每次切换都要 `recognizer_set_active()` **和** `recognizer_reset_smooth()` —— 平滑缓冲里残留的概率峰会紧接在切换后把另一个模型重新触发 |
| `led_show()` | 本板 WS2812 灯链相对 `led_strip_set_pixel` 的参数顺序 R/G 是反的；在这里一次性换过来，调用方传逻辑 RGB |

检测配置：两模型统一 `threshold=0.70f`，L1–L5 门控全关（纯阈值判决，板上已验证）。导出包推荐每词 0.5 —— 0.70 对所有命令都够，但如果某个词漏识别可以调低。

## 模型目录：什么放哪里

| 文件 | 内容 | 替换为 |
|---|---|---|
| `spiffs_content/hallo_lumo.tflite` | 唤醒 backbone（int8, ~67KB） | 你的唤醒模型 `*.tflite` |
| `spiffs_content/licht_multi.tflite` | 命令 backbone（int8, ~67KB） | 你的命令模型 `*.tflite` |
| `main/head.h` | **合并后**的 MultiProto 头（生成文件 —— 不要手改） | 由 `_merge_heads.py` 重新生成 |

tflite 文件名 stem（去掉扩展名）就是固件解析角色用的模型标识。SPIFFS 分区 1MB，两个 tflite 在编译时自动打进镜像。

**`*.tflite` 和 `main/head.h` 是配套文件，必须一起换** —— 头权重必须和 backbone 出自同一次训练，否则概率无意义。

## head 怎么合并（`main/_merge_heads.py`）

两个模型不能各带各的 `head.h`：导出包里的 head.h 定义同名符号（`KWS_FC_W`、`KWS_PROTO_NORM`、`kws_postprocess_multi`……），链接时会冲突。生成脚本把两份 head.h 机械合并成一个头文件，符号加前缀：

- 唤醒模型 → `KWS_WAKE_*`（`KWS_WAKE_FC_B/W`、`KWS_WAKE_PROTO_NORM`、`KWS_WAKE_ABS_TEMP`）+ 手写的 `kws_postprocess_wake()`（单头：`sigmoid(b + Σ W[k]·cos/T)`）
- 命令模型 → `KWS_CMD_*`（`KWS_CMD_N`、`KWS_CMD_K`、权重数组）+ `kws_postprocess_multi` 改名 `kws_postprocess_cmd`
- 公共 define：`KWS_MEL_TIME 150`、`KWS_N_MELS 32`、`KWS_HEAD_D 256`、`KWS_WAKE_K`、`KWS_CMD_N`、`KWS_CMD_K`、`KWS_CMD_WORD_0..N`

换新模型时重新生成：

1. 解压两个模型导出包，找到各自的 `head.h`（tflite 在 tflite 包里）。
2. 改 `main/_merge_heads.py` 顶部的三个硬编码路径（`HALLO`、`LICHT` = 两份 head.h 输入，`OUT` = `main/head.h`）。
3. 运行（任意 Python 3，纯标准库）：`python main/_merge_heads.py`
4. 两个 tflite 放进 `spiffs_content/`，**stem 要和固件匹配的名字一致**（否则更新 `resolve_model_indices()` ）。
5. 按你的命令词更新 `g_cmd_names[]` / `on_cmd` 里的 LED 映射。

**约束：所有参与合并的模型 mel 配置必须一致** —— 合并头文件只有一套 mel define（`KWS_MEL_TIME`、`KWS_N_MELS`）。mel_time 不同的两个模型没法这样合并，得各自独立头文件和独立输入窗。

烧录前的自检：生成的头文件是纯 C —— 在主机上 `gcc -c`（加 `-lm`）编译，用一段垃圾 int8 缓冲调两个 postprocess 函数；预期唤醒 prob 接近 0、命令头输出 N 个概率。

## 编译与烧录

在本目录打开 ESP-IDF 6.0.x 环境后执行：

```bash
idf.py set-target esp32s3
python patch_led_strip.py
idf.py build
idf.py -p COM6 flash monitor
```

Windows 可以运行 `build.bat`；烧录前请修改 `flash.bat` 中的串口号。

## 串口注意事项（Windows）

- ESP_LOG 行里有 `→`（U+2192），cp1252 的 Python 抓取脚本 print 会崩。用 `PYTHONIOENCODING=utf-8` + `sys.stdout.reconfigure(errors="replace")`（见 `_serial_capture.py`）。
- 重开 USB-Serial-JTAG 口会拉 DTR **硬复位板子** —— 除非就是要复位，异常时不要立刻重连。

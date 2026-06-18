# 轻量离线关键词识别 / 唤醒词 / 语音唤醒


`唤醒词` · `关键词识别` · `语音唤醒` · `自定义唤醒词` · `离线语音识别` · `KWS` · `Keyword Spotting` · `Wake Word` · `ONNX` · `端侧推理` · `ESP32` · `Android` · `开源`

> **离线运行 · 模型 < 130KB · 不上传音频 · ESP32/Android/Linux/Web 全平台**

[:us: English](README_EN.md) | 🚧 English wake word support is under development — [learn more](README_EN.md#-english-wake-word-support--coming-soon)

本仓库提供各平台的开源唤醒词/关键词识别推理代码。支持自定义唤醒词，拿到 ONNX 模型就能在 Android、Web、Linux、ESP32 上跑离线语音识别和语音唤醒，不依赖云端。

## Web Demo

![网页截图](web/screenshot.png)

内置**防误唤醒设置面板**（5 层检测开关 + L5 倍率滑块 + 阈值调节 + 置信度进度条）。

## 获取 ONNX 模型

你可以自己训练，也可以使用在线服务生成。

### 自己训练

以下工具都能导出 ONNX 唤醒词模型：

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — 开源，家居场景，支持多唤醒词
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — 专为 ESP32 等 MCU 设计
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword) — 11 种架构可选，模型极小(40KB)
- 任何能导出 ONNX 的 KWS 训练框架均可

### 在线生成

[voicute.com](https://www.voicute.com) 输入中文唤醒词，自动生成 ONNX 模型。首次免费。

## 模型文件

无论用哪个工具训练，最终需要两个文件：

| 文件 | 说明 |
|------|------|
| `melspectrogram.onnx` | 音频 → 梅尔频谱，通用模块，**本仓库已提供** |
| `你的模型.onnx` | 唤醒词推理模型，训练或在线生成获取 |

外加一个 `model_info.json` 描述模型配置。

> 本仓库 `models/` 目录已包含演示模型，详见 [models/README.md](models/README.md) 版本说明。

## 模型配置

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "曼波", "model_file": "dscnn_multiscale_manbo.onnx", "cons_frames": 3}
  ]
}
```

多个唤醒词：

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "打开灯光", "model_file": "dakaidengguang.onnx", "cons_frames": 3},
    {"wake_word": "你好电脑", "model_file": "nihaodiannao.onnx", "cons_frames": 3}
  ]
}
```

## 文件结构

```
onnx-wakeword/
├── android/     # Android (Java, ONNX Runtime)
├── web/         # Web (JavaScript, ONNX Runtime Web)
├── linux/       # Linux / 树莓派 (Python)
├── esp32/       # ESP32-S3/P4
└── models/      # 测试用模型和配置（不提交 git）
```

## 各平台调用

| 平台 | 目录 | SDK 入口 |
|------|------|------|
| Android | `android/` | `WakeWordEngine.java` |
| Web | `web/` | `wakeword.js` → `VoicuteWakeWord.create()` |
| Linux | `linux/` | `wakeword_engine.py` → `WakeWordEngine()` |

### Web

```html
<script src="onnxruntime-web/ort.min.js"></script>
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();
  // 支持本地路径、网络 URL、ZIP 包
  await engine.load('model_info.json', 'melspectrogram.onnx');
  engine.set_L1(true);
  await engine.start((word, prob) => {
    console.log(`检测到: ${word} (${(prob*100).toFixed(0)}%)`);
  });
</script>
```

### Linux

```bash
pip install onnxruntime numpy pyaudio
```

```python
from wakeword_engine import WakeWordEngine
engine = WakeWordEngine()
engine.load('model_info.json', 'melspectrogram.onnx')
engine.set_L1(True)
engine.start(lambda word, prob, info: print(f'{word}'))
```

### Android

复制模型到 `assets/`，编译运行。

```java
WakeWordEngine engine = new WakeWordEngine(context);
DetectionResult result = engine.process(audioChunk);
```

## 防误触发检测层

5 层可独立开关，生产环境建议全部开启：

| 开关 | 默认 | 解决什么问题 |
|:---:|:---:|------|
| L1 连续帧 | 开 / 2~3帧 | 键盘敲击、椅子响等瞬态短噪声 |
| L2 峰值/背景 | 关 | 安静环境下模型对底噪的幻觉 |
| L3 冷却 | 关 | 同一句话被重复识别多次 |
| L4 爆发封锁 | 关 | 音频回路导致密集误触发 |
| L5 能量跳变 | 关 | 视频播放、背景音乐等持续噪声 |

L5 倍率可调：`engine.set_L5_ratio(3.0)`（2.0-8.0x，默认 3.0）

## 版本历史

**v9.0 (2026-06)**
- 端到端推理架构升级，模型 ~128KB
- Web/Linux/Android 三端统一 API
- L1-L5 五层防误触发检测，可独立开关
- L5 能量跳变（curRms/preMin），防视频/音乐误触发
- COS ZIP 直加载
- 安静守卫（preMin<50 && rms<80）
- cons_frames 自动计算（2字=2帧, 3+字=3帧）

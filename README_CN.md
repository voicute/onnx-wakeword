# 轻量离线关键词识别 / 唤醒词 / 语音唤醒


`唤醒词` · `关键词识别` · `语音唤醒` · `自定义唤醒词` · `离线语音识别` · `KWS` · `Keyword Spotting` · `Wake Word` · `ONNX` · `端侧推理` · `ESP32` · `Android` · `开源`

> **离线运行 · 模型 < 130KB · 不上传音频 · ESP32/Android/Python/Web 全平台**

[:us: English](README.md)

本仓库提供各平台的开源唤醒词/关键词识别推理代码。支持自定义唤醒词，拿到 ONNX 模型就能在 Android、Web、Python (Linux/Windows/macOS)、ESP32 上跑离线语音识别和语音唤醒，不依赖云端。

## 性能数据

测试基于 **v9.3 Causal DS-TCN 模型**（~25K 参数，~128KB ONNX，2000 正样本 + 15000 负样本训练）。
架构介绍见 [models/README.md](models/README.md)。

| 项目 | 数值 |
|------|------|
| 参数量 | ~25K |
| ONNX 大小 | ~128KB (FP32) / ~25KB (INT8) |
| 桌面推理 | <5ms / 帧 |
| ESP32-S3 推理 | <10ms / 帧 |

**关键词召回率**（Azure TTS 留出集，10 发音人 × 多种语速/音调/音量）：

| 关键词 | 语言 | 召回率 |
|---------|:---:|:------:|
| 小坦小坦 | 中文 | 96.7% |
| 打开灯光 | 中文 | 94.2% |
| Cyclops | 英文 | 92.3% |
| Hey Friday | 英文 | 91.5% |
| Hey Limi | 英文 | 89.9% |

> 加入 5–20 条用户录音做语音增强训练后，真人召回可达 90%+。

**误触发抵抗**（20,000 条负样本，覆盖语音/音乐/噪声/近似词；阈值 0.5，L1+L3 开启）：

| 指标 | 数值 |
|------|------|
| 负样本准确率（静态测试集） | 93–96% |
| 实际环境估算（安静室内） | 0.1–0.3 次/小时 |
| 开启 L5 后 | ~0.1 次/小时 |

**防误触发层效果**（安静环境底噪测试）：

| 配置 | 误触发/小时 | 降幅 |
|:---|:---:|:---:|
| 关闭 | ~2.5 | — |
| L1 | ~0.7 | -72% |
| L1 + L3 | ~0.2 | -91% |
| L1 + L3 + L5 | ~0.1 | -96% |

**vs OpenWakeWord**（相同关键词 Cyclops，相同测试集）：

| 对比项 | Voicute (v9.3) | OpenWakeWord |
|--------|:---:|:---:|
| Cyclops 召回率 | **92.3%** | 37–41% |
| 负样本准确率（2 万负样本） | **93–96%** | 2.1–3.9 次/时 |
| 模型大小 | **128KB** | ~800KB |
| 训练耗时 | **~30 分钟** | ~75–90 分 |
| 多语言 | ✅ 150+ 语言 | ❌ 仅英文 |
| 自定义词质量 | ✅ 稳定 90%+ | ⚠️ 50–80% 波动 |

## Web Demo

![网页截图](web/screenshot.png)

内置**防误唤醒设置面板**（5 层检测开关 + L5 增量滑块 + 阈值调节 + 置信度进度条）。

## 获取 ONNX 模型

你可以自己训练，也可以使用在线服务生成。

### 自己训练

以下工具都能导出 ONNX 唤醒词模型：

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — 开源，家居场景，支持多唤醒词
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — 专为 ESP32 等 MCU 设计
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword) — 11 种架构可选，模型极小(40KB)
- 任何能导出 ONNX 的 KWS 训练框架均可

### 在线生成

[voicute.com](https://www.voicute.com) 输入唤醒词或关键词，自动生成 ONNX 模型。

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
    {"wake_word": "曼波", "model_file": "dscnn_multiscale_manbo.onnx", "cons_frames": 2}
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
    {"wake_word": "打开灯光", "model_file": "dakaidengguang.onnx", "cons_frames": 2},
    {"wake_word": "你好电脑", "model_file": "nihaodiannao.onnx", "cons_frames": 2}
  ]
}
```

## 文件结构

```
onnx-wakeword/
├── android/     # Android (Java, ONNX Runtime)
├── web/         # Web (JavaScript, ONNX Runtime Web)
├── python/      # Linux / Windows / macOS (Python)
├── esp32/       # ESP32-S3/P4
└── models/      # 测试用模型和配置（不提交 git）
```

## 各平台调用

| 平台 | 目录 | SDK 入口 |
|------|------|------|
| Android | `android/` | `WakeWordEngine.java` |
| Web | `web/` | `wakeword.js` → `VoicuteWakeWord.create()` |
| Python | `python/` | `wakeword_engine.py` → `WakeWordEngine()` |

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

### Python (Linux / Windows / macOS)

```bash
pip install onnxruntime numpy sounddevice
```

```bash
# 快速麦克风测试
python mic_test.py
python mic_test.py --all     # L1-L5 全开

# 代码调用
python -c "
from wakeword_engine import WakeWordEngine
engine = WakeWordEngine()
engine.load('models/model_info.json', 'models/melspectrogram.onnx')
engine.set_L1(True)
engine.start(lambda word, prob, info: print(f'{word}'))
"
```

### Android

复制模型到 `assets/`，编译运行。

```java
WakeWordEngine engine = new WakeWordEngine(context);
DetectionResult result = engine.process(audioChunk);
```

## 防误触发检测层

5 层可独立开关：

| 层 | 推荐 | 说明 | 什么时候开 |
|:---:|:---:|------|------|
| L1 | **必开** | 连续帧过滤瞬态噪声（键盘、椅子响） | 始终开启 |
| L3 | **建议开** | 1.5s 冷却防重复触发 | L1 不够时开 |
| L5 | 按需 | 能量跳变防视频/音乐误触发 | 音箱/嘈杂环境 |
| L2 | 按需 | 峰值/背景比，防模型幻觉 | 极安静环境有误触时开 |
| L4 | 按需 | 爆发封锁防回声回路 | 喇叭播放回声导致误触时开 |

> **建议流程**：先只开 L1 测识别率 → 有连击开 L3 → 有噪音误触发开 L5 → 还不行再开 L2/L4。不要一次全开，层数越多识别越严格。
>
> L2/L4 随模型升级（v9.3+）噪声地板已大幅降低，多数场景不需要开启。按实际情况测试后决定。

## 版本历史

**v9.3 (2026-06)**
- Web: 修复 run() 双重调度导致的卡死, 支持 model_type='tcn'
- Web: 添加缓存禁止 meta, 微信扫码缓存刷新
- 模型升级到 v9.3，误触发率大幅降低

模型版本更新日志见 [models/README.md](models/README.md)。

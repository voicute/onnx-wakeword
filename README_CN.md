# 轻量离线关键词识别 / 唤醒词 / 语音唤醒


`KWS` · `关键词识别` · `唤醒词` · `语音唤醒` · `自定义唤醒词` · `离线语音识别` · `Keyword Spotting` · `Wake Word` · `ONNX` · `端侧推理` · `ESP32` · `Android` · `开源`

> **离线运行 · 模型 < 130KB · 不上传音频 · ESP32/Android/Python/Web 全平台**

[:us: English](README.md)

本仓库提供 Voicute 关键词识别（KWS）/唤醒词的各平台开源推理引擎。支持自定义关键词，模型统一在 Voicute 平台训练生成，任意关键词稳定召回 90%+，拿到 ONNX 模型即可在 Android、Web、Python (Linux/Windows/macOS)、ESP32 上离线运行，不依赖云端。目前支持中文、英文、日语、法语、德语。

## 性能数据

测试基于 **v9.3 模型**。

| 项目 | 数值 |
|------|------|
| ONNX 大小 | ~128KB (FP32) / ~74KB (INT8) |
| 桌面推理 | <5ms / 帧 |
| ESP32-S3 推理 | <10ms / 帧 |

**关键词召回率**（Azure TTS 留出集，10 发音人 × 多种语速/音调/音量；滑动窗口峰值检测）：

| 关键词 | 语言 | 召回率 |
|---------|:---:|:------:|
| 小坦小坦 | 中文 | 100% |
| 元宝元宝 | 中文 | 100% |
| 你好琥珀 | 中文 | 100% |
| 小黑 | 中文 | 97.4% |
| Hey Robot | 英文 | 100% |
| サクラ (Sakura) | 日语 | 100% |
| Apfelstrudel | 德语 | 99.4% |
| Monsieur Sadin | 法语 | 100% |
| Croissant | 法语 | 90.3% |

> 近 20 个已训练关键词（5 种语言）实测：召回率 90.3%–100%，均值 98.8%，20/20 ≥ 90%。

> 加入 5–20 条用户录音做语音增强训练后，真人召回可达 90%+。

**误触发抵抗**（验证集约 25,000 条负样本，覆盖语音/音乐/噪声/近似词；阈值 0.5）：

| 指标 | 数值 |
|------|------|
| 负样本准确率（验证集） | 96–98% |

> **负样本挖掘与增强训练**（开发中）：针对单个关键词的误触发问题，后续将支持用户收集误触发音频，作为增强负样本加入训练，持续提升关键词准确率。

> 单个关键词训练耗时约 30 分钟。目前支持中文、英文、日语、法语、德语 5 种语言。

## Web Demo

![网页截图](web/screenshot.png)

内置**防误唤醒设置面板**（5 层检测开关 + L5 增量滑块 + 阈值调节 + 置信度进度条）。

## 获取 ONNX 模型

模型统一在 [voicute.com](https://www.voicute.com) 平台训练生成：输入关键词或唤醒词，自动生成 ONNX 模型。

> 本推理引擎为 Voicute 专用（Causal DS-TCN + MultiProto 架构，v9.3），模型需在 Voicute 平台训练。OpenWakeWord、MicroWakeWord 等第三方框架导出的 ONNX 模型架构不同，暂不兼容。

## 模型文件

每个模型需要两个文件：

| 文件 | 说明 |
|------|------|
| `melspectrogram.onnx` | 音频 → 梅尔频谱，通用模块，**本仓库已提供** |
| `你的模型.onnx` | 关键词推理模型，由 Voicute 平台训练生成 |

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

多个关键词：

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

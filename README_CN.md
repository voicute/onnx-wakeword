# onnx-wakeword — 离线唤醒词与关键词推理引擎


`KWS` · `关键词识别` · `唤醒词` · `语音唤醒` · `自定义唤醒词` · `离线语音识别` · `隐私优先` · `不上传音频` · `Keyword Spotting` · `Wake Word` · `ONNX` · `端侧推理` · `ESP32` · `Android` · `开源`

> **离线运行 · 不上传音频 · 模型 < 130KB · ESP32/Android/Python/Web 全平台**

[:us: English](README.md)

onnx-wakeword 是一个开源、完全离线的**唤醒词与关键词检测（KWS）推理引擎**。

项目提供从音频预处理、Mel 特征提取、模型执行、配套分类头计算到唤醒判定的完整运行时，主要面向采用**因果时序卷积网络（Causal TCN）**及配套分类头或原型头的关键词模型。训练管线完全自研，不依赖 Porcupine、OpenWakeWord 或其他第三方项目。

仓库提供 Python、Web、Android 的 ONNX 推理实现，以及针对 ESP32-S3 优化的 INT8 TFLite 推理实现。**所有推理均在本地完成，不需要上传音频**。完整的流程是：在 [voicute.com](https://www.voicute.com) 在线训练你的自定义关键词模型，下载后在任何支持的平台上离线加载运行。训练一次，永久免费使用，无 API 调用、无月费。

### 推理流程

```text
音频 → Mel 特征 → 关键词 TCN 模型 → 配套分类头或原型头 → 检测逻辑 → 识别结果
```

## 性能数据

测试基于 **v9.3 模型**。

| 项目 | 数值 |
|------|------|
| ONNX 大小 | ~128KB (FP32) / ~74KB (INT8) |
| 桌面推理 | <5ms / 帧 |
| ESP32-S3 TFLite Invoke（内置 Demo） | 约 155ms / 帧 |

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

> 加入 5 条用户录音做语音增强训练后，真人召回可达 90%+。

**误触发抵抗**（验证集约 25,000 条负样本，覆盖语音/音乐/噪声/近似词；阈值 0.5）：

| 指标 | 数值 |
|------|------|
| 负样本准确率（验证集） | 96–98% |

> **负样本挖掘与增强训练**（开发中）：针对单个关键词的误触发问题，后续将支持用户收集误触发音频，作为增强负样本加入训练，持续提升关键词准确率。

> 单个关键词训练耗时约 30 分钟。目前支持中文、英文、日语、法语、德语 5 种语言。

## 训练与部署

onnx-wakeword 采用「在线训练 + 离线运行」架构：

1. [输入你的关键词](https://www.voicute.com)（中/英/日/法/德），平台自动生成 TTS 训练数据并训练 Causal TCN 模型（~30 分钟）
2. 下载 `model.zip`，在任何支持的平台上用本仓库推理引擎加载运行

**模型在平台上训练**。训练完成后下载的模型完全离线运行——**推理时你的音频永远不会离开你的设备**。想先免费试用？`models/` 目录内置多语言演示模型，可以用相同流程零成本验证效果。

## Web Demo

![网页截图](web/screenshot.png)

内置**防误唤醒设置面板**（5 层检测开关 + L5 增量滑块 + 阈值调节 + 置信度进度条）。

## 模型

onnx-wakeword 是一个只提供推理代码的开源项目，运行时需要兼容的关键词模型及其配套分类头。自定义模型在 [voicute.com](https://www.voicute.com) 在线生成，详见上方「训练与部署」章节。

## 模型文件

每个模型需要两个文件：

| 文件 | 说明 |
|------|------|
| `melspectrogram.onnx` | 音频 → 梅尔频谱，通用模块，**本仓库已提供** |
| `你的模型.onnx` | 兼容的关键词推理模型 |

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
├── wyoming/     # Home Assistant（Wyoming 协议服务）
├── ha-addon/    # Home Assistant 加载项
├── Dockerfile   # Docker 镜像（voicute/voicute-wyoming）
└── models/      # 测试用模型和配置（不提交 git）
```

## 各平台调用

| 平台 | 目录 | SDK 入口 |
|------|------|------|
| Android | `android/` | `WakeWordEngine.java` |
| Web | `web/` | `wakeword.js` → `VoicuteWakeWord.create()` |
| Python | `python/` | `wakeword_engine.py` → `WakeWordEngine()` |
| Home Assistant | `wyoming/` | `wyoming_voicute.py` → Wyoming 协议 |

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
# 快速麦克风测试（默认 xiaona/小娜，自动搜索 zh/en/de/fr/ja）
python mic_test.py
python mic_test.py --all     # L1-L5 全开
python mic_test.py --model manbo    # 指定关键词名
python mic_test.py --path D:\models\custom.onnx   # 指定完整模型路径

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

### Home Assistant（Wyoming 协议）

内置 [Wyoming 协议](https://github.com/rhasspy/wyoming) 服务，可将 onnx-wakeword 直接接入 Home Assistant 作为唤醒词引擎——不需要 add-on，所有 HA 安装方式（HAOS / Supervised / Container / Core）都支持。

**1. 启动服务（Docker）：**

```bash
docker run -d --name voicute-wakeword --restart unless-stopped --network host \
  -v /你的模型目录:/models \
  voicute/voicute-wyoming:latest \
  --model-info /models/model_info.json --mel /app/models/melspectrogram.onnx
```

`melspectrogram.onnx` 已内置在镜像里，只需挂载你自己的 `model_info.json` + 关键词 `.onnx`。

**2. 在 Home Assistant 里添加：**

设置 → 设备与服务 → 添加集成 → **Wyoming Protocol** → 填主机 `IP` + 端口 `10400`。

**3. 语音助手选唤醒词：** 设置 → 语音助手 → 你的助手 → Wake word。

完整指南（麦克风实时测试、docker-compose、Home Assistant 加载项）：[`wyoming/README.md`](wyoming/README.md)。

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

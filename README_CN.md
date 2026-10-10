# onnx-wakeword — 离线唤醒词与关键词推理引擎


`KWS` · `关键词识别` · `唤醒词` · `语音唤醒` · `自定义唤醒词` · `离线语音识别` · `隐私优先` · `不上传音频` · `Keyword Spotting` · `Wake Word` · `ONNX` · `端侧推理` · `ESP32` · `Android` · `Home Assistant` · `Wyoming` · `开源`

> **离线运行 · 不上传音频 · 模型 < 130KB · ESP32/Android/Python/Web 全平台**

[:us: English](README.md)

onnx-wakeword 是一个开源、完全离线的**唤醒词与关键词检测（KWS）推理引擎**。

项目提供从音频预处理、Mel 特征提取、模型执行、配套分类头计算到唤醒判定的完整运行时，主要面向采用**因果时序卷积网络（Causal TCN）**及配套分类头或原型头的关键词模型。训练管线完全自研，不依赖 Porcupine、OpenWakeWord 或其他第三方项目。

仓库提供 Python、Web、Android 的 ONNX 推理实现，以及针对 ESP32-S3 优化的 INT8 TFLite 推理实现。**所有推理均在本地完成，不需要上传音频**。完整的流程是：在 [voicute.com](https://www.voicute.com/?utm_source=github&utm_medium=readme&utm_content=readme_cn) 在线训练你的自定义关键词模型，下载后在任何支持的平台上离线加载运行。训练一次，永久免费使用，无 API 调用、无月费。

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
| ESP32-S3 TFLite Invoke（内置 Demo） | 约 51ms / 帧（完整 AFE Demo） |

单独测量 `Invoke()` 的基线约为 39.5–40.2 ms；对外公布的约 51 ms 包含实时 ESP-SR AFE 流程，以及并行运行带来的调度和缓存开销。

### 召回率

（Azure TTS 留出集，400 条 × 多种语速/音调/音量；滑动窗口峰值检测）

| 关键词 | 语言 | 召回率 |
|---------|:---:|:------:|
| 你好小娜（优化版） | 中文 | 100% |
| Hey Jarvis（优化版） | 英文 | 99.0% |
| Salut Nova（优化版） | 法语 | 100% |
| Apfelstrudel（优化版） | 德语 | 98.5% |
| みらい（优化版） | 日语 | 100% |
| 小娜（优化版） | 中文 | 98.3% |
| 豆包豆包（优化版） | 中文 | 100% |
| Hola Orion（优化版） | 西班牙语 | 99.3% |
| हैलो तारक（优化版） | 印地语 | 99.3% |
| Hey Robot | 英文 | 100% |
| サクラ (Sakura) | 日语 | 100% |
| Apfelstrudel（基础版） | 德语 | 99.4% |
| Monsieur Sadin | 法语 | 100% |
| Croissant | 法语 | 90.3% |

> 标注「（优化版）」的行为误触发优化版模型（难负样本挖掘重训，见下）；未标注的行为基础版演示模型。

> 近 20 个已训练关键词（5 种语言）实测：召回率 90.3%–100%，均值 98.8%，20/20 ≥ 90%。

> 加入 5 条用户录音做语音增强训练后，真人召回可达 90%+。

### 误触发控制

**误触发优化：优化前后对比**（每个关键词用**自己的词标语料**实测——同语言 Common Voice 朗读语音 + 音乐 + 家用噪声/静音，训练留出；裸模型、阈值 0.5、40ms 滑窗、按触发文件计）：

| 关键词 | 语料 | 优化前 (次/小时) | 优化后 (次/小时) | 降幅 | 召回变化 |
|---------|---:|---:|---:|---:|---|
| 你好小娜 | 16.0h | 74.5 | 3.5 | −95.3% | 97.8% → 100% |
| Hey Jarvis | 26.9h | 44.6 | 4.8 | −89.2% | 100% → 99.0% |
| Salut Nova | 22.6h | 222.7 | 4.5 | −98.0% | 100% → 100% |
| Apfelstrudel | 24.4h | 265.3 | 2.7 | −99.0% | 97.0% → 98.5% |
| みらい | 12.1h | 330.9 | 5.1 | −98.5% | 97.8% → 100% |
| Hola Orion | 20.8h | 456.7 | 2.9 | −99.4% | 99.8% → 99.3% |
| हैलो तारक | 18.5h | 212.2 | 3.3 | −98.5% | 99.5% → 99.3% |

> 表中数字测于**词标留出语料**（同语言 Common Voice 朗读语音 + 音乐 + 家用噪声/静音，训练留出；每行用该词自己的语料，时长见行内）——刻意贴近真实噪声的**最坏情况**口径。在干净的公开朗读语料 AISHELL-1（5000 条抽样）上更低：优化版你好小娜 **0.8 次/小时**（基线 35.5）。

> 以上均为**裸模型**口径（检测层全关）。运行时的防误触发检测层可以继续压低——仅开启 L1 连续帧（默认即开）后，优化版你好小娜在其 16 小时留出语料上从 3.5 降到 **2.3 次/小时**（cons=2），cons=3 进一步降到 **1.0 次/小时**，这还没叠加其他层（L3/L5 等，按需再开）。

> **误触发优化版**：关键词模型训练完成后，一键启动优化——训练好的模型自动扫描负样本语料，
> 找出自己会误判为唤醒词的片段（难负样本）回流重训。模型从自己的错误中学习，
> **无需用户上报任何误触发录音**，召回不受影响（上表：单轮降幅 91%–98%）。

**如何使用误触发优化**：关键词训练完成后，在听词(Voicute)控制台里操作：

```text
我的模型 → 负样本扫描 → 全量重训 → 完成后下载 R1 模型
```

下载到的 R1 模型就是误触发优化版，加载方式与基础版完全一致（同样的 `model_info.json` 和推理代码，无需改任何代码），下载后照常离线使用。

本仓库内置的对比模型（`models/<语言>/`）：

| 关键词 | 语言 | 优化前基线 (`*_r0.onnx`) | 误触发优化版 (`*_r1.onnx`) |
|---------|:---:|---------|---------|
| 你好小娜 | 中文 | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` |
| Hey Jarvis | 英文 | `heyjarvis_r0.onnx` | `heyjarvis_r1.onnx` |
| Salut Nova | 法语 | `salutnova_r0.onnx` | `salutnova_r1.onnx` |
| Apfelstrudel | 德语 | `apfelstrudel_r0.onnx` | `apfelstrudel_r1.onnx` |
| みらい | 日语 | `mirai_r0.onnx` | `mirai_r1.onnx` |
| Hola Orion | 西班牙语 | `Hola Orion_v10.0_basic.zip` | `Hola Orion_v10.0_basic_r1.zip` |
| हैलो तारक | 印地语 | `हैलो तारक_v10.0_basic.zip` | `हैलो तारक_v10.0_basic_r1.zip` |

> 误触发优化已在**全部 7 种支持语言**（中/英/日/法/德/西班牙语/印地语）上完成生产验证；优化流程（负样本扫描 + 全量重训）实测约 40–80 分钟。
> 单个关键词训练耗时约 30 分钟，目前支持中文、英文、日语、法语、德语、西班牙语、印地语（7 种语言）。

## 多关键词模型

**一个模型识别多个关键词**：单次推理同时输出所有关键词的概率，不需要为每个词单独跑一遍模型。所有关键词共享同一个骨干网络，每新增一个词只增加约 0.8KB 的分类头——3 个词的模型约 135KB，10 个词也只有 167KB。

### 多唤醒词

把一个唤醒词的多种叫法收进同一个模型，用户怎么说都能唤醒：

| 压缩包 | 关键词 |
|---------|---------|
| `models/zh/multi_xiaona_v9.3.zip` | 小娜 · 你好小娜 · 小娜小娜 |
| `models/en/multi_jarvis_v9.3.zip` | Hey Jarvis · Jarvis · Hi Jarvis |
| `models/de/multi_martina_v9.3.zip` | Martina · Tina · Hey Tina |

### 语音控制

十条播放器指令装进一个模型：

| 压缩包 | 关键词 |
|---------|---------|
| `models/zh/multi_commands_v9.3.zip` | 播放 · 暂停 · 下一首 · 上一首 · 开始播放 · 停止播放 · 声音大一点 · 声音小一点 · 静音 · 继续播放 |

四条灯光控制指令，每种语言一个模型：

| 压缩包 | 关键词 |
|---------|---------|
| `models/es/multi_N4_990ee7_v10.0.zip` | Luz roja · Luz azul · Luz verde · Luz blanca |
| `models/fr/multi_N4_f97a7e_v10.0.zip` | Lumière rouge · Lumière bleue · Lumière verte · Lumière blanche |
| `models/hi/multi_N4_8a1d43_v10.0.zip` | लाल बत्ती · नीली बत्ती · हरी बत्ती · सफ़ेद बत्ती |

> 所有**多关键词演示包均为基础版（R0）模型**，尚未做误触发优化；误触发优化流程（负样本扫描 → 全量重训）**同样支持多关键词模型**，需要时按相同流程即可生成优化版；单关键词的优化版见 `models/<语言>/`（对比模型表）。

ZIP 压缩包**直接加载即可，无需解压**（Python / Web 引擎自动识别 ZIP 格式）：

```python
engine.load('models/zh/multi_commands_v9.3.zip', 'models/melspectrogram.onnx')
```

回调返回命中的关键词和置信度（见各平台调用示例）。自定义多关键词模型在 [voicute.com](https://www.voicute.com/?utm_source=github&utm_medium=readme&utm_content=readme_cn) 训练生成。

## 训练与部署

onnx-wakeword 采用「在线训练 + 离线运行」架构：

1. [输入你的关键词](https://www.voicute.com/?utm_source=github&utm_medium=readme&utm_content=train_cn)（中/英/日/法/德/西班牙语/印地语），平台自动生成 TTS 训练数据并训练 Causal TCN 模型（~30 分钟）
2. 下载 `model.zip`，在任何支持的平台上用本仓库推理引擎加载运行

**模型在平台上训练**。训练完成后下载的模型完全离线运行——**推理时你的音频永远不会离开你的设备**。想先免费试用？`models/` 目录内置多语言演示模型，可以用相同流程零成本验证效果。

## Web Demo

![网页截图](web/screenshot.png)

内置**防误唤醒设置面板**（5 层检测开关 + L5 增量滑块 + 阈值调节 + 置信度进度条）。

## 模型

onnx-wakeword 是一个只提供推理代码的开源项目，运行时需要兼容的关键词模型及其配套分类头。自定义模型在 [voicute.com](https://www.voicute.com/?utm_source=github&utm_medium=readme&utm_content=readme_cn) 在线生成，详见上方「训练与部署」章节。

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
  "model_type": "multi_keyword",
  "keywords": ["你好小娜"],
  "model_file": "nihaoxiaona_r1.onnx",
  "mel_time": 98,
  "cons_frames": 2,
  "n_mels": 32
}
```

多个关键词（单模型多输出）：

```json
{
  "model_type": "multi_keyword",
  "keywords": ["hey friday", "turn on light"],
  "model_file": "model.onnx",
  "mel_time": 98,
  "cons_frames": 2,
  "n_mels": 32
}
```

仓库内置 4 个多关键词演示模型，见上方「多关键词模型」章节。

> **`mel_time`** 是模型输入窗长（mel 帧数）：98 ≈ 1.0 秒，150 ≈ 1.5 秒，200 ≈ 2.0 秒。v10.0 起训练时按唤醒词语音尾部自动选窗（长词自动加大），引擎从 `model_info.json` 读取，无需手动配置。内置演示模型均为 98。

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
└── models/      # 演示模型、多关键词演示包、误触发优化对比模型
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

完整指南（麦克风实时测试、docker-compose、Home Assistant 加载项）：[`wyoming/README.md`](wyoming/README.md)。想要带截图的分步教程和浏览器在线试词，看[中文版 Home Assistant 自定义唤醒词指南](https://www.voicute.com/articles/home-assistant-custom-wake-word-zh.html?utm_source=github&utm_medium=readme&utm_content=ha_article_zh)。

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

**v10.0 (2026-09)** — 长词自适应识别窗，+35% 语速覆盖

**v9.3 (2026-06)**
- Web: 修复 run() 双重调度导致的卡死, 支持 model_type='tcn'
- Web: 添加缓存禁止 meta, 微信扫码缓存刷新
- 模型升级到 v9.3，误触发率大幅降低

模型版本更新日志见 [models/README.md](models/README.md)。

# Web SDK — Keyword Spotting & Wake Word Engine

Browser-based keyword detection. Fully offline — no audio leaves the device.

## Quick Start

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();

  // Load model (local path, URL, or ZIP package)
  await engine.load('models/model_info.json', 'models/melspectrogram.onnx');

  // Optional: enable debug logging
  engine.setDebug(true);

  // Start listening
  await engine.start((word, prob, info) => {
    console.log(`Detected: ${word} (${(prob*100).toFixed(0)}%)`);
  });

  // Stop
  engine.stop();
</script>
```

## API

### engine.load(modelInfoUrl, melUrl)

Load model config and mel feature extractor. Auto-detects ZIP packages (PK header).

```js
// Local path
await engine.load('../models/model_info.json', '../models/melspectrogram.onnx');

// Remote URL
await engine.load('https://cdn.example.com/model_info.json', 'https://cdn.example.com/mel.onnx');

// ZIP package (requires jszip.min.js)
await engine.load('https://cdn.example.com/model.zip', '../models/melspectrogram.onnx');
```

### engine.start(onDetect)

Start microphone listening. `onDetect(word, prob, info)` — `info` contains `{ bg, all, rms }`.

### engine.stop()

Stop listening and release microphone.

### engine.predict(audioData)

Single inference without microphone. `audioData` = Float32Array, 16kHz int16 range.

```js
const result = await engine.predict(chunk);
// { word: 'hey friday', prob: 0.92, bg: 0.05, all: {...}, consFrames: 3 }
```

### Configuration

```js
engine.setThreshold(0.4);    // threshold 0.3–0.95, default 0.4
engine.setCooldown(1500);    // cooldown ms, default 1500
engine.setDebug(true);       // toggle debug logging
engine.setL1(true);          // L1 consecutive frames
engine.setL2(false);         // L2 peak/background ratio
engine.setL3(false);         // L3 cooldown
engine.setL4(false);         // L4 burst detection
engine.setL5(false);         // L5 energy jump
```

## Anti-False-Trigger Layers

| Layer | Default | Purpose |
|:-----:|:-------:|---------|
| L1 | **ON** | Consecutive frames — filters transient clicks/noise |
| L3 | OFF | 1.5s cooldown — prevents duplicate triggers |
| L5 | OFF | Energy jump — blocks video/music playback |
| L2 | OFF | Peak/background ratio — prevents silence hallucination |
| L4 | OFF | Burst detection — blocks audio feedback loops |

## model_info.json Format

### Multi-keyword (single model, N outputs) — Recommended

```json
{
  "model_type": "multi_keyword",
  "keywords": ["xiaona", "hello xiaona"],
  "model_file": "model.onnx",
  "mel_time": 98,
  "n_mels": 32,
  "cons_frames": 2
}
```

### Legacy: multi-model (one ONNX per keyword)

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "hey friday", "model_file": "model.onnx", "cons_frames": 3}
  ]
}
```

---

## 中文说明

浏览器本地关键词检测，不上传音频。API 与 Python/Android 统一。

### 快速开始

```html
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();
  await engine.load('models/model_info.json', 'models/melspectrogram.onnx');
  await engine.start((word, prob) => {
    console.log(`检测到: ${word}`);
  });
</script>
```

### 防误触发检测层

| 层 | 默认 | 说明 |
|:---:|:---:|------|
| L1 | 开 | 连续帧过滤瞬态噪声 |
| L3 | 关 | 1.5s 冷却防重复触发 |
| L5 | 关 | 能量跳变防视频/音乐误触发 |
| L2 | 关 | 峰值/背景比，防模型幻觉 |
| L4 | 关 | 爆发封锁防回声回路 |

### model_info.json 格式

多关键词（推荐）：

```json
{
  "model_type": "multi_keyword",
  "keywords": ["小娜", "你好小娜"],
  "model_file": "model.onnx",
  "mel_time": 98,
  "n_mels": 32,
  "cons_frames": 2
}
```

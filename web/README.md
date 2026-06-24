# Voicute Wake Word — Web SDK

浏览器本地运行唤醒词模型，不上传音频。

## 快速开始

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();

  // 加载模型（支持本地路径、网络URL、或COS ZIP包）
  await engine.load('models/model_info.json', 'models/melspectrogram.onnx');

  // 可选：调试日志
  engine.setDebug(true);

  // 开始监听
  await engine.start((word, prob, info) => {
    console.log(`检测到: ${word} (${(prob*100).toFixed(0)}%)`);
  });

  // 停止
  engine.stop();
</script>
```

## API

### engine.load(modelInfoUrl, melUrl)

加载模型配置和 mel 特征提取器。

- `modelInfoUrl` — `model_info.json` 路径，也支持 COS ZIP 包
- `melUrl` — `melspectrogram.onnx` 路径
- 自动检测 ZIP：如果响应头是 PK 开头，自动解压提取

```js
// 本地相对路径
await engine.load('../models/model_info.json', '../models/melspectrogram.onnx');

// 网络 URL
await engine.load('https://cdn.example.com/model_info.json', 'https://cdn.example.com/mel.onnx');

// COS ZIP 包（需要 jszip.min.js）
await engine.load('https://cos.example.com/manbo_basic_dcnn_v9.0.zip', '../models/melspectrogram.onnx');
```

### engine.start(onDetect)

开始麦克风监听。

- `onDetect(word, prob, info)` — 检测回调。`info` 含 `{ bg, all, rms }`

### engine.stop()

停止监听，释放麦克风。

### engine.predict(audioData)

单次推理，不经过麦克风。`audioData` 为 Float32Array，16kHz int16 范围。

```js
const result = await engine.predict(chunk);
// { word: '曼波', prob: 0.92, bg: 0.05, all: {曼波:0.92}, consFrames: 3 }
```

### 配置方法

```js
engine.setThreshold(0.4);    // 阈值 0.3~0.95，默认 0.4
engine.setCooldown(1500);    // 冷却时间 ms，默认 1500
engine.setDebug(true);       // 开启/关闭调试日志
engine.setL1(true);          // L1 连续帧过滤（默认开）
engine.setL2(false);         // L2 峰值比过滤
engine.setL3(false);         // L3 冷却
engine.setL4(false);         // L4 爆发封锁
engine.setL5(false);         // L5 能量跳变
```

### 状态查询

```js
engine.isLoaded();           // 模型是否加载完成
engine.getModels();          // [{ name: '曼波', consFrames: 3 }]
engine.debug;                // { sampleRate, inferCount, ... }
```

## 防误唤醒设置

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

## model_info.json 格式

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

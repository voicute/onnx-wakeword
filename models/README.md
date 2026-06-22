# 模型目录

放置训练好的 ONNX 唤醒词模型。

## 共享模型

`melspectrogram.onnx` 是通用的音频预处理模块，**本项目已提供**，无需额外下载。

## 如何使用

### 单个唤醒词

1. 把训练好的模型文件放入 `models/`
2. 编辑 `model_info.json`：

```json
{
  "wake_word": "你的唤醒词",
  "model_file": "your_model.onnx",
  "emb_frames": 1,
  "cons_frames": 3
}
```

### 多个唤醒词

```json
{
  "multi_model": true,
  "models": [
    { "wake_word": "打开灯光", "model_file": "dakaidengguang.onnx", "emb_frames": 1, "cons_frames": 3 },
    { "wake_word": "你好电脑", "model_file": "nihaodiannao.onnx",   "emb_frames": 1, "cons_frames": 3 }
  ]
}
```

## 演示模型

当前 `models/` 目录包含以下演示模型（来自 [voicute.com](https://www.voicute.com)）：

| 唤醒词 | 模型文件 | 版本 |
|--------|----------|:---:|
| 曼波 | `manbo.onnx` | v9.3 |
| 曼波 (语音定制) | `manbo_voice_model.onnx` | v9.3-voice |
| 你好电脑 | `nihaodiannao.onnx` | v9.3 |
| 开始播放 | `kaishibofang.onnx` | v9.3 |
| 来福 | `laifu.onnx` | v9.3 |
| 咕咕嘎嘎 | `gugugaga.onnx` | v9.3 |

> **语音定制版 (voice)**: 在标准 TTS 训练基础上加入真人录音 x50 权重 + 80 epoch 训练，误触发率比标准版低约 17%，对特定用户的发音习惯识别更稳定。

## 版本说明

| 版本 | 主要更新 |
|------|------|
| v9.3 | 优化误唤醒率，提升多音色覆盖 |
| v9.2 | 扩展训练数据多样性 |
| v9.1 | 增强远场识别能力 |
| v9.0 | 新一代唤醒架构，替代旧版 |

## `emb_frames` 说明

训练时自动确定的帧数参数，记录在训练输出的 `config.json` 中。**必须和模型训练时一致**，填写错误会导致无法识别。

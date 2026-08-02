# Android — Keyword Spotting & Wake Word Demo

Full Android project. Open with Android Studio, build, and run.

## Quick Start

1. Copy ONNX model + `model_info.json` to `app/src/main/assets/`
2. Open `android/` in Android Studio, build & run

## Model Config

Multi-keyword (recommended):

```json
{
  "model_type": "multi_keyword",
  "keywords": ["hey friday", "turn on light"],
  "model_file": "commands.onnx",
  "mel_time": 98,
  "n_mels": 32,
  "cons_frames": 2
}
```

## Core Classes

| File | Purpose |
|------|---------|
| `WakeWordEngine.java` | Model loading, mel extraction, inference |
| `DetectionLogic.java` | L1-L5 anti-false-trigger pipeline |
| `AudioCapture.java` | Mic capture, lock-free ring buffer |
| `MainActivity.java` | Demo UI |

## L1-L5 Detection Layers

| Layer | Default | Purpose |
|:-----:|:-------:|---------|
| L1 | ON | Consecutive frames — filters transient clicks/noise |
| L3 | OFF | 1.5s cooldown — prevents duplicate triggers |
| L5 | OFF | Energy jump — blocks video/music playback |
| L2 | OFF | Peak/background ratio — prevents silence hallucination |
| L4 | OFF | Burst detection — blocks audio feedback loops |

## Dependencies

- `onnxruntime-android` (1.20+)
- Android 8.0+ (API 26+)

---

## 中文说明

完整 Android 项目，Android Studio 打开即可运行。

### 使用

1. 复制 ONNX 模型和 `model_info.json` 到 `app/src/main/assets/`
2. Android Studio 打开 `android/` 目录，编译运行

### 核心类

| 文件 | 说明 |
|------|------|
| `WakeWordEngine.java` | 模型加载、mel 特征提取、推理 |
| `DetectionLogic.java` | L1-L5 五层防误触发检测 |
| `AudioCapture.java` | 麦克风采集、无锁环形缓冲区 |
| `MainActivity.java` | Demo UI |

### L1-L5 检测层

| 层 | 推荐 | 说明 |
|:---:|:---:|------|
| L1 | 必开 | 连续帧过滤瞬态噪声 |
| L3 | 建议 | 1.5s 冷却防重复触发 |
| L5 | 按需 | 能量跳变防视频/音乐误触发 |
| L2 | 按需 | 峰值/背景比，防模型幻觉 |
| L4 | 按需 | 爆发封锁防回声回路 |

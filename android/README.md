# Android 唤醒词 Demo

完整 Android 项目，Android Studio 打开即可运行。

## 使用

1. 复制 ONNX 模型和 `model_info.json` 到 `app/src/main/assets/`
2. Android Studio 打开 `android/` 目录，编译运行

## 模型配置

DS-CNN 架构（推荐）：

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "曼波", "model_file": "dscnn_multiscale_manbo.onnx", "cons_frames": 3},
    {"wake_word": "你好电脑", "model_file": "dscnn_multiscale_nihaodiannao.onnx", "cons_frames": 3}
  ]
}
```

## 核心类

| 文件 | 说明 |
|------|------|
| `WakeWordEngine.java` | 模型加载、mel 特征提取、推理（支持 DS-CNN 和 NWW） |
| `DetectionLogic.java` | L1-L5 五层防误触发检测 |
| `AudioCapture.java` | 麦克风采集、锁-free 环形缓冲区 |
| `MainActivity.java` | Demo UI |

## L1-L5 检测层

| 层 | 推荐 | 说明 | 什么时候开 |
|:---:|:---:|------|------|
| L1 | **必开** | 连续帧过滤瞬态噪声（键盘、椅子响） | 始终开启 |
| L3 | **建议开** | 1.5s 冷却防重复触发 | L1 不够时开 |
| L5 | 按需 | 能量跳变防视频/音乐误触发 | 音箱/嘈杂环境 |
| L2 | 按需 | 峰值/背景比，防模型幻觉 | 极安静环境有误触时开 |
| L4 | 按需 | 爆发封锁（3次/3s→5s）防回声回路 | 喇叭播放回声导致误触时开 |

> **建议流程**：先只开 L1 测识别率 → 有连击开 L3 → 有噪音误触发开 L5 → 还不行再开 L2/L4。
>
> L2/L4 随模型升级（v9.3+）噪声地板已大幅降低，多数场景不需要开启。

## 依赖

- `onnxruntime-android` (1.20+)
- Android 8.0+ (API 26+)

## 注意

`assets/` 下的 `.onnx` 和 `model_info.json` 仅本地测试用，**不要提交到 git**（已配 `.gitignore`）。

# ESP32-S3-HMI-DevKit Hey Robot 演示

[English](README.md) | 中文

这是使用 Voicute SDK 的完整板级应用。

## 运行平台

当前实测平台为 **ESP32-S3-HMI-DevKit**：ESP32-S3 运行在 240 MHz，8 MB Octal PSRAM、16 MB Flash、4 颗经 ES7210 接入的 MEMS 麦克风、ES8311 音频输出、WS2812 状态灯和 16 kHz 音频。验证框架为 **ESP-IDF 6.0.1**。

移植到其他开发板时，需要修改 `main/bsp_board.c`、GPIO、Codec 配置，并按硬件调整分区和 PSRAM 设置。

## 演示流程

1. 说 **Hey Robot** 唤醒设备。
2. 说 Turn the light red、blue、green 或 white。
3. LED 显示对应颜色，程序返回唤醒词检测状态。

`spiffs_content/hey_robot.tflite` 和 `main/head.h` 是配套文件，更换模型时必须同时替换。

## 编译与烧录

在本目录打开 ESP-IDF 6.0.x 环境后执行：

```bash
idf.py set-target esp32s3
python patch_led_strip.py
idf.py build
idf.py -p COM6 flash monitor
```

Windows 可以运行 `build.bat`；烧录前请修改 `flash.bat` 中的串口号。

# 模型存放目录

把 `.tflite` 唤醒词模型文件放到这个目录下。

## 使用方式

```bash
# 1. 复制你的模型到这里
cp /path/to/your_wakeword.tflite spiffs_content/

# 2. 编译 (模型会自动打包到 SPIFFS 镜像)
idf.py build

# 3. 烧录 (会同时烧录模型分区)
idf.py flash
```

## 注意事项

- 文件名 (不含扩展名) 会作为唤醒词标识符
  - 例如 `xiaona.tflite` → wake_word = `"xiaona"`
- 支持多个模型同时存在 (最多 3 个)
- 模型文件会打包到 `build/models.bin` SPIFFS 镜像
- SPIFFS 分区大小: 1MB (足够放下 76KB 的模型)

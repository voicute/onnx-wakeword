# Model Storage

Place `.tflite` wake word / keyword model files here.

> 把 `.tflite` 模型文件放到这个目录下。

## Usage / 使用

```bash
# Copy your model here
cp /path/to/your_model.tflite spiffs_content/

# Build (model auto-packed into SPIFFS image)
idf.py build

# Flash (includes model partition)
idf.py flash
```

## Notes / 注意

- Filename (without extension) is used as the keyword identifier
  - e.g. `hey_friday.tflite` → keyword = `"hey_friday"`
- Supports up to 3 models simultaneously
- SPIFFS partition size: 1MB

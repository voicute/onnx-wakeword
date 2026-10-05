# Model Storage

Place `.tflite` backbone model files here. See the parent
[README](../README.md) for the full integration guide (model pairing, head
merging, state machine).

> 把 `.tflite` backbone 模型文件放到这个目录下。完整集成说明（模型配对、head 合并、状态机）见上级 [README](../README_CN.md)。

## Current models / 当前模型

- `hallo_lumo.tflite` — wake model ("Hallo Lumo"), single head K=5
- `licht_multi.tflite` — command model ("Licht rot/blau/grün/weiß"), N=4 heads × K=3

## Usage / 使用

```bash
# Copy your model here
cp /path/to/your_model.tflite spiffs_content/

# Build (models auto-packed into SPIFFS image)
idf.py build

# Flash (includes model partition)
idf.py flash
```

## Notes / 注意

- The tflite contains the **backbone only**; the MultiProto head weights live
  in `main/head.h` (merged by `main/_merge_heads.py`). A tflite and head.h are
  a matched pair from the same training run — replace them together.
- Filename (without extension) is the model identifier the firmware resolves
  roles from (`resolve_model_indices()` in main.cpp matches on
  `hallo_lumo` / `licht_multi` prefixes; readdir order is not guaranteed).
- tflite 文件只含 **backbone**；MultiProto 头权重在 `main/head.h`（由
  `main/_merge_heads.py` 合并生成）。tflite 和 head.h 必须出自同一次训练，
  更换时一起换。
- 文件名（去扩展名）就是固件解析角色的标识（main.cpp 的
  `resolve_model_indices()` 按 `hallo_lumo` / `licht_multi` 前缀匹配；
  readdir 顺序不保证）。
- SPIFFS partition size: 1MB / SPIFFS 分区 1MB。

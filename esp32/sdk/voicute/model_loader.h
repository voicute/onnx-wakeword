/**
 * Model Loader — TFLite 模型加载 (从 SPIFFS 分区)
 *
 * 替换原先的 esp-sr WakeNet/MultiNet 加载。
 * 支持多唤醒词: 每个词一个 .tflite 文件 + 一个 model_info.json。
 *
 * SPIFFS 分区文件格式:
 *   /spiffs/manbo.tflite         # 曼波唤醒词模型
 *   /spiffs/nihaodiannao.tflite  # 你好电脑模型
 *   /spiffs/model_info.json      # 模型配置
 *
 * model_info.json 格式:
 *   {"multi_model":true,"models":[{"wake_word":"曼波","model_file":"manbo.tflite",
 *     "emb_frames":1,"cons_frames":2},...]}
 */

#pragma once

#include <stdint.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_WAKE_WORDS 3
#define TFLITE_ARENA_SIZE (64 * 1024)  // Model uses ~54KB; keep arena small enough for internal SRAM

typedef struct {
    char     wake_word[32];          // 唤醒词文本
    char     model_file[64];         // SPIFFS 上的文件名
    int      cons_frames;            // L1 连续帧数
    float    threshold;              // 检测阈值 (默认 0.7)

    // TFLite 运行时
    const tflite::Model       *tflite_model;
    tflite::MicroAllocator    *allocator;       // 该版本 TFLite Micro 必需
    tflite::MicroInterpreter  *interpreter;
    TfLiteTensor              *input_tensor;
    TfLiteTensor              *output_tensor;
    uint8_t                   *arena;           // tensor arena
} wake_model_t;

typedef struct {
    int          num_models;
    wake_model_t models[MAX_WAKE_WORDS];
} model_registry_t;

/**
 * @brief  从 SPIFFS 加载所有唤醒词模型；若 SPIFFS 为空则 fallback 到 compiled_model
 *
 * @param  registry         输出: 模型注册表
 * @param  base_path        SPIFFS 基础路径 (如 "/spiffs")
 * @param  resolver         TFLite OpResolver
 * @param  compiled_model   可选: 编译进固件的模型数据
 * @param  compiled_len     可选: 编译模型的字节数
 * @return 0 成功, -1 失败
 */
int model_loader_init(model_registry_t *registry, const char *base_path,
                       tflite::MicroOpResolver *resolver,
                       const uint8_t *compiled_model, size_t compiled_len);

/**
 * @brief  加载单个 TFLite 模型（从 SPIFFS 文件或 compiled data）
 */
int model_loader_load_one(wake_model_t *m, const char *filepath,
                           tflite::MicroOpResolver *resolver,
                           const uint8_t *compiled_model, size_t compiled_len);

#ifdef __cplusplus
}
#endif

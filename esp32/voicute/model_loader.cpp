/**
 * Model Loader — TFLite 模型加载 (实现)
 */

#include "model_loader.h"
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_heap_caps.h"

// FREE_MODEL_DATA: only free if model was loaded from SPIFFS (not compiled-in Flash)
#define FREE_MODEL_DATA(ptr, is_compiled) do { if (!(is_compiled)) free(ptr); } while(0)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"

static const char *TAG = "ModelLoader";

/* ============================================================
 *  Built-in op code → name 映射 (诊断用)
 * ============================================================ */

static const char *builtin_op_name(int code) {
    switch (code) {
        case  0: return "ADD";
        case  2: return "CONCATENATION";
        case  3: return "CONV_2D";
        case  4: return "DEPTHWISE_CONV_2D";
        case  9: return "FULLY_CONNECTED";
        case 14: return "LOGISTIC";
        case 18: return "MUL";
        case 22: return "RESHAPE";
        case 25: return "SOFTMAX";
        case 32: return "CUSTOM";
        case 34: return "PAD";
        case 37: return "BATCH_TO_SPACE_ND";
        case 38: return "SPACE_TO_BATCH_ND";
        case 39: return "TRANSPOSE";
        case 40: return "MEAN";
        case 42: return "DIV";
        case 45: return "STRIDED_SLICE";
        case 53: return "CAST";
        case 55: return "MAXIMUM";
        case 70: return "EXPAND_DIMS";
        case 74: return "SUM";
        case 75: return "SQRT";
        case 77: return "SHAPE";
        case 83: return "PACK";
        case 92: return "SQUARE";
        case  6: return "DEQUANTIZE";
        case 114: return "CUSTOM_114";
        case 94: return "FILL";
        default:  return "???";
    }
}

/* ============================================================
 *  诊断: 列出模型中所有 op code
 * ============================================================ */

static void dump_model_ops(const tflite::Model *model) {
    auto subgraphs = model->subgraphs();
    if (!subgraphs || subgraphs->size() == 0) {
        ESP_LOGE(TAG, "Model has no subgraphs!");
        return;
    }
    auto sg = (*subgraphs)[0];
    auto opcodes = model->operator_codes();
    auto ops = sg->operators();
    if (!opcodes || !ops) {
        ESP_LOGE(TAG, "Model has no opcodes or operators!");
        return;
    }

    int counts[256] = {0};
    int max_opcode_idx = -1;
    for (size_t i = 0; i < ops->size(); i++) {
        auto op = (*ops)[i];
        int idx = op->opcode_index();
        if (idx >= 0 && idx < 256) {
            counts[idx]++;
            if (idx > max_opcode_idx) max_opcode_idx = idx;
        }
    }

    ESP_LOGI(TAG, "=== Model diagnostic ===");
    ESP_LOGI(TAG, "  ops=%d tensors=%d opcodes=%d",
             (int)ops->size(), (int)sg->tensors()->size(),
             (int)opcodes->size());

    int total_used = 0;
    for (int i = 0; i <= max_opcode_idx; i++) {
        if (counts[i] > 0 && i < (int)opcodes->size()) {
            auto oc = (*opcodes)[i];
            int builtin = oc->builtin_code();
            if (builtin == 32) {
                auto custom = oc->custom_code();
                const char *name = custom ? custom->c_str() : "(null)";
                ESP_LOGI(TAG, "    [%d] CUSTOM(%s): %d uses", i, name, counts[i]);
                total_used++;
            } else {
                ESP_LOGI(TAG, "    [%d] %s(%d): %d uses",
                         i, builtin_op_name(builtin), builtin, counts[i]);
                total_used++;
            }
        }
    }
    ESP_LOGI(TAG, "  Unique opcodes used: %d", total_used);
}

/* ============================================================
 *  SPIFFS 文件读取
 * ============================================================ */

/* Read model: if compiled data is provided use it (in Flash), else read from SPIFFS */
static uint8_t *read_file(const char *path, size_t *out_len, int *is_compiled,
                           const uint8_t *compiled_model, size_t compiled_len) {
    if (compiled_model && compiled_len > 0) {
        ESP_LOGI(TAG, "Using compiled model (%d bytes)", (int)compiled_len);
        *out_len = compiled_len;
        *is_compiled = 1;
        return (uint8_t *)compiled_model;  // in Flash, accessible by NN accelerator
    }
    // SPIFFS fallback
    FILE *f = fopen(path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Cannot open: %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        ESP_LOGE(TAG, "OOM for %s (%d bytes)", path, (int)len);
        fclose(f);
        return NULL;
    }
    size_t rd = fread(buf, 1, len, f);
    fclose(f);
    if (rd != len) {
        ESP_LOGE(TAG, "Short read %s: %d/%d", path, (int)rd, (int)len);
        free(buf);
        return NULL;
    }
    *out_len = len;
    *is_compiled = 0;
    return buf;
}

/* ============================================================
 *  model_loader_load_one — 加载单个 TFLite 模型
 * ============================================================ */

int model_loader_load_one(wake_model_t *m, const char *filepath,
                           tflite::MicroOpResolver *resolver,
                           const uint8_t *compiled_model, size_t compiled_len) {
    // 1. 读取模型
    ESP_LOGI(TAG, "[1/7] Reading model: %s", filepath);
    size_t model_len; int is_compiled = 0;
    uint8_t *model_data = read_file(filepath, &model_len, &is_compiled,
                                     compiled_model, compiled_len);
    if (!model_data) return -1;

    // 2-7: standard TFLite loading
    ESP_LOGI(TAG, "[2/7] GetModel (data=%p len=%d)...", (void*)model_data, (int)model_len);
    m->tflite_model = tflite::GetModel(model_data);
    if (!m->tflite_model) {
        ESP_LOGE(TAG, "  GetModel returned NULL! FlatBuffer parse failed.");
        FREE_MODEL_DATA(model_data, is_compiled);
        return -1;
    }

    int schema_ver = m->tflite_model->version();
    ESP_LOGI(TAG, "  Schema: model=%d  lib=%d  %s",
             schema_ver, TFLITE_SCHEMA_VERSION,
             schema_ver == TFLITE_SCHEMA_VERSION ? "(match)" : "(MISMATCH!)");
    if (schema_ver != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "  FATAL: schema version mismatch!");
        FREE_MODEL_DATA(model_data, is_compiled);
        return -1;
    }

    dump_model_ops(m->tflite_model);

    // 3. Arena
    m->arena = (uint8_t *)heap_caps_aligned_alloc(16, TFLITE_ARENA_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!m->arena) {
        ESP_LOGW(TAG, "Internal OOM, fallback PSRAM");
        m->arena = (uint8_t *)heap_caps_malloc(TFLITE_ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    assert(m->arena);
    ESP_LOGI(TAG, "[3/7] Arena %dKB at %p (%s)", TFLITE_ARENA_SIZE/1024, (void*)m->arena,
             (uint32_t)m->arena >= 0x3C000000 ? "PSRAM" : "Internal SRAM");

    // 4. MicroAllocator
    ESP_LOGI(TAG, "[4/7] MicroAllocator::Create(arena=%p, size=%d KB)...",
             (void*)m->arena, TFLITE_ARENA_SIZE / 1024);
    m->allocator = tflite::MicroAllocator::Create(m->arena, TFLITE_ARENA_SIZE);
    if (!m->allocator) {
        ESP_LOGE(TAG, "  MicroAllocator::Create returned NULL!");
        free(m->arena);
        FREE_MODEL_DATA(model_data, is_compiled);
        return -1;
    }
    ESP_LOGI(TAG, "  MicroAllocator OK (%p)", (void*)m->allocator);

    // 5. MicroInterpreter
    ESP_LOGI(TAG, "[5/7] Creating MicroInterpreter...");
    m->interpreter = new tflite::MicroInterpreter(m->tflite_model, *resolver, m->allocator);
    if (!m->interpreter) {
        ESP_LOGE(TAG, "  MicroInterpreter constructor returned NULL!");
        free(m->arena);
        FREE_MODEL_DATA(model_data, is_compiled);
        return -1;
    }

    // 6. AllocateTensors
    ESP_LOGI(TAG, "[6/7] AllocateTensors...");
    TfLiteStatus status = m->interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "  AllocateTensors FAILED (status=%d)", (int)status);
        delete m->interpreter; m->interpreter = nullptr;
        free(m->arena);
        FREE_MODEL_DATA(model_data, is_compiled);
        return -1;
    }
    ESP_LOGI(TAG, "  AllocateTensors OK, used: %d / %d bytes",
             (int)m->interpreter->arena_used_bytes(), TFLITE_ARENA_SIZE);

    // 7. Tensors
    m->input_tensor  = m->interpreter->input(0);
    m->output_tensor = m->interpreter->output(0);
    if (!m->input_tensor || !m->output_tensor) {
        ESP_LOGE(TAG, "[7/7] input or output tensor is NULL!");
        delete m->interpreter; m->interpreter = nullptr;
        free(m->arena);
        FREE_MODEL_DATA(model_data, is_compiled);
        return -1;
    }

    ESP_LOGI(TAG, "[7/7] Loaded OK");
    return 0;
}

/* ============================================================
 *  model_loader_init — 从 SPIFFS 加载所有模型
 * ============================================================ */

int model_loader_init(model_registry_t *registry, const char *base_path,
                       tflite::MicroOpResolver *resolver,
                       const uint8_t *compiled_model, size_t compiled_len) {
    memset(registry, 0, sizeof(*registry));
    int count = 0;

    ESP_LOGI(TAG, "Scanning %s for .tflite files...", base_path);

    DIR *dir = opendir(base_path);
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && count < MAX_WAKE_WORDS) {
            const char *name = entry->d_name;
            size_t len = strlen(name);
            if (len < 7 || strcmp(name + len - 7, ".tflite") != 0) continue;

            wake_model_t *m = &registry->models[count];
            char fname[64];
            strncpy(fname, name, len - 7);
            fname[len - 7] = '\0';
            strncpy(m->wake_word, fname, sizeof(m->wake_word) - 1);
            m->cons_frames = 3;
            m->threshold = 0.7f;
            strncpy(m->model_file, name, sizeof(m->model_file) - 1);

            char fpath[320];
            snprintf(fpath, sizeof(fpath), "%s/%s", base_path, name);
            ESP_LOGI(TAG, "Found: %s → trying load...", name);
            if (model_loader_load_one(m, fpath, resolver, NULL, 0) == 0) {
                ESP_LOGI(TAG, "  [%d] %s → %s (cons=%d)", count,
                         m->wake_word, m->model_file, m->cons_frames);
                count++;
            } else {
                ESP_LOGE(TAG, "  FAILED to load %s", name);
            }
        }
        closedir(dir);
    }

    // Fallback: if no SPIFFS models found, use compiled model if provided
    if (count == 0 && compiled_model && compiled_len > 0) {
        ESP_LOGW(TAG, "No .tflite files in %s, loading compiled model...", base_path);
        strncpy(registry->models[0].wake_word, "wake", sizeof(registry->models[0].wake_word) - 1);
        strncpy(registry->models[0].model_file, "compiled", sizeof(registry->models[0].model_file) - 1);
        registry->models[0].cons_frames = 1;
        registry->models[0].threshold   = 0.7f;
        if (model_loader_load_one(&registry->models[0], "compiled", resolver,
                                   compiled_model, compiled_len) == 0) {
            count = 1;
        }
    }

    registry->num_models = count;
    ESP_LOGI(TAG, "Loaded %d/%d models", count, MAX_WAKE_WORDS);
    return count > 0 ? 0 : -1;
}

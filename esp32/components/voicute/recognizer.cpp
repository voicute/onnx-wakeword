#include "recognizer.h"
#include "mel_extractor.h"
#include "detect_logic.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

static const char *TAG = "Recognizer";

static model_registry_t    g_registry;
static recognizer_config_t g_cfg;
static float               mel_buffer[MEL_TIME][MEL_N_MELS];
static voice_event_callback_t g_cb = NULL;
static void              *g_cb_ud = NULL;
static dl_state_t         g_dl[MAX_WAKE_WORDS];   // L1-L5 per model

// ---- Init ----
#include "esp_heap_caps.h"
void recognizer_start(const recognizer_config_t *cfg) {
    g_cfg = *cfg;
    memset(&g_registry, 0, sizeof(g_registry));

    // Allocate dedicated 64KB scratch buffer for esp-nn (16-byte aligned)
    static uint8_t *nn_scratch = NULL;
    if (!nn_scratch) {
        nn_scratch = (uint8_t*)heap_caps_malloc(64 * 1024, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGI(TAG, "esp-nn scratch: %p", (void*)nn_scratch);
    }

    // OpResolver — 10 operators for manbo_backbone_int8_FINAL
    static tflite::MicroMutableOpResolver<10> resolver;
    #define R(op) resolver.Add##op()
    R(Conv2D); R(DepthwiseConv2D); R(Pad); R(Add); R(Mul);
    R(Reshape); R(Transpose); R(Slice); R(Sum); R(Concatenation);
    #undef R

    model_loader_init(&g_registry, cfg->model_path, &resolver, NULL, 0);
    mel_extractor_init();

    if (g_registry.num_models == 0) {
        ESP_LOGE(TAG, "No models loaded");
        return;
    }

    // Init L1-L5 detection state per model from config
    for (int i = 0; i < g_registry.num_models; i++) {
        dl_init(&g_dl[i]);
        dl_set_cons_frames(&g_dl[i], g_registry.models[i].cons_frames);
        dl_set_l1_enabled(&g_dl[i], cfg->l1_enabled);
        dl_set_l2_enabled(&g_dl[i], cfg->l2_enabled);
        dl_set_l3_enabled(&g_dl[i], cfg->l3_enabled);
        dl_set_l4_enabled(&g_dl[i], cfg->l4_enabled);
        dl_set_l5_enabled(&g_dl[i], cfg->l5_enabled);
        g_dl[i].l5_delta = cfg->l5_delta > 0 ? cfg->l5_delta : L5_DELTA;
    }

    ESP_LOGI(TAG, "Ready: %d models, thr=%.2f L1=%d L2=%d L3=%d L4=%d L5=%d L5delta=%.0f",
             g_registry.num_models, cfg->threshold,
             cfg->l1_enabled, cfg->l2_enabled, cfg->l3_enabled,
             cfg->l4_enabled, cfg->l5_enabled,
             cfg->l5_delta > 0 ? (double)cfg->l5_delta : (double)L5_DELTA);
}

void recognizer_stop(void) {}
static float g_last_prob = 0.0f;
float recognizer_get_last_prob(void) { return g_last_prob; }

void recognizer_feed_rms(float rms, int64_t now_ms) {
    // Update RMS history for L5 detection (no inference)
    for (int i = 0; i < g_registry.num_models; i++)
        dl_record(&g_dl[i], 0.0f, "", rms, now_ms);
}

void recognizer_evaluate_silence(float rms, int64_t now_ms) {
    // Only record RMS for L5 history (matches Python: record() always runs)
    // evaluate() runs in recognizer_run_frame (inference frames handle L5b)
    for (int i = 0; i < g_registry.num_models; i++)
        dl_record(&g_dl[i], 0.0f, "", rms, now_ms);
}
void recognizer_register_callback(int idx, voice_event_callback_t cb, void *ud) {
    g_cb = cb; g_cb_ud = ud;
}

// ---- Run one frame ----
void recognizer_run_frame(const int16_t *pcm, float rms, int64_t now_ms) {
    if (g_registry.num_models == 0) return;

    // Mel extraction
    if (mel_extract(pcm, mel_buffer) != 0) return;

    for (int m = 0; m < g_registry.num_models; m++) {
        wake_model_t *model = &g_registry.models[m];
        if (!model->interpreter) continue;

        // Fill input — shape [1, 98, 32] time-major
        // Flat layout: mel[t * 32 + f] = mel_buffer[t][f]
        int is_int8 = (model->input_tensor->type == kTfLiteInt8);
        if (is_int8) {
            int8_t *inp = model->input_tensor->data.int8;
            float scale = model->input_tensor->params.scale;
            int zero_point = model->input_tensor->params.zero_point;
            int n = MEL_TIME * MEL_N_MELS;
            // Generic float→int8 quantization (model-independent)
            for (int i = 0; i < n; i++) {
                int q = (int)lrintf(mel_buffer[0][i] / scale) + zero_point;
                if (q < -128) q = -128; else if (q > 127) q = 127;
                inp[i] = (int8_t)q;
            }
        } else {
            float *inp = model->input_tensor->data.f;
            for (int t = 0; t < MEL_TIME; t++)
                for (int f = 0; f < MEL_N_MELS; f++)
                    inp[t * MEL_N_MELS + f] = mel_buffer[t][f];
        }

        // TFLite inference
        int64_t t_invoke_start = esp_timer_get_time();
        TfLiteStatus st = model->interpreter->Invoke();
        int64_t t_invoke_end = esp_timer_get_time();
        if (st != kTfLiteOk) { ESP_LOGE(TAG, "Invoke fail st=%d", (int)st); continue; }

        // Postprocess: backbone output → head → prob (caller-provided)
        float prob = 0.0f;
        if (g_cfg.postprocess) {
            prob = g_cfg.postprocess(
                model->output_tensor->data.int8,
                model->output_tensor->params.scale,
                model->output_tensor->params.zero_point);
        } else {
            prob = model->output_tensor->data.f[0];  // float fallback
        }

        // Store for miss detection diagnostics
        g_last_prob = prob;

        // ── Posterior smoothing: max over last N frames ──
        // Model uses multi-scale pooling biased toward window END.
        // Word at wrong position → single-frame prob can drop to 0.
        // Smoothing catches the "sweet spot" frame within ~1.5s window.
        #define SMOOTH_N 5
        static float prob_hist[MAX_WAKE_WORDS][SMOOTH_N] = {{0}};
        static int   prob_idx[MAX_WAKE_WORDS] = {0};
        prob_hist[m][prob_idx[m] % SMOOTH_N] = prob;
        prob_idx[m]++;
        float prob_smooth = prob;
        if (prob_idx[m] >= SMOOTH_N) {
            prob_smooth = prob_hist[m][0];
            for (int i = 1; i < SMOOTH_N; i++)
                if (prob_hist[m][i] > prob_smooth) prob_smooth = prob_hist[m][i];
        }

        // ── L1-L5 Detection Pipeline ──
        float thr = g_cfg.threshold > 0 ? g_cfg.threshold : model->threshold;

        // Pass empty word for bg frames; wake_word only for candidates
        const char *word = (prob_smooth > thr) ? model->wake_word : "";
        dl_record(&g_dl[m], prob, word, rms, now_ms);

        // Evaluate: uses smoothed prob so L1 continuity doesn't break on single-frame drops
        const char *trigger = dl_evaluate(&g_dl[m], word,
                                          prob_smooth, rms, thr, now_ms);

        // Log every 10 frames, prob>0.05, or trigger; show smoothed prob
        static int cnt = 0; cnt++;
        if (cnt == 1 || cnt % 10 == 0 || prob_smooth > 0.05f || trigger != NULL)
            ESP_LOGI(TAG, "%s: prob=%.4f rms=%.0f (cnt=%d) invoke=%.1fms%s",
                     model->wake_word, (double)prob_smooth, (double)rms, cnt,
                     (double)(t_invoke_end - t_invoke_start) / 1000.0,
                     trigger ? " *TRIG*" : "");

        // Fire callback if detection pipeline confirmed
        if (trigger != NULL && g_cb) {
            voice_evt_data_t evt = { .awaken_channel = 0 };
            g_cb(VOICE_EVT_AWAKEN, evt, g_cb_ud);
        }
    }
}

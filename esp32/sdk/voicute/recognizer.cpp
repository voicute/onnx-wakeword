#include "recognizer.h"
#include "mel_extractor.h"
#include "detect_logic.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"

#define KWS_PROFILE 0
#define KWS_PROFILE_EVERY 40
#if KWS_PROFILE
static tflite::MicroProfiler s_profiler;
#endif

static const char *TAG = "Recognizer";

static model_registry_t    g_registry;
static recognizer_config_t g_cfg;
static float               mel_buffer[MEL_TIME][MEL_N_MELS];
static voice_event_callback_t g_cbs[MAX_WAKE_WORDS] = {0};
static void              *g_cb_uds[MAX_WAKE_WORDS] = {0};
static uint8_t             g_active[MAX_WAKE_WORDS] = {0};  // per-model gate
static dl_state_t         g_dl[MAX_WAKE_WORDS];   // L1-L5 per model

// Posterior smoothing state (max over last SMOOTH_N frames), file-scope so it
// can be cleared on IDLE<->CMD transitions — a frozen peak from before command
// mode otherwise lingers ~3s after returning to IDLE and re-fires the trigger.
#define SMOOTH_N 5
static float prob_hist[MAX_WAKE_WORDS][SMOOTH_N];
static int   prob_idx[MAX_WAKE_WORDS];

// ---- Init ----
#include "esp_heap_caps.h"
void recognizer_start(const recognizer_config_t *cfg) {
    g_cfg = *cfg;
    memset(&g_registry, 0, sizeof(g_registry));

#if CONFIG_NN_SKIP_NUDGE
    ESP_LOGW(TAG, "ESP-NN fast requantize enabled (NN_SKIP_NUDGE); validate +/-1 LSB score impact");
#else
    ESP_LOGI(TAG, "ESP-NN bit-exact requantize enabled");
#endif

    // ESP-NN requests scratch buffers from the Tensor Arena. No separate
    // application-side scratch allocation is needed.

    // Minimal operator set for the NHWC, batch-norm-folded backbone.
    static tflite::MicroMutableOpResolver<10> resolver;
    #define R(op) resolver.Add##op()
    R(Conv2D); R(DepthwiseConv2D); R(Pad); R(Add);
    R(Reshape); R(StridedSlice); R(Mean); R(Concatenation);
    R(Sum); R(Mul);
    #undef R

    model_loader_init(&g_registry, cfg->model_path, &resolver, NULL, 0,
#if KWS_PROFILE
                      &s_profiler
#else
                      nullptr
#endif
    );
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
        g_active[i] = 1;
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

void recognizer_reset_smooth(void) {
    memset(prob_hist, 0, sizeof(prob_hist));
    memset(prob_idx, 0, sizeof(prob_idx));
}

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
    if (idx < 0 || idx >= MAX_WAKE_WORDS) return;
    g_cbs[idx] = cb; g_cb_uds[idx] = ud;
}

void recognizer_set_active(int idx, int enabled) {
    if (idx < 0 || idx >= MAX_WAKE_WORDS) return;
    g_active[idx] = (uint8_t)(enabled != 0);
}

int recognizer_num_models(void) { return g_registry.num_models; }
const char *recognizer_model_word(int idx) {
    if (idx < 0 || idx >= g_registry.num_models) return "";
    return g_registry.models[idx].wake_word;
}

// ---- Run one frame ----
void recognizer_run_frame(const int16_t *pcm, float rms, int64_t now_ms) {
    if (g_registry.num_models == 0) return;

    // Mel extraction
    int64_t t_mel_start = esp_timer_get_time();
    if (mel_extract(pcm, mel_buffer) != 0) return;
    int64_t t_mel_end = esp_timer_get_time();

    for (int m = 0; m < g_registry.num_models; m++) {
        wake_model_t *model = &g_registry.models[m];
        if (!model->interpreter || !g_active[m]) continue;

        // Fill input — shape [1, 98, 32] time-major
        // Flat layout: mel[t * 32 + f] = mel_buffer[t][f]
        int is_int8 = (model->input_tensor->type == kTfLiteInt8);
        if (is_int8) {
            int8_t *inp = model->input_tensor->data.int8;
            float scale = model->input_tensor->params.scale;
            int zero_point = model->input_tensor->params.zero_point;
            // Generic float→int8 quantization (model-independent)
            for (int t = 0; t < MEL_TIME; t++) {
                for (int f = 0; f < MEL_N_MELS; f++) {
                    int i = t * MEL_N_MELS + f;
                    int q = (int)lrintf(mel_buffer[t][f] / scale) + zero_point;
                    if (q < -128) q = -128; else if (q > 127) q = 127;
                    inp[i] = (int8_t)q;
                }
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

#if KWS_PROFILE
        static int profile_count = 0;
        if (++profile_count % KWS_PROFILE_EVERY == 0) {
            ESP_LOGI(TAG, "=== per-op profile (invoke #%d) ===", profile_count);
            s_profiler.Log();
            s_profiler.ClearEvents();
        }
#endif

        // Postprocess: backbone output → head → prob (caller-provided)
        float prob = 0.0f;
        if (g_cfg.postprocess) {
            prob = g_cfg.postprocess(
                model->output_tensor->data.int8,
                model->output_tensor->params.scale,
                model->output_tensor->params.zero_point, m);
        } else {
            prob = model->output_tensor->data.f[0];  // float fallback
        }

        // Store for miss detection diagnostics
        g_last_prob = prob;

        // ── Posterior smoothing: max over last N frames ──
        // Model uses multi-scale pooling biased toward window END.
        // Word at wrong position → single-frame prob can drop to 0.
        // Smoothing catches the "sweet spot" frame within ~1.5s window.
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
            ESP_LOGI(TAG, "%s: prob=%.4f rms=%.0f (cnt=%d) mel=%.1fms invoke=%.1fms%s",
                     model->wake_word, (double)prob_smooth, (double)rms, cnt,
                     (double)(t_mel_end - t_mel_start) / 1000.0,
                     (double)(t_invoke_end - t_invoke_start) / 1000.0,
                     trigger ? " *TRIG*" : "");

        // Fire this model's callback if detection pipeline confirmed
        if (trigger != NULL && g_cbs[m]) {
            voice_evt_data_t evt = { .awaken_channel = 0 };
            g_cbs[m](VOICE_EVT_AWAKEN, evt, g_cb_uds[m]);
        }
    }
}

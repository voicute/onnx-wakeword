/*
 * voice_engine — full app
 * Official feed (bsp_board) → AFE (beamforming + WebRTC NS) → wake word (voicute) + command (voicute).
 * Two own KWS models, strictly time-divided (replaces the former MultiNet):
 *   IDLE    → wake model (Hallo Lumo) runs on the rolling window
 *   COMMAND → wake model gated off, command model (Licht rot/blau/grün/weiß)
 *             runs the same window; word argmax drives the LED color; 6s timeout.
 * Both tflite backbones live in SPIFFS; their float heads are merged into one
 * head.h (see _merge_heads.py) with KWS_WAKE_* / KWS_CMD_* symbols.
 */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/stream_buffer.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_timer.h"
#include "esp_rom_sys.h"
#include "bsp_board.h"
#include "recognizer.h"
#include "head.h"
// Multi-keyword test build: head.h is SELF-CONTAINED (postprocess functions
// live inside it). Do NOT also include kws_postprocess.h — KWS_ABS_TEMP
// would clash (scalar in the single-keyword header vs array in head.h).
#include "esp_process_sdkconfig.h"
#include "afe_processor.h"
#include "led_strip.h"
#include <math.h>
#include <string.h>

#define WAKE_WORD  "Hey Robot"
#define KWS_HOP    2560   // request a fresh latest window every 160 ms at 16 kHz
#define CMD_TIMEOUT_MS 6000

static const char *TAG = "APP";

// Model roles, resolved from SPIFFS filename stems at startup (readdir order
// is not guaranteed, so never hard-code registry indices).
static int g_idx_wake = -1;   // hallo_lumo.tflite
static int g_idx_cmd  = -1;   // licht_multi.tflite

// CMD head outputs of the most recent command-model frame (postprocess writes
// them; the on_cmd callback — fired later in the same run_frame call — reads).
static float g_cmd_probs[KWS_CMD_N];
static int   g_cmd_best = -1;

static float dispatch_postprocess(const int8_t *out_i8, float out_scale,
                                  int out_zero, int model_idx) {
    if (model_idx == g_idx_cmd) {
        // kws_postprocess_cmd returns the head COUNT; argmax is ours to take.
        int n = kws_postprocess_cmd(out_i8, out_scale, out_zero, g_cmd_probs);
        int best = 0;
        for (int i = 1; i < n; i++)
            if (g_cmd_probs[i] > g_cmd_probs[best]) best = i;
        g_cmd_best = best;
        if (g_cmd_probs[best] > 0.05f) {
            printf("[cmd]");
            for (int i = 0; i < n; i++)
                printf(" p%d=%.3f", i, (double)g_cmd_probs[i]);
            printf("\n");
        }
        return g_cmd_probs[best];
    }
    return kws_postprocess_wake(out_i8, out_scale, out_zero);
}

static int mount_spiffs(void) {
    esp_vfs_spiffs_conf_t c = { .base_path="/spiffs", .partition_label="models",
                                .max_files=10, .format_if_mount_failed=false };
    if (esp_vfs_spiffs_register(&c) != ESP_OK) { ESP_LOGE(TAG, "SPIFFS fail"); return -1; }
    size_t t = 0, u = 0; esp_spiffs_info(c.partition_label, &t, &u);
    ESP_LOGI(TAG, "SPIFFS: %d/%d KB", (int)(u / 1024), (int)(t / 1024));
    return 0;
}

// ---- AFE (Audio Front-End) ----
static afe_processor_t *s_afe = NULL;
static int16_t *g_audio_hist = NULL;
static int g_audio_wr = 0;
static int g_audio_count = 0;
static uint32_t g_audio_since_infer = 0;
static SemaphoreHandle_t g_audio_mtx = NULL;

// ---- WS2812 LED driver ----
static led_strip_handle_t g_led = NULL;

static void led_init(void) {
    led_strip_config_t cfg = {}; cfg.strip_gpio_num = 38; cfg.max_leds = 7;
    cfg.led_model = LED_MODEL_WS2812;
    led_strip_rmt_config_t rmt = {}; rmt.clk_src = RMT_CLK_SRC_DEFAULT;
    rmt.resolution_hz = 10 * 1000 * 1000;
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&cfg, &rmt, &g_led));
    ESP_ERROR_CHECK(led_strip_clear(g_led));
}
static void led_show(uint8_t r, uint8_t g, uint8_t b) {
    // This board's WS2812 chain renders R/G swapped relative to set_pixel's
    // argument order (the former MultiNet path pre-swapped per colour, e.g.
    // the red command wrote green values). Swap here once so callers pass
    // logical RGB.
    for (int i = 0; i < 7; i++) ESP_ERROR_CHECK(led_strip_set_pixel(g_led, i, g, r, b));
    ESP_ERROR_CHECK(led_strip_refresh(g_led));
}

// ---- State machine (wake model + command model, time-divided) ----
typedef enum { STATE_IDLE, STATE_COMMAND } state_t;
static state_t g_state = STATE_IDLE;
static int64_t g_cmd_deadline_ms = 0;
static const char *g_cmd_names[KWS_CMD_N] = {
    "Licht rot", "Licht blau", "Licht grün", "Licht weiß" };

typedef enum { LED_OFF, LED_BLINK, LED_SOLID } led_mode_t;
static led_mode_t g_led_mode = LED_OFF;
static int g_led_blink_cnt = 0;
static uint8_t g_led_r, g_led_g, g_led_b;

static void led_set_mode(led_mode_t m, uint8_t r, uint8_t g, uint8_t b) {
    g_led_mode = m; g_led_r = r; g_led_g = g; g_led_b = b; g_led_blink_cnt = 0;
    if (m == LED_OFF) ESP_ERROR_CHECK(led_strip_clear(g_led));
    else if (m == LED_SOLID) led_show(r, g, b);
}
static void led_tick(void) {
    if (g_led_mode != LED_BLINK) return;
    // Time-based half-period (250 ms): the detect loop cadence varies between
    // states (an inference frame adds ~120 ms), so loop counting would make
    // the blink visibly slower in COMMAND mode.
    static int64_t last_toggle_ms = 0;
    int64_t now = esp_timer_get_time() / 1000;
    if (now - last_toggle_ms < 250) return;
    last_toggle_ms = now;
    static int on = 0;
    if (on) led_show(g_led_r, g_led_g, g_led_b);
    else ESP_ERROR_CHECK(led_strip_clear(g_led));
    on = !on;
}
static void resolve_model_indices(void) {
    for (int i = 0; i < recognizer_num_models(); i++) {
        const char *w = recognizer_model_word(i);
        if (!strncmp(w, "hallo_lumo", 10)) g_idx_wake = i;
        else if (!strncmp(w, "licht_multi", 11)) g_idx_cmd = i;
    }
    ESP_LOGI(TAG, "models: wake=%d cmd=%d (of %d)",
             g_idx_wake, g_idx_cmd, recognizer_num_models());
    if (g_idx_wake < 0 || g_idx_cmd < 0)
        ESP_LOGE(TAG, "FATAL: expected hallo_lumo.tflite + licht_multi.tflite in SPIFFS");
}

// led_off: timeout reverts to dark; a matched command keeps the requested
// color solid (same behaviour as the former MultiNet command path).
static void leave_cmd(const char *reason, bool led_off) {
    g_state = STATE_IDLE;
    recognizer_set_active(g_idx_cmd, 0);
    recognizer_set_active(g_idx_wake, 1);
    // Clear smoothing so a prob peak near the transition can't re-fire
    // either model right after the switch.
    recognizer_reset_smooth();
    if (led_off) led_set_mode(LED_OFF, 0, 0, 0);
    // else: keep whatever the command handler lit up
    ESP_LOGI(TAG, "CMD -> IDLE (%s)", reason);
}

static void on_wake(voice_event_t ev, voice_evt_data_t d, void *u) {
    (void)ev; (void)d; (void)u;
    if (g_state != STATE_IDLE) return;
    g_state = STATE_COMMAND;
    recognizer_set_active(g_idx_wake, 0);
    recognizer_set_active(g_idx_cmd, 1);
    recognizer_reset_smooth();
    g_cmd_deadline_ms = esp_timer_get_time() / 1000 + CMD_TIMEOUT_MS;
    led_set_mode(LED_BLINK, 0, 0, 255);
    ESP_LOGI(TAG, ">>> WAKE -> CMD mode");
}

static void on_cmd(voice_event_t ev, voice_evt_data_t d, void *u) {
    (void)ev; (void)d; (void)u;
    if (g_state != STATE_COMMAND || g_cmd_best < 0) return;
    int id = g_cmd_best;
    switch (id) {
        case 0: led_set_mode(LED_SOLID, 255, 0, 0); break;     // rot
        case 1: led_set_mode(LED_SOLID, 0, 0, 255); break;     // blau
        case 2: led_set_mode(LED_SOLID, 0, 255, 0); break;     // grün
        case 3: led_set_mode(LED_SOLID, 255, 255, 255); break; // weiß
    }
    // printf, not esp_rom_printf: the ROM console is lost to UART0 on the
    // USB_SERIAL_JTAG console (only usable if USB was attached at reset)
    printf("\n*** CMD #%d: %s (prob=%.3f) ***\n\n",
        id, g_cmd_names[id], (double)g_cmd_probs[id]);
    fflush(stdout);
    leave_cmd("matched", false);
}

// ---- LIVE MODE with AFE beamforming ----
static void feed_task(void *arg) {
    int cs = afe_processor_feed_chunksize(s_afe);
    int nc = afe_processor_feed_channels(s_afe);
    int16_t *buf = (int16_t*)heap_caps_malloc(cs * nc * 2, MALLOC_CAP_SPIRAM);
    assert(buf);
    while (1) {
        esp_get_feed_data(true, buf, cs * nc * 2);
        afe_processor_feed(s_afe, buf, cs * nc * 2);
    }
}

// Sole owner of AFE fetch. It must keep draining while TFLite Invoke runs.
static void audio_fetch_task(void *arg) {
    int16_t afe_out[1024];
    while (1) {
        int n = afe_processor_fetch(s_afe, afe_out, 1024);
        if (n <= 0) { vTaskDelay(pdMS_TO_TICKS(2)); continue; }

        xSemaphoreTake(g_audio_mtx, portMAX_DELAY);
        for (int i = 0; i < n; i++) {
            g_audio_hist[g_audio_wr++] = afe_out[i];
            if (g_audio_wr == MEL_AUDIO_LEN) g_audio_wr = 0;
        }
        g_audio_count = (g_audio_count + n < MEL_AUDIO_LEN)
            ? g_audio_count + n : MEL_AUDIO_LEN;
        g_audio_since_infer += n;
        xSemaphoreGive(g_audio_mtx);

        // fetch() can keep returning buffered frames immediately. Yield so the
        // core idle task can service the watchdog without risking AFE backlog.
        vTaskDelay(2);
    }
}

static bool snapshot_latest_audio(int16_t *dst) {
    bool ready = false;
    xSemaphoreTake(g_audio_mtx, portMAX_DELAY);
    if (g_audio_count == MEL_AUDIO_LEN && g_audio_since_infer >= KWS_HOP) {
        int tail = MEL_AUDIO_LEN - g_audio_wr;
        memcpy(dst, g_audio_hist + g_audio_wr, tail * sizeof(int16_t));
        if (g_audio_wr > 0)
            memcpy(dst + tail, g_audio_hist, g_audio_wr * sizeof(int16_t));
        // Always infer on the newest continuous window; do not replay backlog.
        g_audio_since_infer = 0;
        ready = true;
    }
    xSemaphoreGive(g_audio_mtx);
    return ready;
}

static void detect_loop(void *arg) {
    // Diagnostic baseline: bypass L1-L5 and let the model threshold be the
    // only wake decision. Re-enable the gates one at a time after validating
    // recall and false-positive behaviour on the repaired audio path.
    recognizer_config_t cfg = { .model_path="/spiffs", .threshold=0.70f,
        .l1_enabled=0,.l2_enabled=0,.l3_enabled=0,.l4_enabled=0,.l5_enabled=0,.l5_delta=200.0f,
        .postprocess=dispatch_postprocess };
    recognizer_start(&cfg);
    resolve_model_indices();
    recognizer_register_callback(g_idx_wake, on_wake, NULL);
    recognizer_register_callback(g_idx_cmd, on_cmd, NULL);
    recognizer_set_active(g_idx_cmd, 0);   // CMD model only listens after wake
    ESP_LOGI(TAG, "=== AFE Live Ready (wake=hallo_lumo cmd=licht_multi) ===");

    led_init();
    int16_t *pcm = (int16_t*)heap_caps_malloc(MEL_AUDIO_LEN*2, MALLOC_CAP_SPIRAM);
    assert(pcm);

    int loop_cnt=0;
    int infer_cnt = 0;
    while (1) {
        if (g_state == STATE_IDLE) {
            // Rolling window: append AFE chunks; a full MEL_AUDIO_LEN window is
            // fed to the recognizer, then the window slides by KWS_HOP.
            // (AFE fetch delivers small chunks — a single fetch is never MEL_AUDIO_LEN.)
            if (snapshot_latest_audio(pcm)) {
                float rms = 0; int64_t sq = 0;
                for (int i=0; i<MEL_AUDIO_LEN; i++) sq += (int64_t)pcm[i]*pcm[i];
                rms = sqrtf((float)(sq/MEL_AUDIO_LEN));
                if (rms >= 0.0f) {  // multi test build: infer continuously to measure latency
                    recognizer_run_frame(pcm, rms, esp_timer_get_time()/1000);
                    if (++infer_cnt % 3 == 0)
                        ESP_LOGI(TAG, "kws prob=%.3f rms=%.0f", recognizer_get_last_prob(), rms);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(20));
        } else {
            // COMMAND: the Licht model runs the same rolling window (wake model
            // is gated off). Trigger fires on_cmd; deadline reverts to IDLE.
            led_tick();
            if (esp_timer_get_time()/1000 >= g_cmd_deadline_ms) {
                leave_cmd("timeout", true);
                continue;
            }
            if (snapshot_latest_audio(pcm)) {
                float rms = 0; int64_t sq = 0;
                for (int i=0; i<MEL_AUDIO_LEN; i++) sq += (int64_t)pcm[i]*pcm[i];
                rms = sqrtf((float)(sq/MEL_AUDIO_LEN));
                recognizer_run_frame(pcm, rms, esp_timer_get_time()/1000);
            }
            vTaskDelay(pdMS_TO_TICKS(20));
        }
        if (++loop_cnt%200==0) ESP_LOGI(TAG, "loop#%d state=%d", loop_cnt, g_state);
    }
}

extern "C" void app_main() {
    ESP_ERROR_CHECK(esp_board_init(16000,2,16));
    afe_processor_config_t afe_cfg = {};
    afe_cfg.ns_init = true;
    s_afe = afe_processor_init(&afe_cfg);  // RMNM + WebRTC NS
    assert(s_afe);

    g_audio_hist = (int16_t*)heap_caps_malloc(
        MEL_AUDIO_LEN * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    g_audio_mtx = xSemaphoreCreateMutex();
    assert(g_audio_hist && g_audio_mtx);

    if (mount_spiffs()!=0) return;
    xTaskCreatePinnedToCore(feed_task,"feed",6*1024,NULL,7,NULL,0);
    // Keep the complete AFE pipeline on core 0 so TFLite can run on core 1
    // without being preempted by the higher-priority fetch task.
    xTaskCreatePinnedToCore(audio_fetch_task,"afe_fetch",6*1024,NULL,6,NULL,0);
    xTaskCreatePinnedToCore(detect_loop,"kws_detect",12*1024,NULL,5,NULL,1);
    while(1) vTaskDelay(pdMS_TO_TICKS(1000));
}

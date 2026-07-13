#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_timer.h"
#include "bsp_board.h"
#include "recognizer.h"
#include "head.h"              // model-specific head weights (user-provided)
#include "kws_postprocess.h"  // generic postprocess
#include "esp_mn_iface.h"           // MultiNet command recognition
#include "esp_mn_models.h"
#include "esp_mn_speech_commands.h"  // esp_mn_commands_add
#include "model_path.h"
#include "esp_process_sdkconfig.h"
#include <math.h>
#include <string.h>

// ── User configuration ──
#define WAKE_WORD  "关键词"   // 唤醒词显示名 (仅用于日志, 不影响模型检测)

#define TEST_MODE 0  // 0=live, 1=PCM test, 2=PCM dump
#if TEST_MODE == 1
#include "test_pcm.h"  // 63KB, only for PCM test
#endif

static const char *TAG = "APP";

// (ring buffer allocated in PSRAM at runtime)

// ---- SPIFFS ----
static int mount_spiffs(void) {
    esp_vfs_spiffs_conf_t c = { .base_path="/spiffs", .partition_label="models",
                                .max_files=10, .format_if_mount_failed=false };
    if (esp_vfs_spiffs_register(&c) != ESP_OK) { ESP_LOGE(TAG, "SPIFFS fail"); return -1; }
    size_t t = 0, u = 0; esp_spiffs_info(c.partition_label, &t, &u);
    ESP_LOGI(TAG, "SPIFFS: %d/%d KB", (int)(u / 1024), (int)(t / 1024));
    return 0;
}

// ---- Ring buffer (declared early for state machine access) ----
#define RB_CAP 24576
static int16_t g_rb[RB_CAP] __attribute__((aligned(16)));
static volatile int g_rb_w = 0;
static volatile int g_rb_r = 0;

// ---- WS2812 LED driver (exact match to Demo's rgb_led_driver) ----
#include "led_strip.h"
static led_strip_handle_t g_led = NULL;

static void led_init(void) {
    led_strip_config_t cfg = {};
    cfg.strip_gpio_num = 38;
    cfg.max_leds = 7;
    cfg.led_model = LED_MODEL_WS2812;
    led_strip_rmt_config_t rmt = {};
    rmt.clk_src = RMT_CLK_SRC_DEFAULT;
    rmt.resolution_hz = 10 * 1000 * 1000;
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&cfg, &rmt, &g_led));
    ESP_ERROR_CHECK(led_strip_clear(g_led));
}

static void led_show(uint8_t r, uint8_t g, uint8_t b) {
    for (int i = 0; i < 7; i++)
        ESP_ERROR_CHECK(led_strip_set_pixel(g_led, i, r, g, b));
    ESP_ERROR_CHECK(led_strip_refresh(g_led));
}

static void led_on_cmd(int cmd_id) {
    switch (cmd_id) {
        case 0: led_show(0, 255, 0);   break; // 红(swapped)
        case 1: led_show(0, 0, 255);   break; // 蓝
        case 2: led_show(255, 0, 0);   break; // 绿(swapped)
        case 3: led_show(255, 255, 255); break; // 白
        default: break;
    }
}

// ---- State machine + MultiNet (matches Demo pattern) ----
typedef enum { STATE_IDLE, STATE_COMMAND } state_t;
static state_t g_state = STATE_IDLE;

static esp_mn_iface_t    *g_multinet = NULL;
static model_iface_data_t *g_mn_data = NULL;
static int g_mn_chunksize = 0;
static int16_t *g_mn_buf = NULL;

// LED modes (matching Demo's rgb_led_driver)
typedef enum { LED_OFF, LED_BLINK, LED_SOLID } led_mode_t;
static led_mode_t g_led_mode = LED_OFF;
static int g_led_blink_cnt = 0;
static uint8_t g_led_r, g_led_g, g_led_b;

static void led_set_mode(led_mode_t mode, uint8_t r, uint8_t g, uint8_t b) {
    g_led_mode = mode;
    g_led_r = r; g_led_g = g; g_led_b = b;
    g_led_blink_cnt = 0;
    if (mode == LED_OFF) {
        ESP_ERROR_CHECK(led_strip_clear(g_led));
    } else if (mode == LED_SOLID) {
        led_show(r, g, b);
    }
}

static void led_tick(void) {
    if (g_led_mode != LED_BLINK) return;
    g_led_blink_cnt++;
    if (g_led_blink_cnt >= 3) {  // toggle every ~3 COMMAND loops
        g_led_blink_cnt = 0;
        static int on = 0;
        if (on) { led_show(g_led_r, g_led_g, g_led_b); }
        else    { ESP_ERROR_CHECK(led_strip_clear(g_led)); }
        on = !on;
    }
}

// Command name table (matches factory_01 demo sdkconfig)
static const char *g_cmd_name(int id) {
    switch (id) {
        case 0: return "灯光变成红色";
        case 1: return "灯光变成蓝色";
        case 2: return "灯光变成绿色";
        case 3: return "灯光变成白色";
        default: return "?";
    }
}

static void mn_init(void) {
    srmodel_list_t *models = esp_srmodel_init("model");
    char *mn_name = esp_srmodel_filter(models, ESP_MN_PREFIX, ESP_MN_CHINESE);
    ESP_LOGI(TAG, "MultiNet: %s", mn_name);
    g_multinet = esp_mn_handle_from_name(mn_name);
    g_mn_data = g_multinet->create(mn_name, 6000);
    esp_mn_commands_update_from_sdkconfig(g_multinet, g_mn_data);
    // Add short aliases for convenience
    esp_mn_commands_add(0, "deng guang bian hong se");
    esp_mn_commands_add(1, "deng guang bian lan se");
    esp_mn_commands_add(2, "deng guang bian lv se");
    esp_mn_commands_add(3, "deng guang bian bai se");
    g_multinet->print_active_speech_commands(g_mn_data);
    g_mn_chunksize = g_multinet->get_samp_chunksize(g_mn_data);
    ESP_LOGI(TAG, "MultiNet ready: chunksize=%d sr=%d",
             g_mn_chunksize, g_multinet->get_samp_rate(g_mn_data));
    g_mn_buf = (int16_t*)heap_caps_malloc(g_mn_chunksize * 2, MALLOC_CAP_SPIRAM);
    assert(g_mn_buf);
}

static void on_trigger(voice_event_t ev, voice_evt_data_t d, void *u) {
    if (ev != VOICE_EVT_AWAKEN) return;
    if (g_state == STATE_IDLE) {
        g_state = STATE_COMMAND;
        if (g_multinet) g_multinet->clean(g_mn_data);
        led_set_mode(LED_BLINK, 0, 0, 255);  // blink blue = listening
        ESP_LOGI(TAG, ">>> WAKE → enter CMD mode");
    }
}

#if TEST_MODE == 1
// ---- TEST MODE 1: single-frame test with recorded PCM ----
static void test_task(void *arg) {
    recognizer_config_t cfg = { .model_path = "/spiffs", .threshold = 0.0f,
                                .l1_enabled=0,.l2_enabled=0,.l3_enabled=0,
                                .l4_enabled=0,.l5_enabled=0,.l5_delta=1200.0f };
    recognizer_start(&cfg);
    ESP_LOGI(TAG, "=== TEST: 1 frame ===");
    int16_t *pcm = (int16_t*)heap_caps_malloc(MEL_AUDIO_LEN*2, MALLOC_CAP_SPIRAM);
    assert(pcm);
    memcpy(pcm, test_pcm, MEL_AUDIO_LEN * 2);
    float rms = 0;
    for (int i = 0; i < MEL_AUDIO_LEN; i++) rms += (float)pcm[i] * (float)pcm[i];
    rms = sqrtf(rms / MEL_AUDIO_LEN);
    recognizer_run_frame(pcm, rms);
    free(pcm);
    ESP_LOGI(TAG, "=== TEST DONE ===");
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}
#elif TEST_MODE == 2
// ---- TEST MODE 2: record 5s, dump as decimal text lines ----
#include "esp_rom_sys.h"
static void record_task(void *arg) {
    int ch_cnt = esp_get_feed_channel();
    int chunk = 512, total = 16000 * 5;  // 5 seconds
    int16_t *raw = (int16_t*)malloc(chunk * ch_cnt * 2);
    int16_t *pcm = (int16_t*)heap_caps_malloc(total * 2, MALLOC_CAP_SPIRAM);
    assert(raw && pcm);
    // 3-second countdown so user can prepare
    ESP_LOGI(TAG, "=== REC: 3... ==="); vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "=== REC: 2... ==="); vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "=== REC: 1... ==="); vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "=== REC: SPEAK NOW! 5s ===");
    int idx = 0;
    for (int i = 0; i < total; i += chunk) {
        esp_get_feed_data(true, raw, chunk * ch_cnt * 2);
        for (int j = 0; j < chunk && idx < total; j++)
            pcm[idx++] = raw[ch_cnt * j + 1];
    }
    // Dump: "S%d\n" per sample for fast parsing (just the int value)
    esp_rom_printf("START\n");
    for (int i = 0; i < total; i++)
        esp_rom_printf("%d\n", (int)pcm[i]);
    esp_rom_printf("END\n");
    free(raw); free(pcm);
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}
#else
// ---- LIVE MODE: feed(Core0,pri7) NEVER blocks + detect(Core1,pri5) ----

static void feed_task(void *arg) {
    esp_rom_printf("FEED_START\n");
    int ch = esp_get_feed_channel();
    int chunk = 512;
    int16_t *raw = (int16_t*)malloc(chunk * ch * 2);
    esp_rom_printf("FEED ch=%d raw=%p\n", ch, (void*)raw);
    assert(raw);
    int cnt = 0;
    while (1) {
        esp_get_feed_data(true, raw, chunk * ch * 2);
        int w = g_rb_w;
        for (int i = 0; i < chunk; i++) { g_rb[w % RB_CAP] = raw[ch * i + 1]; w++; }
        __sync_synchronize();  // memory barrier: flush writes before updating w
        g_rb_w = w;
    }
}

static void detect_loop(void *arg) {
    // ── ESP32-S3 Wake Word Detection Config ──────────────────
    // L1 (consecutive frames): OFF — ESP32 inference ~400ms/frame,
    //   wake word ~500ms → at most 1 frame catches it. Multi-frame
    //   confirmation is not viable at this inference speed.
    // L2 (peak/bg ratio): ON — core filter, works on prob history.
    // L3 (cooldown 1.5s): ON — prevents double-trigger.
    // L4 (burst block): OFF — optional, enable if looping triggers.
    // L5 (energy jump): ON — critical for blocking music/video/TV
    //   false triggers. Uses additive delta (curRms > preMin+200).
    // ──────────────────────────────────────────────────────────
    recognizer_config_t cfg = { .model_path = "/spiffs", .threshold = 0.70f,
                                .l1_enabled=0,.l2_enabled=0,.l3_enabled=0,
                                .l4_enabled=0,.l5_enabled=1,.l5_delta=200.0f,
                                .postprocess = kws_postprocess };
    recognizer_start(&cfg);
    recognizer_register_callback(0, (voice_event_callback_t)on_trigger, (void *)WAKE_WORD);
    ESP_LOGI(TAG, "=== Ready (wake+cmd) ===");
    while ((g_rb_w - g_rb_r) < MEL_AUDIO_LEN) vTaskDelay(pdMS_TO_TICKS(10));
    ESP_LOGI(TAG, "Buffer ready: %d", g_rb_w - g_rb_r);

    led_init();

    // Static PCM buffer — allocate once, reuse (matches Demo pattern)
    int16_t *pcm = (int16_t*)heap_caps_malloc(MEL_AUDIO_LEN * 2, MALLOC_CAP_SPIRAM);
    if (!pcm) {
        ESP_LOGE(TAG, "FATAL: cannot allocate PCM buffer");
        vTaskDelete(NULL);
        return;
    }

    // Init MultiNet for command recognition
    mn_init();

    int loop_cnt = 0;
    while (1) {
        __sync_synchronize();
        int avail = g_rb_w - g_rb_r;

        if (g_state == STATE_IDLE) {
            // ── IDLE: TFLite wake word detection ──
            int64_t now_ms2 = esp_timer_get_time() / 1000;
            // Quick RMS every loop for L5 history (~50ms cadence, independent of inference)
            float rms_quick = 0; int64_t sq_q = 0;
            for (int i = 0; i < 512; i++) { int16_t s = g_rb[((g_rb_w-512+i) % RB_CAP)]; sq_q += (int64_t)s*s; }
            rms_quick = sqrtf((float)(sq_q/512));
            recognizer_evaluate_silence(rms_quick, now_ms2);

            if (avail < MEL_AUDIO_LEN + 1280) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
            g_rb_r = g_rb_w - MEL_AUDIO_LEN - 1280;  // sync, not accumulate
            int start = g_rb_w - MEL_AUDIO_LEN;
            float rms = 0; int64_t sum_sq = 0;
            for (int i = 0; i < MEL_AUDIO_LEN; i++) { int16_t s = g_rb[(start+i) % RB_CAP]; sum_sq += (int64_t)s*s; }
            rms = sqrtf((float)(sum_sq / MEL_AUDIO_LEN));
            if (rms < 5.0f) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }
            for (int i = 0; i < MEL_AUDIO_LEN; i++) pcm[i] = g_rb[(start+i) % RB_CAP];
            recognizer_run_frame(pcm, rms, now_ms2);
            // No extra delay — run inference at max rate (~310ms/frame)
            // Provides denser sliding windows for better alignment with model's pooling sweet spot
        } else {
            // ── COMMAND: MultiNet recognition (feed latest audio) ──
            led_tick();  // blink animation
            if (avail < g_mn_chunksize + 1280) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
            // Take latest audio from write head (same pattern as TFLite in IDLE)
            int start = g_rb_w - g_mn_chunksize;
            for (int i = 0; i < g_mn_chunksize; i++)
                g_mn_buf[i] = g_rb[(start + i) % RB_CAP];
            g_rb_r += g_mn_chunksize;

            // Record RMS for L5 history (at audio data rate, ~32ms)
            float rms_mn = 0; int64_t sq_mn = 0;
            for (int i = 0; i < g_mn_chunksize; i++) { sq_mn += (int64_t)g_mn_buf[i] * g_mn_buf[i]; }
            rms_mn = sqrtf((float)(sq_mn / g_mn_chunksize));
            recognizer_feed_rms(rms_mn, esp_timer_get_time() / 1000);

            esp_mn_state_t st = g_multinet->detect(g_mn_data, g_mn_buf);

            if (st == ESP_MN_STATE_DETECTED) {
                esp_mn_results_t *r = g_multinet->get_results(g_mn_data);
                int cmd_id = r->command_id[0];
                led_on_cmd(cmd_id);         // solid color
                esp_rom_printf("\n*** CMD #%d: %s (prob=%.3f) ***\n\n",
                         cmd_id, g_cmd_name(cmd_id), (double)r->prob[0]);
                g_state = STATE_IDLE;
                g_rb_r = g_rb_w - MEL_AUDIO_LEN - 1280;  // sync for IDLE
                ESP_LOGI(TAG, ">>> CMD %s → back to IDLE", g_cmd_name(cmd_id));
            } else if (st == ESP_MN_STATE_TIMEOUT) {
                led_set_mode(LED_OFF, 0, 0, 0);  // off
                g_state = STATE_IDLE;
                g_rb_r = g_rb_w - MEL_AUDIO_LEN - 1280;  // sync for IDLE
                ESP_LOGI(TAG, ">>> CMD TIMEOUT → back to IDLE");
            }
            // ESP_MN_STATE_DETECTING: continue feeding
        }

        if (++loop_cnt % 100 == 0)
            ESP_LOGI(TAG, "loop#%d avail=%d state=%d", loop_cnt, g_rb_w - g_rb_r, g_state);
        // IDLE delay is inside the branch above; COMMAND runs tight
    }
}
#endif

// ---- Entry ----
extern "C" void app_main() {
#if TEST_MODE == 1
    ESP_ERROR_CHECK(esp_board_init(16000, 2, 16));
    if (mount_spiffs() != 0) return;
    xTaskCreatePinnedToCore(test_task, "test", 16 * 1024, NULL, 5, NULL, 1);
#elif TEST_MODE == 2
    ESP_ERROR_CHECK(esp_board_init(16000, 2, 16));
    xTaskCreatePinnedToCore(record_task, "record", 32 * 1024, NULL, 5, NULL, 1);
#else
    ESP_ERROR_CHECK(esp_board_init(16000, 2, 16));
    {
        int ch = esp_get_feed_channel(); int16_t *b = (int16_t *)malloc(512 * ch * 2);
        int64_t sq = 0; int pk = 0;
        for (int i = 0; i < 16000 / 512; i++) {
            esp_get_feed_data(true, b, 512 * ch * 2);
            for (int j = 0; j < 512; j++) { int16_t s = b[ch * j + 1]; sq += (int64_t)s * s; if (abs(s) > pk) pk = abs(s); }
        }
        free(b); ESP_LOGI(TAG, "Mic: peak=%d RMS=%.0f", pk, sqrt((double)sq / 16000));
    }
    if (mount_spiffs() != 0) return;
    xTaskCreatePinnedToCore(feed_task, "feed", 6 * 1024, NULL, 7, NULL, 0);
    xTaskCreatePinnedToCore(detect_loop, "kws_detect", 12 * 1024, NULL, 5, NULL, 1);
#endif
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}

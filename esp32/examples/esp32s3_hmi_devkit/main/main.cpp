/*
 * voice_engine — full app
 * Official feed (bsp_board) → AFE (beamforming + WebRTC NS) → wake word (voicute) + command (MultiNet)
 * Build target: menuconfig → "Voice Engine App" → "Full app" (default).
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
#include "kws_postprocess.h"
#include "esp_mn_iface.h"
#include "esp_mn_models.h"
#include "esp_mn_speech_commands.h"
#include "model_path.h"
#include "esp_process_sdkconfig.h"
#include "afe_processor.h"
#include "led_strip.h"
#include <math.h>
#include <string.h>

#define WAKE_WORD  "Hey Robot"
#define KWS_HOP    2560   // request a fresh latest window every 160 ms at 16 kHz

static const char *TAG = "APP";

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
static StreamBufferHandle_t g_cmd_stream = NULL;

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
    for (int i = 0; i < 7; i++) ESP_ERROR_CHECK(led_strip_set_pixel(g_led, i, r, g, b));
    ESP_ERROR_CHECK(led_strip_refresh(g_led));
}
static void led_on_cmd(int id) {
    switch (id) {
        case 0: led_show(0, 255, 0); break;
        case 1: led_show(0, 0, 255); break;
        case 2: led_show(255, 0, 0); break;
        case 3: led_show(255, 255, 255); break;
    }
}

// ---- State machine + MultiNet ----
typedef enum { STATE_IDLE, STATE_COMMAND } state_t;
static state_t g_state = STATE_IDLE;
static esp_mn_iface_t    *g_multinet = NULL;
static model_iface_data_t *g_mn_data = NULL;
static int g_mn_chunksize = 0;
static int16_t *g_mn_buf = NULL;

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
    if (++g_led_blink_cnt >= 3) { g_led_blink_cnt = 0;
        static int on = 0;
        if (on) led_show(g_led_r, g_led_g, g_led_b);
        else ESP_ERROR_CHECK(led_strip_clear(g_led));
        on = !on;
    }
}
static const char *g_cmd_name(int id) {
    switch (id) {
        case 0: return "turn the light red"; case 1: return "turn the light blue";
        case 2: return "turn the light green"; case 3: return "turn the light white";
        default: return "?";
    }
}
static void mn_init(void) {
    srmodel_list_t *models = esp_srmodel_init("model");
    char *mn_name = esp_srmodel_filter(models, ESP_MN_PREFIX, ESP_MN_ENGLISH);
    ESP_LOGI(TAG, "MultiNet: %s", mn_name);
    g_multinet = esp_mn_handle_from_name(mn_name);
    g_mn_data = g_multinet->create(mn_name, 6000);
    ESP_ERROR_CHECK(esp_mn_commands_alloc(g_multinet, g_mn_data));
    ESP_ERROR_CHECK(esp_mn_commands_phoneme_add(
        0, "turn the light red", "TkN jc LiT RfD"));
    ESP_ERROR_CHECK(esp_mn_commands_phoneme_add(
        1, "turn the light blue", "TkN jc LiT BLo"));
    ESP_ERROR_CHECK(esp_mn_commands_phoneme_add(
        2, "turn the light green", "TkN jc LiT GRmN"));
    ESP_ERROR_CHECK(esp_mn_commands_phoneme_add(
        3, "turn the light white", "TkN jc LiT WiT"));
    esp_mn_commands_update();
    esp_mn_commands_print();
    g_mn_chunksize = g_multinet->get_samp_chunksize(g_mn_data);
    ESP_LOGI(TAG, "MultiNet ready: chunksize=%d", g_mn_chunksize);
    g_mn_buf = (int16_t*)heap_caps_malloc(g_mn_chunksize * 2, MALLOC_CAP_SPIRAM);
    assert(g_mn_buf);
}
static void on_trigger(voice_event_t ev, voice_evt_data_t d, void *u) {
    if (ev != VOICE_EVT_AWAKEN) return;
    if (g_state == STATE_IDLE) { g_state = STATE_COMMAND;
        // Inferences pause in CMD mode; clear smoothing so the pre-wake prob
        // peak can't linger and re-trigger right after returning to IDLE.
        recognizer_reset_smooth();
        if (g_multinet) g_multinet->clean(g_mn_data);
        led_set_mode(LED_BLINK, 0, 0, 255);
        ESP_LOGI(TAG, ">>> WAKE -> CMD mode"); }
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

        if (g_state == STATE_COMMAND)
            xStreamBufferSend(g_cmd_stream, afe_out, n * sizeof(int16_t), 0);

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
        .postprocess=kws_postprocess };
    recognizer_start(&cfg);
    recognizer_register_callback(0, (voice_event_callback_t)on_trigger, (void*)WAKE_WORD);
    ESP_LOGI(TAG, "=== AFE Live Ready (wake+cmd) ===");

    led_init();
    int16_t *pcm = (int16_t*)heap_caps_malloc(MEL_AUDIO_LEN*2, MALLOC_CAP_SPIRAM);
    assert(pcm);
    mn_init();

    int loop_cnt=0;
    int infer_cnt = 0;
    int mn_have = 0;
    while (1) {
        if (g_state == STATE_IDLE) {
            // Rolling window: append AFE chunks; a full MEL_AUDIO_LEN window is
            // fed to the recognizer, then the window slides by KWS_HOP.
            // (AFE fetch delivers small chunks — a single fetch is never MEL_AUDIO_LEN.)
            mn_have = 0;
            if (snapshot_latest_audio(pcm)) {
                float rms = 0; int64_t sq = 0;
                for (int i=0; i<MEL_AUDIO_LEN; i++) sq += (int64_t)pcm[i]*pcm[i];
                rms = sqrtf((float)(sq/MEL_AUDIO_LEN));
                if (rms >= 5.0f) {
                    recognizer_run_frame(pcm, rms, esp_timer_get_time()/1000);
                    if (++infer_cnt % 3 == 0)
                        ESP_LOGI(TAG, "kws prob=%.3f rms=%.0f", recognizer_get_last_prob(), rms);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(20));
        } else {
            // COMMAND: consume sequential PCM supplied by audio_fetch_task.
            led_tick();
            size_t got = xStreamBufferReceive(
                g_cmd_stream, g_mn_buf + mn_have,
                (g_mn_chunksize - mn_have) * sizeof(int16_t),
                pdMS_TO_TICKS(20));
            mn_have += got / sizeof(int16_t);
            if (mn_have < g_mn_chunksize) continue;
            float rms_mn = 0; int64_t sq = 0;
            for (int i=0; i<g_mn_chunksize; i++) sq += (int64_t)g_mn_buf[i]*g_mn_buf[i];
            rms_mn = sqrtf((float)(sq/g_mn_chunksize));
            recognizer_feed_rms(rms_mn, esp_timer_get_time()/1000);

            esp_mn_state_t st = g_multinet->detect(g_mn_data, g_mn_buf);
            mn_have = 0;
            if (st == ESP_MN_STATE_DETECTED) {
                esp_mn_results_t *r = g_multinet->get_results(g_mn_data);
                led_on_cmd(r->command_id[0]);
                // printf, not esp_rom_printf: the ROM console is lost to UART0 on the
                // USB_SERIAL_JTAG console (only usable if USB was attached at reset)
                printf("\n*** CMD #%d: %s (prob=%.3f) ***\n\n",
                    r->command_id[0], g_cmd_name(r->command_id[0]), (double)r->prob[0]);
                fflush(stdout);
                g_state = STATE_IDLE;
                ESP_LOGI(TAG, "CMD -> IDLE");
            } else if (st == ESP_MN_STATE_TIMEOUT) {
                led_set_mode(LED_OFF,0,0,0);
                g_state = STATE_IDLE;
                ESP_LOGI(TAG, "CMD TIMEOUT -> IDLE");
            }
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
    g_cmd_stream = xStreamBufferCreate(16 * 1024, sizeof(int16_t));
    assert(g_audio_hist && g_audio_mtx && g_cmd_stream);

    if (mount_spiffs()!=0) return;
    xTaskCreatePinnedToCore(feed_task,"feed",6*1024,NULL,7,NULL,0);
    // Keep the complete AFE pipeline on core 0 so TFLite can run on core 1
    // without being preempted by the higher-priority fetch task.
    xTaskCreatePinnedToCore(audio_fetch_task,"afe_fetch",6*1024,NULL,6,NULL,0);
    xTaskCreatePinnedToCore(detect_loop,"kws_detect",12*1024,NULL,5,NULL,1);
    while(1) vTaskDelay(pdMS_TO_TICKS(1000));
}

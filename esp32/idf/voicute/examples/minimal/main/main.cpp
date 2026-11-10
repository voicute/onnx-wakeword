/*
 * minimal_wakeword — smallest possible voicute ESP-IDF component example.
 *
 * NOTE: this file is C++ (voicute.h pulls in TFLite Micro C++ headers).
 *
 * Compiles and runs with NO model, NO microphone, NO board BSP:
 *   - mounts SPIFFS (graceful if absent),
 *   - starts the recognizer,
 *   - feeds silence windows at 160 ms cadence — exactly the rhythm a real
 *     mic pipeline would use (see MEL_AUDIO_LEN in mel_extractor.h).
 *
 * To actually wake on device you still need (all caller-supplied, by design):
 *   1. a trained .tflite model on SPIFFS (filename stem = wake word),
 *   2. a real model head replacing stub_postprocess (head_stub.h),
 *   3. PCM from your mic driver fed instead of the silence buffer.
 *
 * Tested on ESP32-S3 (IDF v6.x). Other targets: code path should work but
 * is not hardware-verified.
 */
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_spiffs.h"   // IDF v6: was esp_vfs_spiffs.h
#include "voicute.h"
#include "head_stub.h"

static const char *TAG = "minimal";

/* One inference window: mel_time(98) frames -> (98-1)*160 + 512 + 160 = 16080
 * samples of 16 kHz int16 PCM (mel_extractor.h exports MEL_AUDIO_LEN). */
static int16_t g_pcm[MEL_AUDIO_LEN];   // silence

static void on_voice(voice_event_t ev, voice_evt_data_t data, void *user)
{
    (void)data; (void)user;
    switch (ev) {
    case VOICE_EVT_AWAKEN:
        ESP_LOGI(TAG, ">>> WAKE <<<");
        break;
    case VOICE_EVT_CMD:
        ESP_LOGI(TAG, "cmd id=%d", data.cmd_id);
        break;
    case VOICE_EVT_CMD_TIMEOUT:
        ESP_LOGI(TAG, "cmd timeout");
        break;
    }
}

static void mount_spiffs(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "storage",
        .max_files = 5,
        .format_if_mount_failed = false,
    };
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "SPIFFS mount failed (%s) — add a 'storage' partition "
                      "and flash your .tflite model to it", esp_err_to_name(ret));
    }
}

extern "C" void app_main(void)
{
    mount_spiffs();

    recognizer_config_t cfg = {
        .model_path  = "/spiffs",   // models live here as <word>.tflite
        .threshold   = 0.40f,
        /* ESP32 defaults (inference is slow; leaner gating than mobile):
         * L1 off, L2 peak/bg, L3 cooldown, L4 off, L5 energy jump. */
        .l1_enabled  = 0,
        .l2_enabled  = 1,
        .l3_enabled  = 1,
        .l4_enabled  = 0,
        .l5_enabled  = 1,
        .l5_delta    = 200.0f,
        .postprocess = stub_postprocess,   // REPLACE with your real head
    };
    recognizer_start(&cfg);

    if (recognizer_num_models() == 0) {
        ESP_LOGW(TAG, "No models on SPIFFS. Flash a trained .tflite to the "
                      "'storage' partition and replace head_stub.h with your "
                      "real model head. Idling.");
    } else {
        ESP_LOGI(TAG, "%d model(s):", recognizer_num_models());
        for (int i = 0; i < recognizer_num_models(); i++)
            ESP_LOGI(TAG, "  [%d] %s", i, recognizer_model_word(i));
        recognizer_register_callback(0, on_voice, NULL);
    }

    /* Inference cadence: a fresh window every 160 ms. A real application
     * copies the latest MEL_AUDIO_LEN samples from its mic ring buffer. */
    while (1) {
        float rms = 0.0f;   // RMS of the window (drives L5); real pipeline computes it
        recognizer_run_frame(g_pcm, rms, esp_timer_get_time() / 1000);
        vTaskDelay(pdMS_TO_TICKS(160));
    }
}

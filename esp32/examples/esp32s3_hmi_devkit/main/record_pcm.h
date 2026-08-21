// record_pcm.h — Standalone PCM recorder module
// Usage: #define TEST_MODE 2 in main.cpp, then run python record.py
// Records 5 seconds of 16kHz mono audio from I2S CH1, dumps as text over UART
//
// Pipe: I2S → ES7210 TDM → esp_codec_dev → raw[ch*i+1] → text dump
// Channel: CH1 (index 1 in 4-channel TDM, the active mic on this board)

#pragma once
#include "esp_rom_sys.h"
#include "bsp_board.h"
#include "esp_log.h"

static void record_task(void *arg) {
    int ch_cnt = esp_get_feed_channel();
    int chunk = 512, total = 16000 * 5;  // 5 seconds
    int16_t *raw = (int16_t*)malloc(chunk * ch_cnt * 2);
    int16_t *pcm = (int16_t*)heap_caps_malloc(total * 2, MALLOC_CAP_SPIRAM);
    assert(raw && pcm);

    // 3-second countdown
    ESP_LOGI("REC", "=== 3... ==="); vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI("REC", "=== 2... ==="); vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI("REC", "=== 1... ==="); vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI("REC", "=== SPEAK NOW! 5s ===");

    int idx = 0;
    for (int i = 0; i < total; i += chunk) {
        esp_get_feed_data(true, raw, chunk * ch_cnt * 2);
        for (int j = 0; j < chunk && idx < total; j++)
            pcm[idx++] = raw[ch_cnt * j + 1];  // CH1
    }

    // Text dump: START\n value\n ... END\n
    esp_rom_printf("START\n");
    for (int i = 0; i < total; i++)
        esp_rom_printf("%d\n", (int)pcm[i]);
    esp_rom_printf("END\n");

    free(raw); free(pcm);
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}

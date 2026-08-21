/**
 * Mel Extractor — 纯 C mel 声谱图提取
 *
 * 管线 (逐值匹配 melspectrogram.onnx):
 *   PCM int16 → float32 (raw, NO /32768) → NO pre-emphasis
 *   → framing × MEL_WINDOW[512] (从 onnx 导出)
 *   → 512-pt complex FFT (ESP-DSP dsps_fft2r_fc32)
 *   → dsps_bit_rev2r_fc32 (bit-reversal, 专家修复)
 *   → power (= real²+imag²)
 *   → mel filterbank (MEL_W[257][32] 从 onnx 导出)
 *   → S = 10*log10(max(mel_power, 1e-10))
 *   → S = clip(S, global_max - 80, +inf)    ← top_db=80
 *   → mel = S/10 + 2
 */

#include "mel_extractor.h"
#include "mel_filterbank.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"

static const char *TAG = "MelExtractor";

#include "esp_dsp.h"
static float fft_window[MEL_NFFT * 2];  // 512 complex = 1024 floats
static int fft_initialized = 0;
static uint16_t mel_first_bin[MEL_N_MELS];
static uint16_t mel_last_bin[MEL_N_MELS];
static int mel_ranges_initialized = 0;

static void init_fft(void) {
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, MEL_NFFT);
    if (ret != ESP_OK) ESP_LOGE(TAG, "FFT init failed: %d", ret);
    fft_initialized = 1;
}

static void init_mel_ranges(void) {
    for (int m = 0; m < MEL_N_MELS; m++) {
        int first = 0;
        while (first < MEL_NFREQ && MEL_W[first][m] == 0.0f) first++;
        int last = MEL_NFREQ - 1;
        while (last >= first && MEL_W[last][m] == 0.0f) last--;
        mel_first_bin[m] = (uint16_t)first;
        mel_last_bin[m] = (uint16_t)last;
    }
    mel_ranges_initialized = 1;
}

void mel_extractor_init(void) {
    if (!fft_initialized) init_fft();
    if (!mel_ranges_initialized) init_mel_ranges();
    ESP_LOGI(TAG, "Mel ready (n_fft=%d, n_mels=%d, mel_time=%d)", MEL_NFFT, MEL_N_MELS, MEL_TIME);
}

int mel_extract(const int16_t *pcm, float mel_out[MEL_TIME][MEL_N_MELS]) {
    if (!pcm || !mel_out) return -1;
    if (!fft_initialized) init_fft();
    if (!mel_ranges_initialized) init_mel_ranges();

    // --- 2. PASS 1: compute S=10*log10(mel_power), find global_max ---
    float global_max = -1e30f;
    for (int t = 0; t < MEL_TIME; t++) {
        int frame_start = t * MEL_HOP_LEN;

        // Apply MEL_WINDOW + fill complex FFT buffer (imag=0)
        for (int i = 0; i < MEL_NFFT; i++) {
            float sample = (float)pcm[frame_start + i];
            fft_window[i * 2]     = sample * MEL_WINDOW[i];
            fft_window[i * 2 + 1] = 0.0f;
        }

        // 512-point complex FFT + bit-reversal (EXPERT FIX)
        dsps_fft2r_fc32(fft_window, MEL_NFFT);
        dsps_bit_rev2r_fc32(fft_window, MEL_NFFT);

        // Power spectrum: bin k → re=y[2k], im=y[2k+1], power=re²+im²
        float power[MEL_NFREQ];
        for (int k = 0; k < MEL_NFREQ; k++) {
            float re = fft_window[k * 2];
            float im = fft_window[k * 2 + 1];
            power[k] = re * re + im * im;
        }

        // Mel filterbank
        float mel_power[MEL_N_MELS];
        for (int m = 0; m < MEL_N_MELS; m++) {
            float sum = 0.0f;
            // The triangular filterbank is 97.2% zero. Each filter's non-zero
            // support is contiguous, so skipping the zero tails is exactly
            // equivalent while avoiding most multiply/add loop iterations.
            for (int k = mel_first_bin[m]; k <= mel_last_bin[m]; k++)
                sum += power[k] * MEL_W[k][m];
            mel_power[m] = sum;
        }

        // S = 10*log10(max(mel_power, 1e-10))
        for (int m = 0; m < MEL_N_MELS; m++) {
            float clipped = mel_power[m] < MEL_EPSILON ? MEL_EPSILON : mel_power[m];
            float s = 10.0f * log10f(clipped);
            mel_out[t][m] = s;
            if (s > global_max) global_max = s;
        }
    }

    // --- 3. PASS 2: clip + /10+2 ---
    float clip_thresh = global_max - 80.0f;
    for (int t = 0; t < MEL_TIME; t++)
        for (int m = 0; m < MEL_N_MELS; m++) {
            float s = mel_out[t][m];
            if (s < clip_thresh) s = clip_thresh;
            mel_out[t][m] = s / 10.0f + 2.0f;
        }

    return 0;
}

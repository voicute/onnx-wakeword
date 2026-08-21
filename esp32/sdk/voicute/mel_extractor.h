/**
 * Mel Extractor — 纯 C mel 声谱图提取
 *
 * 与训练时 ONNX melspectrogram 精确对齐的参数:
 *   sample_rate=16000, n_fft=512, hop_len=160, win_len=512,
 *   n_mels=32, power=2 (power spectrum, NOT magnitude!),
 *   NO pre-emphasis, ONNX window (MEL_WINDOW from mel_filterbank.h)
 *   Input: raw int16 as float32 (NO /32768 normalization!)
 *
 * 输出后处理:
 *   S = 10*log10(max(mel_power, 1e-10))
 *   S = clip(S, global_max - 80, +inf)
 *   mel = S/10 + 2.0    (与训练一致)
 */

#pragma once

#include <stdint.h>
#include "mel_filterbank.h"  // MEL_NFFT, MEL_HOP, MEL_NFREQ, MEL_NBINS, MEL_WINDOW, MEL_W

#ifdef __cplusplus
extern "C" {
#endif

#define MEL_SAMPLE_RATE  16000
#define MEL_WIN_LEN      MEL_NFFT           // 512 — must match training
#define MEL_HOP_LEN      MEL_HOP            // 160
#define MEL_N_MELS       MEL_NBINS           // 32
#define MEL_NFFT_BINS    MEL_NFREQ          // 257 = n_fft/2 + 1
#define MEL_TIME         98                  // output time frames
#define MEL_AUDIO_LEN    ((MEL_TIME - 1) * MEL_HOP_LEN + MEL_WIN_LEN + MEL_HOP_LEN)
                                            // = 97*160 + 512 + 160 = 16192 samples (~1.0s)
#define MEL_PREEMPH      0.0f               // NO pre-emphasis (training ONNX model has none)
#define MEL_EPSILON      1e-10f             // clip minimum

/**
 * @brief 从 int16 PCM 提取 mel 声谱图
 *
 * @param  pcm     输入 PCM (int16, 16kHz, 单通道), 至少 MEL_AUDIO_LEN 采样点
 * @param  mel_out 输出 mel [MEL_TIME][MEL_N_MELS], 调用者分配 (float)
 * @return 0 成功, -1 失败
 */
int mel_extract(const int16_t *pcm, float mel_out[MEL_TIME][MEL_N_MELS]);

/**
 * @brief 初始化 mel 提取器 (预计算 Hamming 窗 + mel 滤波器组)
 *
 * 在 app_main 中调用一次。
 * mel_filterbank.h 中的 MEL_WINDOW + MEL_W 已就绪。
 * 如果不调用, 首次 mel_extract 会自动初始化。
 */
void mel_extractor_init(void);

#ifdef __cplusplus
}
#endif

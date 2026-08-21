/*
 * AFE Processor — reusable ESP-SR Audio Front-End wrapper
 * Provides beamforming (SE/BSS) + AGC for multi-mic audio
 * Usage:
 *   1. afe_processor_init()   — once at startup
 *   2. afe_processor_feed()   — call continuously with raw I2S data
 *   3. afe_processor_fetch()  — get processed (beamformed+AGC+NS) audio
 *   4. afe_processor_deinit() — cleanup
 *
 * To enable Noise Suppression (no model files needed!):
 *   afe_processor_config_t cfg = { .ns_init = true };
 *   afe_processor_init(&cfg);  // uses WebRTC NS internally
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct afe_processor afe_processor_t;

/**
 * @brief Configuration for AFE processor
 */
typedef struct {
    const char *input_format;   // "RMNM", "NMNM", etc. (default: "RMNM")
    int        afe_type;        // AFE_TYPE_SR=0, AFE_TYPE_VC=1 (default: 0)
    int        afe_mode;        // AFE_MODE_LOW_COST=0, AFE_MODE_HIGH_PERF=1 (default: 0)
    bool       wakenet_init;    // enable wake word (default: false)
    bool       vad_init;        // enable VAD (default: false)
    bool       ns_init;         // enable noise suppression (default: false)
} afe_processor_config_t;

/**
 * @brief Initialize AFE processor with default or custom config
 * @param cfg  NULL for defaults (RMNM, SR, LOW_COST, no wake/VAD/NS)
 * @return handle, or NULL on failure
 */
afe_processor_t *afe_processor_init(const afe_processor_config_t *cfg);

/**
 * @brief Feed raw multi-channel I2S data to AFE
 * @param ap    handle from afe_processor_init
 * @param data  raw interleaved int16_t audio (4 channels * frame_size samples)
 * @param len   total bytes of data
 */
void afe_processor_feed(afe_processor_t *ap, const int16_t *data, int len);

/**
 * @brief Fetch processed (beamformed + AGC) audio from AFE
 * @param ap       handle
 * @param out      output buffer (must be >= max_samples int16_t)
 * @param max_samples  max samples to copy
 * @return number of samples copied (0 if no data ready)
 */
int afe_processor_fetch(afe_processor_t *ap, int16_t *out, int max_samples);

/**
 * @brief Get the recommended feed chunk size (samples per channel)
 */
int afe_processor_feed_chunksize(afe_processor_t *ap);

/**
 * @brief Get the number of feed channels
 */
int afe_processor_feed_channels(afe_processor_t *ap);

/**
 * @brief Cleanup and free AFE processor
 */
void afe_processor_deinit(afe_processor_t *ap);

#ifdef __cplusplus
}
#endif

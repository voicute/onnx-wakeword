#pragma once
#include "model_loader.h"
#include "mel_extractor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { VOICE_EVT_AWAKEN, VOICE_EVT_CMD, VOICE_EVT_CMD_TIMEOUT } voice_event_t;

typedef union { int awaken_channel; int cmd_id; } voice_evt_data_t;

typedef void (*voice_event_callback_t)(voice_event_t, voice_evt_data_t, void*);

// Postprocess: backbone INT8 output [256] → float prob [0,1]
// Called after TFLite Invoke, before L1-L5 detection pipeline
typedef float (*kws_postprocess_fn)(const int8_t *backbone_out,
                                    float out_scale, int out_zero);

typedef struct {
    char  model_path[64];
    float threshold;
    int   l1_enabled, l2_enabled, l3_enabled, l4_enabled, l5_enabled;
    float l5_delta;
    kws_postprocess_fn postprocess;  // model-specific head (required)
    const uint8_t *compiled_model;   // optional: embedded model data
    size_t          compiled_len;    // optional: embedded model length
} recognizer_config_t;

// Init model + detection state (no ring buffer needed)
void recognizer_start(const recognizer_config_t *cfg);
void recognizer_stop(void);
void recognizer_register_callback(int idx, voice_event_callback_t cb, void *user);

// Run one inference frame on pre-captured PCM window
// now_ms: current time in ms (for L1-L5 detection pipeline)
void recognizer_run_frame(const int16_t *pcm, float rms, int64_t now_ms);

// Feed RMS to detection pipeline without running inference (for COMMAND mode)
void recognizer_feed_rms(float rms, int64_t now_ms);

#ifdef __cplusplus
}
#endif

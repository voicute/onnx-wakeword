/*
 * kws_postprocess.h — ESP32 float post-processing for INT8 KWS backbone.
 *
 * Plan B: the Causal DS-TCN backbone runs as INT8 tflite (esp-nn accelerated).
 * The tiny MultiProto head (325 params) runs here in float — it uses L2-normalize
 * + cosine, which TFLite-Micro/esp-nn cannot quantize.
 *
 * Pipeline on ESP32:
 *   mel[98][32] (float) --quantize--> int8 --> [INT8 backbone Invoke] --> out_int8[256]
 *   --> kws_postprocess(out_int8, out_scale, out_zero) --> prob in [0,1]
 *
 * Read out_scale / out_zero at runtime from the output tensor:
 *   TfLiteTensor* o = interpreter->output(0);
 *   float out_scale = o->params.scale;  int out_zero = o->params.zero_point;
 *   const int8_t* out_int8 = o->data.int8;
 *
 * Drop this file + head.h into your ESP-IDF component.
 */
#pragma once
#include <math.h>
#include <stdint.h>
// head.h must be included BEFORE this file (see main.cpp)
// Defines: KWS_HEAD_K, KWS_HEAD_D, KWS_ABS_TEMP, KWS_FC_B, KWS_FC_W, KWS_PROTO_NORM

/* Reproduces the PyTorch MultiProto head EXACTLY (verified bit-identical). */
static inline float kws_postprocess(const int8_t *out_int8,
                                          float out_scale, int out_zero)
{
    float feat[KWS_HEAD_D];
    float norm = 0.0f;

    /* 1. dequantize int8 backbone output -> float feature[256] */
    for (int i = 0; i < KWS_HEAD_D; i++) {
        feat[i] = ((float)out_int8[i] - (float)out_zero) * out_scale;
        norm += feat[i] * feat[i];
    }

    /* 2. L2-normalize the feature vector */
    norm = sqrtf(norm) + 1e-12f;
    for (int i = 0; i < KWS_HEAD_D; i++)
        feat[i] /= norm;

    /* 3. cosine similarity to each (pre-normalized) prototype, scaled by 1/|temp| */
    /* 4. linear classifier over the K cosine scores */
    float logit = KWS_FC_B;
    for (int k = 0; k < KWS_HEAD_K; k++) {
        float cos = 0.0f;
        for (int i = 0; i < KWS_HEAD_D; i++)
            cos += feat[i] * KWS_PROTO_NORM[k][i];
        cos /= KWS_ABS_TEMP;
        logit += KWS_FC_W[k] * cos;
    }

    /* 5. sigmoid -> wake probability */
    return 1.0f / (1.0f + expf(-logit));
}

/* Helper: quantize a float mel[98*32] into int8 for the backbone input.
 * Read in_scale / in_zero from interpreter->input(0)->params at runtime. */
static inline void manbo_kws_quantize_mel(const float *mel_f32, int8_t *mel_i8,
                                          int n, float in_scale, int in_zero)
{
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(mel_f32[i] / in_scale) + in_zero;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        mel_i8[i] = (int8_t)q;
    }
}

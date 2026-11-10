#pragma once
#include <stdint.h>

/*
 * head_stub.h — placeholder model head.
 *
 * The voicute ESP32 pipeline runs the INT8 backbone in TFLite-Micro; the
 * tiny classification head runs on top of the backbone's int8 output and
 * must be supplied by YOU (it contains your model's trained weights, so it
 * cannot ship inside the component).
 *
 * Replace `stub_postprocess` with the real head generated together with
 * your packaged model:
 *   - copy kws_postprocess.h + your head.h (from the voicute.com export or
 *     your keywords export dir, tflite subfolder) into main/
 *   - dispatch per model_idx as shown in
 *     esp32/idf/examples/esp32s3_hmi_devkit/main/main.cpp
 *
 * Signature matches kws_postprocess_fn in recognizer.h.
 */
static inline float stub_postprocess(const int8_t *out_i8, float out_scale,
                                     int out_zero, int model_idx)
{
    (void)out_i8;
    (void)out_scale;
    (void)out_zero;
    (void)model_idx;
    return 0.0f;   // never wakes — replace with the real head
}

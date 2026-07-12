/**
 * Ring Buffer — SPSC lock-free, PSRAM-backed
 *
 * feed_task (Core 0) writes PCM → detect_task (Core 1) reads.
 * Buffer allocated from PSRAM via heap_caps_malloc in rb_init.
 * Always writes; oldest data discarded when full.
 */
#pragma once
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "esp_heap_caps.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int16_t *buf;
    int      capacity;
    volatile int write_idx;
    volatile int read_idx;
} ring_buffer_t;

// Init with PSRAM allocation
static inline int rb_init(ring_buffer_t *rb, int capacity) {
    rb->buf = (int16_t*)heap_caps_malloc(capacity * sizeof(int16_t),
                                          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rb->buf) return -1;
    rb->capacity = capacity;
    rb->write_idx = 0;
    rb->read_idx  = 0;
    memset(rb->buf, 0, capacity * sizeof(int16_t));
    return 0;
}

static inline int rb_avail(ring_buffer_t *rb) {
    return (rb->write_idx - rb->read_idx + rb->capacity) % rb->capacity;
}

// Core 0: always write
static inline int rb_write(ring_buffer_t *rb, const int16_t *data, int len) {
    int w = rb->write_idx, r = rb->read_idx, cap = rb->capacity;
    int space = (r - w - 1 + cap) % cap;
    if (len > space) {
        int drop = len - space;
        rb->read_idx = (r + drop) % cap;
    }
    int16_t *buf = rb->buf;
    for (int i = 0; i < len; i++) {
        buf[w] = data[i];
        w = (w + 1) % cap;
    }
    rb->write_idx = w;
    return len;
}

// Core 1: read
static inline int rb_read(ring_buffer_t *rb, int16_t *dst, int len) {
    if (rb_avail(rb) < len) return 0;
    int r = rb->read_idx, cap = rb->capacity;
    int16_t *buf = rb->buf;
    for (int i = 0; i < len; i++) {
        dst[i] = buf[r];
        r = (r + 1) % cap;
    }
    rb->read_idx = r;
    return len;
}

// Core 1: peek recent
static inline int rb_peek_recent(ring_buffer_t *rb, int16_t *dst, int len) {
    if (rb_avail(rb) < len) return 0;
    int w = rb->write_idx, cap = rb->capacity;
    int start = (w - len + cap) % cap;
    int16_t *buf = rb->buf;
    for (int i = 0; i < len; i++)
        dst[i] = buf[(start + i) % cap];
    return len;
}

#ifdef __cplusplus
}
#endif

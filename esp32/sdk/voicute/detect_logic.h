/**
 * DetectLogic — L1-L5 唤醒词检测管线 (C 实现)
 *
 * 从 Android DetectionLogic.java 完整移植 (v9.0)。
 *
 * L1: N 连续帧超阈值 → 滤除瞬态噪声  (MAX_GAP=2, defCons=5)
 * L2: peak ≫ background (3×)        → 滤除模型波动  (sliding window max over 1500ms)
 * L3: 1.5s 冷却时间                 → 防止重复唤醒
 * L4: burst 3×/3s → block 5s        → 阻断循环播放  (FINAL GATE, 每次触发都经过)
 * L5: 能量跳变比                    → 阻断视频/音乐
 *     pre:  curRms / minRms[0.5-2.0s 前] > jumpRatio(5.0) → 孤立突发 (人声)
 *     post: minRms[tail 300ms] < preMin * 2.5              → 说完后安静了
 *     无绝对阈值。适配任何手机/音箱/房间。
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  L1-L4 常量
 * ============================================================ */

#define MAX_GAP     2          // L1: 允许 2 帧空档后重置
#define CD_MS       1500       // L3: 1.5s 冷却
#define BURST_WIN   3000       // L4: 3s 窗口
#define BURST_N     3          // L4: 3 次触发
#define BURST_BLOCK 5000       // L4: block 5s
#define BURST_HIST  8          // L4: 环形缓冲区大小

/* ============================================================
 *  L2 常量
 * ============================================================ */

#define PROB_HIST    128       // L2: prob history size
#define PEAK_WIN     1500      // L2: sliding window for peak (ms)
#define PEAK_RATIO   3.0f      // L2: peak/bg ratio threshold

/* ============================================================
 *  L5 常量
 * ============================================================ */

#define RMS_HIST      128
#define PRE_WIN_START 500      // ms before now
#define PRE_WIN_END   2000     // ms before now
#define POST_DELAY    400      // wait for keyword tail to decay (ESP32 tuned)
#define POST_TAIL     1000      // check last 1000ms after POST_DELAY (ESP32 slow rate)
#define RETURN_RATIO  2.5f     // postMin/preMin < 2.5 → went quiet
#define L5_DELTA      1200.0f  // curRms > preMin + delta → energy burst (additive, matches Python)
#define L5_QUIET_RMS  50.0f    // below this RMS, room is too quiet for L5 ratio
#define L5_MIN_RMS     80.0f   // current RMS below this in quiet room → block

/* ============================================================
 *  状态结构体
 * ============================================================ */

typedef struct {
    // L1
    int   cons;
    char  cons_word[64];
    int   cons_gap;
    int   cons_frames;         // default 2 (ESP32 tuned: 5 too strict for ~360ms inference)

    // L2: prob history for sliding window peak
    float prob_hist[PROB_HIST];
    int64_t prob_t_hist[PROB_HIST];
    int   prob_hi;
    float bg;                  // EMA background probability

    // L3
    int64_t last_trig;         // last trigger time (ms)

    // L4
    int64_t blocked;           // burst block expiry (ms)
    int64_t burst_t[BURST_HIST];
    char   burst_w[BURST_HIST][64];
    float  burst_p[BURST_HIST];
    int    burst_idx;

    // L5 RMS history
    float   rms_hist[RMS_HIST];
    int64_t rms_t_hist[RMS_HIST];
    int     rms_idx;

    // L5 pending (two-phase: pre burst → post silence)
    char    pending_word[64];
    int64_t pending_time;
    float   pending_prob;
    float   pending_pre_min;
    char    trigger_buf[64];    // L5b confirmed word copy (avoids pointer aliasing)

    // Layer toggles (1=enabled, 0=disabled)
    int     l1_enabled;
    int     l2_enabled;
    int     l3_enabled;
    int     l4_enabled;
    int     l5_enabled;

    // L5 config
    float   jump_ratio;          // deprecated: was multiplicative, now unused
    float   l5_delta;            // curRms > preMin + delta → energy burst (default 1200)

    // output
    int    count;              // trigger count
    float  last_trig_prob;     // last trigger probability

    // debug
    int    dbg_fail;           // which layer failed (0=ok, 1=L1, 2=L2, 3=L3, 4=L4, 5=L5)
    float  dbg_peak;
    float  dbg_bg;
    float  dbg_pre_rms;
    int    cons_frames_dbg;    // current cons count for debug
    int    base_cons;          // base consecutive frames count
} dl_state_t;

/* ============================================================
 *  API
 * ============================================================ */

void dl_init(dl_state_t *s);

void dl_record(dl_state_t *s, float prob, const char *word,
               float rms, int64_t now_ms);

const char *dl_evaluate(dl_state_t *s, const char *word, float prob,
                         float rms, float threshold, int64_t now_ms);

void dl_set_cons_frames(dl_state_t *s, int n);
void dl_set_l1_enabled(dl_state_t *s, int en);
void dl_set_l2_enabled(dl_state_t *s, int en);
void dl_set_l3_enabled(dl_state_t *s, int en);
void dl_set_l4_enabled(dl_state_t *s, int en);
void dl_set_l5_enabled(dl_state_t *s, int en);
void dl_set_jump_ratio(dl_state_t *s, float ratio);
float dl_get_jump_ratio(dl_state_t *s);

#ifdef __cplusplus
}
#endif

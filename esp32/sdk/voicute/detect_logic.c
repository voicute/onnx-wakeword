/**
 * DetectLogic — L1-L5 唤醒词检测管线 (实现)
 *
 * 从 Android DetectionLogic.java 完整移植 (v9.0).
 * L2 使用滑动窗口最大值 (不是 EMA), 与 Java 完全一致.
 */

#include "detect_logic.h"
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char *TAG = "DetectLogic";

/* ============================================================
 *  dl_init
 * ============================================================ */

void dl_init(dl_state_t *s) {
    memset(s, 0, sizeof(*s));
    s->bg          = 0.001f;
    s->cons_frames = 2;         // ESP32 tuned: 2 frames @ ~310ms ≈ 620ms window
    s->base_cons   = 5;
    s->l5_delta    = L5_DELTA;
    // All layers enabled by default
    s->l1_enabled  = 1;
    s->l2_enabled  = 1;
    s->l3_enabled  = 1;
    s->l4_enabled  = 1;
    s->l5_enabled  = 1;
}

void dl_set_cons_frames(dl_state_t *s, int n)  { s->cons_frames = n; s->base_cons = n; }
void dl_set_l1_enabled(dl_state_t *s, int en)  { s->l1_enabled = en; }
void dl_set_l2_enabled(dl_state_t *s, int en)  { s->l2_enabled = en; }
void dl_set_l3_enabled(dl_state_t *s, int en)  { s->l3_enabled = en; }
void dl_set_l4_enabled(dl_state_t *s, int en)  { s->l4_enabled = en; }
void dl_set_l5_enabled(dl_state_t *s, int en)  { s->l5_enabled = en; }
void dl_set_jump_ratio(dl_state_t *s, float r) { s->jump_ratio = r < 1.0f ? 1.0f : (r > 5.0f ? 5.0f : r); }
float dl_get_jump_ratio(dl_state_t *s)          { return s->jump_ratio; }

/* ============================================================
 *  dl_record — matches Java record() exactly
 * ============================================================ */

void dl_record(dl_state_t *s, float prob, const char *word,
               float rms, int64_t now_ms) {
    // L2 prob history (sliding window for peak)
    s->prob_hist[s->prob_hi]   = prob;
    s->prob_t_hist[s->prob_hi] = now_ms;
    s->prob_hi = (s->prob_hi + 1) % PROB_HIST;

    // L2 background EMA (only for background frames)
    if (!word || word[0] == '\0') {
        s->bg = s->bg * 0.995f + prob * 0.005f;
    }

    // L5 RMS history
    s->rms_hist[s->rms_idx]   = rms;
    s->rms_t_hist[s->rms_idx] = now_ms;
    s->rms_idx = (s->rms_idx + 1) % RMS_HIST;
}

/* ============================================================
 *  dl_evaluate — matches Java evaluate() exactly
 * ============================================================ */

const char *dl_evaluate(dl_state_t *s, const char *word, float prob,
                         float rms, float threshold, int64_t now_ms) {
    int need = s->cons_frames;
    const char *trigger_word = NULL;
    float trigger_prob = 0;
    s->dbg_fail = 0;

    // ── L5b: post-speech silence check (runs FIRST) ──
    if (s->l5_enabled && s->pending_word[0] != '\0') {
        int64_t elapsed = now_ms - s->pending_time;
        ESP_LOGI(TAG, "L5b: pending='%s' elapsed=%lld/%d ms",
                 s->pending_word, (long long)elapsed, POST_DELAY);
        if (elapsed >= POST_DELAY) {
            int64_t tail_start = now_ms - POST_TAIL;
            float post_min = 1e38f; int post_n = 0;
            for (int i = 0; i < RMS_HIST; i++) {
                if (s->rms_t_hist[i] >= tail_start && s->rms_t_hist[i] <= now_ms) {
                    float v = s->rms_hist[i];
                    if (v < post_min) post_min = v;
                    post_n++;
                }
            }
            ESP_LOGI(TAG, "L5b tail: post_n=%d postMin=%.0f preMin=%.0f ratio=%.1f (need<%d,<%.0f)",
                     post_n, (double)post_min, (double)s->pending_pre_min,
                     (double)(post_min / (s->pending_pre_min > 0 ? s->pending_pre_min : 1)),
                     3, (double)(s->pending_pre_min * RETURN_RATIO));
            if (post_n >= 3 && post_min < s->pending_pre_min * RETURN_RATIO
                && s->pending_word[0] != '\0') {
                strncpy(s->trigger_buf, s->pending_word, sizeof(s->trigger_buf) - 1);
                s->trigger_buf[sizeof(s->trigger_buf) - 1] = '\0';
                trigger_word = s->trigger_buf;
                trigger_prob = s->pending_prob;
                ESP_LOGI(TAG, "L5 OK: -> '%s'", s->trigger_buf);
            } else {
                ESP_LOGI(TAG, "L5 fail: post_n=%d ratio=%.1f",
                         post_n, (double)(post_min / (s->pending_pre_min > 0 ? s->pending_pre_min : 1)));
            }
            s->pending_word[0] = '\0';
            s->pending_time = 0;
        }
    }

    // ── L1-L5a: normal pipeline ──
    if (trigger_word == NULL) {
        // Threshold gate — always active (matches Python L62-63)
        int hi = (prob > threshold && word && word[0] != '\0');
        if (!hi) {
            s->cons = 0; s->cons_word[0] = '\0'; s->cons_gap = 0;
            return NULL;
        }
        ESP_LOGI(TAG, "THR pass: prob=%.3f > thr=%.2f word='%s'",
                 (double)prob, (double)threshold, word);

        // L1: consecutive frames (off → bypass, need=1)
        if (s->l1_enabled) {
            int hi = (prob > threshold && word && word[0] != '\0');
            if (hi && strcmp(word, s->cons_word) == 0) {
                s->cons++;
                s->cons_gap = 0;
            } else if (hi) {
                strncpy(s->cons_word, word, sizeof(s->cons_word) - 1);
                s->cons = 1;
                s->cons_gap = 0;
            } else if (s->cons > 0) {
                s->cons_gap++;
                if (s->cons_gap > MAX_GAP) {
                    s->cons = 0;
                    s->cons_word[0] = '\0';
                    s->cons_gap = 0;
                }
            }
            s->cons_frames_dbg = s->cons;
            if (s->cons < need) { ESP_LOGI(TAG, "L1 fail: cons=%d/%d", s->cons, need); s->dbg_fail = 1; goto done; }
        } else {
            s->cons_frames_dbg = 1;
        }

        // L2: peak / background (off → bypass)
        if (s->l2_enabled) {
            float peak = 0;
            for (int i = 0; i < PROB_HIST; i++) {
                if (s->prob_t_hist[i] > 0 && (now_ms - s->prob_t_hist[i]) < PEAK_WIN) {
                    float p = s->prob_hist[i];
                    if (p > peak) peak = p;
                }
            }
            s->dbg_peak = peak;
            s->dbg_bg   = s->bg;
            if (peak <= s->bg * PEAK_RATIO) {
                ESP_LOGI(TAG, "L2 fail: peak=%.3f bg=%.3f ratio=%.1f", (double)peak, (double)s->bg, (double)(peak/(s->bg+1e-9f)));
                s->dbg_fail = 2; goto done;
            }
        } else {
            s->dbg_peak = prob;
            s->dbg_bg   = s->bg;
        }

        // L3: cooldown (off → bypass)
        if (s->l3_enabled) {
            if ((now_ms - s->last_trig) < CD_MS) {
                ESP_LOGI(TAG, "L3 fail: cooldown %lld/%d ms", (long long)(now_ms - s->last_trig), CD_MS);
                s->dbg_fail = 3; goto done;
            }
        }

        // L4a: burst cooldown (off → bypass)
        if (s->l4_enabled) {
            if (now_ms < s->blocked) {
                ESP_LOGI(TAG, "L4 fail: blocked for %lld ms", (long long)(s->blocked - now_ms));
                s->cons = 0; s->cons_word[0] = '\0'; s->dbg_fail = 4; goto done;
            }
        }

        // L5: energy jump + post-speech silence (ESP32: quick RMS provides dense tail data)
        if (s->l5_enabled) {
            int64_t pre_start = now_ms - PRE_WIN_END;
            int64_t pre_end   = now_ms - PRE_WIN_START;
            float pre_min = 1e38f; int pre_n = 0;
            for (int i = 0; i < RMS_HIST; i++) {
                if (s->rms_t_hist[i] > 0 && s->rms_t_hist[i] >= pre_start
                    && s->rms_t_hist[i] <= pre_end) {
                    float v = s->rms_hist[i];
                    if (v < pre_min) pre_min = v;
                    pre_n++;
                }
            }
            s->dbg_pre_rms = (pre_n > 0) ? pre_min : -1.0f;

            if (pre_n >= 5) {
                if (pre_min < L5_QUIET_RMS && rms < L5_MIN_RMS) {
                    ESP_LOGI(TAG, "L5 quiet: curRms=%.0f preMin=%.0f", (double)rms, (double)pre_min);
                    s->dbg_fail = 5; goto done;
                }
                if (rms < pre_min + s->l5_delta) {
                    ESP_LOGI(TAG, "L5 steady: curRms=%.0f preMin=%.0f + delta=%.0f",
                             (double)rms, (double)pre_min, (double)s->l5_delta);
                    s->dbg_fail = 5; goto done;
                }
                // Energy jump → defer to L5b for post-speech confirmation
                ESP_LOGI(TAG, "L5 JUMP: curRms=%.0f > preMin=%.0f + %.0f → pending",
                         (double)rms, (double)pre_min, (double)s->l5_delta);
                strncpy(s->pending_word, word, sizeof(s->pending_word) - 1);
                s->pending_time    = now_ms;
                s->pending_prob    = prob;
                s->pending_pre_min = pre_min;
                s->cons = 0;
                s->cons_word[0] = '\0';
                s->dbg_fail = 0;
                goto done;
            }
        }

        // Normal trigger (no L5 pending, or L5 skipped)
        trigger_word = word;
        trigger_prob = prob;          // matches Python: trigger_prob = prob
    }

    // ═══════════════════════════════════════════════════════════
    // L4 burst gate (matches Python: gated by l4)
    // ═══════════════════════════════════════════════════════════
    if (s->l4_enabled && trigger_word != NULL) {
        s->burst_t[s->burst_idx] = now_ms;
        size_t word_cap = sizeof(s->burst_w[0]);
        size_t word_len = strnlen(trigger_word, word_cap - 1);
        memcpy(s->burst_w[s->burst_idx], trigger_word, word_len);
        s->burst_w[s->burst_idx][word_len] = '\0';
        s->burst_p[s->burst_idx] = trigger_prob;
        s->burst_idx = (s->burst_idx + 1) % BURST_HIST;

        int bc = 0;
        for (int i = 0; i < BURST_HIST; i++) {
            if (s->burst_t[i] > 0
                && (now_ms - s->burst_t[i]) < BURST_WIN
                && strcmp(trigger_word, s->burst_w[i]) == 0
                && s->burst_p[i] > 0.8f) bc++;
        }
        if (bc >= BURST_N) {
            s->blocked = now_ms + BURST_BLOCK;
            s->cons = 0;
            s->cons_word[0] = '\0';
            ESP_LOGW(TAG, "BURST %dx '%s' -> block %ds", bc, trigger_word, BURST_BLOCK / 1000);
            s->dbg_fail = 4;
            return NULL;
        }
    }

    // ── Trigger finalization (always runs, regardless of L4) ──
    if (trigger_word != NULL) {
        s->last_trig_prob = trigger_prob;
        s->last_trig = now_ms;
        s->count++;
        s->cons = 0;
        s->cons_word[0] = '\0';
        s->dbg_fail = 0;

        // Safety: never return empty string
        if (!trigger_word || trigger_word[0] == '\0') {
            ESP_LOGW(TAG, "BUG: empty trigger_word, suppressing");
            return NULL;
        }

        ESP_LOGI(TAG, "TRIGGER '%s' prob=%.3f peak=%.3f bg=%.3f cons=%d",
                 trigger_word, (double)trigger_prob, (double)s->dbg_peak,
                 (double)s->dbg_bg, s->cons_frames_dbg);
        return trigger_word;
    }

done:
    { static int dl_cnt = 0; dl_cnt++;
      if (dl_cnt % 100 == 0 || s->dbg_fail == 5) {
        ESP_LOGI(TAG, "dbg#%d: prob=%.3f peak=%.3f bg=%.3f cons=%d/%d preRms=%.0f fail=%d",
                 dl_cnt, (double)prob, (double)s->dbg_peak, (double)s->dbg_bg,
                 s->cons_frames_dbg, s->cons_frames, (double)s->dbg_pre_rms, s->dbg_fail);
      }}
    return NULL;
}

"""Wake word detection pipeline (L1-L5) — Python port from Java DetectionLogic.java.

L1: N consecutive frames above threshold → filters transient noise
L2: peak ≫ background level (3×)        → filters model fluctuation
L3: 1.5s cooldown                        → prevents double-trigger
L4: burst 3×/3s → suppress 5s            → blocks playback loops
L5: energy jump ratio                    → blocks video/music

Configurable: cons_frames, threshold, jump_ratio, per-layer toggles
"""
import numpy as np


class DetectionLogic:
    def __init__(self, thr=0.5, cons_frames=5, jump_ratio=5.0):
        self.thr, self.cons_frames, self.jump_ratio = thr, cons_frames, jump_ratio
        self.HIST, self.RMS_HIST = 128, 128
        self.PEAK_WIN, self.CD_MS = 1500, 1500
        self.BURST_WIN, self.BURST_N, self.BURST_BLOCK, self.BH = 3000, 3, 5000, 8
        self.MAX_GAP = 2
        self.PRE_WIN_START, self.PRE_WIN_END = 500, 2000
        self.POST_DELAY, self.POST_TAIL, self.RETURN_RATIO = 700, 300, 2.5
        # Per-layer toggles
        self.l1, self.l2, self.l3, self.l4, self.l5 = True, True, True, True, True
        self.count = 0
        self._reset()

    def _reset(self):
        self.cons, self.cons_word, self.cons_gap = 0, '', 0
        self.p_hist = np.zeros(self.HIST); self.t_hist = np.zeros(self.HIST); self.hi = 0
        self.bg = 0.001
        self.rms_hist = np.zeros(self.RMS_HIST); self.rms_t = np.zeros(self.RMS_HIST); self.ri = 0
        self.last_trig, self.count, self.blocked = 0, 0, 0
        self.bT = np.zeros(self.BH); self.bW = [''] * self.BH; self.bP = np.zeros(self.BH); self.bi = 0
        self.pending_word = None; self.pending_time = 0; self.pending_pre_min = 0; self.pending_prob = 0

    def reset(self):
        self._reset()

    def record(self, prob, word, rms, now_ms):
        self.p_hist[self.hi] = prob; self.t_hist[self.hi] = now_ms; self.hi = (self.hi + 1) % self.HIST
        # Only track non-wake-word frames for background EMA (matches Android)
        if not word: self.bg = self.bg * 0.995 + prob * 0.005
        self.rms_hist[self.ri] = rms; self.rms_t[self.ri] = now_ms; self.ri = (self.ri + 1) % self.RMS_HIST

    def evaluate(self, word, prob, rms, now_ms):
        trigger_word = None; trigger_prob = 0
        # L5b: post-speech confirmation
        if self.l5 and self.pending_word is not None:
            if (now_ms - self.pending_time) >= self.POST_DELAY:
                tail_start = now_ms - self.POST_TAIL
                post_min, post_n = float('inf'), 0
                for i in range(self.RMS_HIST):
                    if tail_start <= self.rms_t[i] <= now_ms:
                        post_min = min(post_min, self.rms_hist[i]); post_n += 1
                if post_n >= 3 and post_min < self.pending_pre_min * self.RETURN_RATIO:
                    trigger_word, trigger_prob = self.pending_word, self.pending_prob
                self.pending_word = None
        # L1-L5
        if trigger_word is None:
            hi = prob > self.thr and bool(word)
            if self.l1:
                if hi and word == self.cons_word: self.cons += 1; self.cons_gap = 0
                elif hi: self.cons_word = word; self.cons = 1; self.cons_gap = 0
                elif self.cons > 0:
                    self.cons_gap += 1
                    if self.cons_gap > self.MAX_GAP: self.cons, self.cons_word, self.cons_gap = 0, '', 0
                if self.cons < self.cons_frames: return None
            # L2: peak/background ratio — matches Android DetectionLogic L2
            if self.l2:
                peak = max([self.p_hist[i] for i in range(self.HIST)
                            if self.t_hist[i] > 0 and (now_ms - self.t_hist[i]) < self.PEAK_WIN] + [0])
                if peak <= self.bg * 3: return None
            # L3: cooldown
            if self.l3 and (now_ms - self.last_trig) < self.CD_MS: return None
            # L4a: burst cooldown
            if self.l4 and now_ms < self.blocked: self.cons = 0; self.cons_word = ''; return None
            # L5a: energy jump
            if self.l5:
                pre_start, pre_end = now_ms - self.PRE_WIN_END, now_ms - self.PRE_WIN_START
                pre_min, pre_n = float('inf'), 0
                for i in range(self.RMS_HIST):
                    if pre_start <= self.rms_t[i] <= pre_end:
                        pre_min = min(pre_min, self.rms_hist[i]); pre_n += 1
                if pre_n >= 5 and pre_min < 50 and rms < 80: return None
                if pre_n >= 5 and rms < pre_min * self.jump_ratio: return None
                if pre_n >= 5:
                    self.pending_word = word; self.pending_time = now_ms
                    self.pending_prob = prob; self.pending_pre_min = pre_min
                    self.cons = 0; self.cons_word = ''; return None
            trigger_word, trigger_prob = word, prob
        # L4 burst gate
        if self.l4 and trigger_word is not None:
            self.bT[self.bi] = now_ms; self.bW[self.bi] = trigger_word; self.bP[self.bi] = trigger_prob
            self.bi = (self.bi + 1) % self.BH
            bc = sum(1 for i in range(self.BH)
                     if self.bT[i] > 0 and (now_ms - self.bT[i]) < self.BURST_WIN
                     and trigger_word == self.bW[i] and self.bP[i] > 0.8)
            if bc >= self.BURST_N:
                self.blocked = now_ms + self.BURST_BLOCK; self.cons = 0; self.cons_word = ''; return None
        if trigger_word:
            self.last_trig = now_ms; self.count += 1; self.cons = 0; self.cons_word = ''
            return trigger_word
        return None

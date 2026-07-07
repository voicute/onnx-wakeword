package com.voicute.wakeword;

import android.util.Log;
import java.util.Locale;

/**
 * Wake word detection pipeline (5 layers).
 *
 *  L1: N consecutive frames above threshold → filters transient noise
 *  L2: peak ≫ background level (3×)        → filters model fluctuation
 *  L3: 1.5s cooldown                        → prevents double-trigger
 *  L4: burst 3×/3s → suppress 5s            → blocks playback loops
 *  L5: energy jump ratio                    → blocks video/music
 *       pre:  curRms / minRms[0.5-2.0s ago] > 4.0 → isolated burst (human)
 *       post: minRms[tail 300ms] < preMin * 2.0  → went quiet after
 *       No absolute thresholds. Adapts to any phone/speaker/room.
 */
class DetectionLogic {

    private static final String TAG = "Voicute.Detect";

    // L1: consecutive frames
    private int cons;
    private String consWord = "";
    private int consGap;                        // gap frames since last hi (allow brief pauses)
    private static final int MAX_GAP = 2;       // max gap frames before resetting cons
    private int modelCons = 5;

    // L2: relative confidence
    private static final int HIST = 128;
    private final float[] pHist = new float[HIST];
    private final long[] tHist = new long[HIST];
    private int hi;
    private static final long PEAK_WIN = 1500;
    private float bg = 0.001f;

    // L3: cooldown
    private static final long CD_MS = 1500;
    private long lastTrig;

    // L4: burst
    private static final int BURST_WIN = 3000, BURST_N = 3;
    private static final long BURST_BLOCK = 5000;
    private static final int BH = 8;
    private final long[] bT = new long[BH];
    private final String[] bW = new String[BH];
    private final float[] bP = new float[BH];
    private int bi;
    private long blocked;

    // L5: energy jump ratio — simple, no absolute thresholds
    private static final int RMS_HIST = 128;
    private final float[] rmsHist = new float[RMS_HIST];
    private final long[] rmsTHist = new long[RMS_HIST];
    private int ri;
    private static final long PRE_WIN_START = 500;
    private static final long PRE_WIN_END = 2000;
    private static final long POST_DELAY = 700;       // wait for keyword tail to decay
    private static final long POST_TAIL = 300;        // check last 300ms of post window
    int l5Delta = 1200;          // L5 energy delta: curRms > preMin + l5Delta
    private static final float RETURN_RATIO = 2.5f;   // postMin/preMin < 2.5 → went quiet

    // L5 pending
    private String pendingWord;
    private long pendingTime;
    private float pendingProb;
    private float pendingPreMin;  // preMin at detection time (for post comparison)

    // L5 toggle
    boolean l5Enabled = false;  // off by default, user enables after testing RMS

    // output (package-private)
    int count;
    float lastTrigProb;
    float dbgPeak, dbgBg, dbgPreRms;
    int dbgFail;
    int consFrames;
    int baseCons;

    // ---------- public ----------

    void record(float prob, String word, float rms, long now) {
        pHist[hi] = prob;
        tHist[hi] = now;
        hi = (hi + 1) % HIST;
        if (word == null || word.isEmpty()) bg = bg * 0.995f + prob * 0.005f;
        rmsHist[ri] = rms;
        rmsTHist[ri] = now;
        ri = (ri + 1) % RMS_HIST;
    }

    String evaluate(String word, float prob, float rms, float thr, int baseCons, int extra, long now) {
        if (baseCons > 0) modelCons = baseCons;
        baseCons = modelCons;
        int need = Math.max(modelCons, extra);

        // Candidate trigger — set by either L5b confirmation or the normal L1-L5 pipeline.
        // L4 (burst detection) is the SINGLE final gate below, applied to both paths.
        String triggerWord = null;
        float triggerProb = 0;

        // ── L5b: post-speech check (runs first, before L1-L3) ──
        if (l5Enabled && pendingWord != null) {
            if ((now - pendingTime) >= POST_DELAY) {
                long tailStart = now - POST_TAIL;
                float postMin = Float.MAX_VALUE; int postN = 0;
                for (int i = 0; i < RMS_HIST; i++) {
                    if (rmsTHist[i] >= tailStart && rmsTHist[i] <= now) {
                        float v = rmsHist[i]; if (v < postMin) postMin = v;
                        postN++;
                    }
                }
                if (postN >= 3 && postMin < pendingPreMin * RETURN_RATIO) {
                    // Went quiet after keyword → confirm (final gate below)
                    triggerWord = pendingWord;
                    triggerProb = pendingProb;
                    Log.d(TAG, String.format(Locale.US,
                            "L5 OK: preMin=%.0f postMin=%.0f ratio=%.1f → trigger '%s'",
                            pendingPreMin, postMin, postMin / pendingPreMin, triggerWord));
                    pendingWord = null;
                } else if (postN >= 3) {
                    Log.d(TAG, String.format(Locale.US,
                            "L5 post block: preMin=%.0f postMin=%.0f ratio=%.1f → continuous",
                            pendingPreMin, postMin, postMin / pendingPreMin));
                    pendingWord = null;
                    dbgFail = 5;
                }
            }
        }

        // ── L1-L5: normal pipeline (skipped if L5 already confirmed above) ──
        if (triggerWord == null) {
            // L1: consecutive frames — allow brief gaps (empty words) without resetting
            boolean hi = prob > thr && word != null && !word.isEmpty();
            if (hi && word.equals(consWord)) { cons++; consGap = 0; }
            else if (hi) { consWord = word; cons = 1; consGap = 0; }
            else if (cons > 0) { consGap++; if (consGap > MAX_GAP) { cons = 0; consWord = ""; consGap = 0; } }
            if (cons < need) { consFrames = cons; dbgFail = 1; return null; }
            consFrames = cons;

            // L2: peak / background
            float peak = 0;
            for (int i = 0; i < HIST; i++) {
                if (tHist[i] > 0 && (now - tHist[i]) < PEAK_WIN) {
                    float p = pHist[i]; if (p > peak) peak = p;
                }
            }
            dbgPeak = peak; dbgBg = bg;
            if (peak <= bg * 3f) { dbgFail = 2; return null; }

            // L3: cooldown
            if ((now - lastTrig) < CD_MS) { dbgFail = 3; return null; }

            // L4a: burst cooldown
            if (now < blocked) { cons = 0; consWord = ""; dbgFail = 4; return null; }

            // L5a: energy jump ratio
            if (l5Enabled) {
                long preStart = now - PRE_WIN_END;
                long preEnd = now - PRE_WIN_START;
                float preMin = Float.MAX_VALUE; int preN = 0;
                for (int i = 0; i < RMS_HIST; i++) {
                    if (rmsTHist[i] > 0 && rmsTHist[i] >= preStart && rmsTHist[i] <= preEnd) {
                        float v = rmsHist[i]; if (v < preMin) preMin = v;
                        preN++;
                    }
                }
                dbgPreRms = preN > 0 ? preMin : -1f;
                // Quiet-room guard: when ambient is near-silent, ratio alone is unreliable
                if (preN >= 5 && preMin < 50f && rms < 80f) {
                    dbgFail = 5;
                    Log.d(TAG, String.format(Locale.US,
                            "L5 pre: curRms=%.0f preMin=%.0f → too quiet for voice",
                            rms, preMin));
                    return null;
                }
                if (preN >= 5 && rms < preMin + l5Delta) {
                    // No dramatic energy jump → steady noise (video/music)
                    dbgFail = 5;
                    Log.d(TAG, String.format(Locale.US,
                            "L5 pre: curRms=%.0f preMin=%.0f delta=%d need>%.0f → steady noise",
                            rms, preMin, l5Delta, preMin + l5Delta));
                    return null;
                }
                // preN < 5: insufficient RMS history → skip L5, fall through to trigger
                if (preN < 5) {
                    dbgPreRms = -1f;
                    // Fall through to normal trigger (L5 skipped)
                } else {
                    // Energy jump detected → set pending, wait for post confirmation
                    pendingWord = word; pendingTime = now; pendingProb = peak;
                    pendingPreMin = preMin;
                    cons = 0; consWord = "";
                    dbgFail = 0;
                    return null;
                }
            } else {
                dbgPreRms = -1f;
            }

            // L5 disabled → candidate trigger from normal pipeline
            triggerWord = word;
            triggerProb = peak;
        }

        // ═══════════════════════════════════════════════════════════
        // FINAL GATE: L4 burst detection — every trigger goes through here.
        // ═══════════════════════════════════════════════════════════
        if (triggerWord != null) {
            bT[bi] = now; bW[bi] = triggerWord; bP[bi] = triggerProb; bi = (bi + 1) % BH;
            int bc = 0;
            for (int i = 0; i < BH; i++) {
                if (bT[i] > 0 && (now - bT[i]) < BURST_WIN
                        && triggerWord.equals(bW[i]) && bP[i] > 0.8f) bc++;
            }
            if (bc >= BURST_N) {
                blocked = now + BURST_BLOCK; cons = 0; consWord = "";
                Log.w(TAG, String.format(Locale.US,
                        "BURST %dx '%s' → block %ds", bc, triggerWord, (int)(BURST_BLOCK/1000)));
                dbgFail = 4; return null;
            }

            lastTrigProb = triggerProb;
            dbgFail = 0; lastTrig = now; count++; cons = 0; consWord = "";
            return triggerWord;
        }

        return null;
    }

    void reset() {
        cons = 0; consWord = ""; consGap = 0; count = 0; lastTrig = 0; blocked = 0; bg = 0.001f;
        pendingWord = null; pendingTime = 0;
        for (int i = 0; i < HIST; i++) { pHist[i] = 0; tHist[i] = 0; }
        for (int i = 0; i < RMS_HIST; i++) { rmsHist[i] = 0; rmsTHist[i] = 0; }
    }
}

/**
 * Voicute Wake Word Engine — Web Edition v9.0
 *
 * Usage:
 *   const engine = VoicuteWakeWord.create();
 *   await engine.load('models/model_info.json', 'models/melspectrogram.onnx');
 *   await engine.start((word, prob, info) => console.log('detected:', word, prob));
 *
 * Detection pipeline (all toggleable, off by default except L1):
 *   L0: threshold > 0.4          — always active
 *   L1: N consecutive frames      — filters transient clicks/noise
 *   L2: peak/background ratio     — filters model hallucination on silence
 *   L3: cooldown 1.5s             — prevents duplicate triggers
 *   L4: burst detection 3/3s→5s   — blocks audio feedback loops
 *   L5: energy jump ratio         — blocks video/music playback
 */

(function () {

const SAMPLE_RATE = 16000;
const MEL_HOP = 160, MEL_WIN = 400, N_MELS = 32;

// ═══════════════════════════════════════════════
// Engine
// ═══════════════════════════════════════════════

window.VoicuteWakeWord = {
    create() {
        let melSession = null, models = [];
        let dscnnMode = false, dscnnMelTime = 98;
        let audioSamplesNeeded = 0;

        // ═══════════════════════════════════════════════
        // DetectionLogic (matches Java DetectionLogic.java exactly)
        // ═══════════════════════════════════════════════
        const HIST = 128, PEAK_WIN = 1500, CD_MS = 1500, MAX_GAP = 2;
        const BURST_WIN = 3000, BURST_N = 5, BURST_BLOCK = 3000, BH = 8;
        const RMS_HIST = 128, PRE_WIN_START = 500, PRE_WIN_END = 2000;
        const POST_DELAY = 700, POST_TAIL = 300, RETURN_RATIO = 2.5;

        // Internal state
        let cons = 0, consWord = '', consGap = 0;
        let bg = 0.001, lastTrig = 0, blocked = 0;
        const pHist = new Float32Array(HIST), tHist = new Array(HIST).fill(0); let pHi = 0;
        const rmsHist = new Float32Array(RMS_HIST), rmsTHist = new Array(RMS_HIST).fill(0); let ri = 0;
        // L5 pending
        let pendingWord = null, pendingTime = 0, pendingProb = 0, pendingPreMin = 0;
        // L4 burst
        const bT = new Float32Array(BH), bW = new Array(BH), bP = new Float32Array(BH); let bi = 0;
        // Layer toggles + config
        let L1 = true, L2 = false, L3 = false, L4 = false, L5 = false;
        let l5delta = 60;  // L5 energy delta: curRms > preMin + l5delta
        let _lastBlock = '';

        let _debugLog = false;
        const _log = (...a) => { if (_debugLog) console.log(...a); };
        const _warn = (...a) => { if (_debugLog) console.warn(...a); };
        const debug = { sampleRate: 0, lastScores: {}, inferCount: 0, consCount: 0, lastBlock: '', bg: 0, lastRms: 0 };

        // ---- Audio state ----
        let audioCtx = null, stream = null, listening = false;
        let ringBuf = null, ringPos = 0, lastRingPos = 0;
        let processorNode = null, busy = false;
        let cfgThreshold = 0.40, cfgCooldown = 1500;

        // ═══════════════════════════
        // Model loading
        // ═══════════════════════════

        async function _loadOrtModel(url) {
            const r = await fetch(url);
            if (!r.ok) throw new Error(`load ${url}: ${r.status}`);
            return await ort.InferenceSession.create(await r.arrayBuffer(), { executionProviders: ['wasm'] });
        }

        async function load(modelInfoUrl, melUrl) {
            melSession = await _loadOrtModel(melUrl);

            const resp = await fetch(modelInfoUrl);
            const buf = await resp.arrayBuffer();
            const header = new Uint8Array(buf, 0, 4);
            const isZip = header[0] === 0x50 && header[1] === 0x4b;

            let info, zip;
            if (isZip) {
                if (typeof JSZip === 'undefined') throw new Error('JSZip required for ZIP packages — add <script src="jszip.min.js">');
                zip = await JSZip.loadAsync(buf);
                const infoFile = zip.file('model_info.json');
                if (!infoFile) throw new Error('model_info.json not found in ZIP');
                info = JSON.parse(await infoFile.async('string'));
            } else {
                info = JSON.parse(new TextDecoder().decode(buf));
            }

            dscnnMode = info.model_type === 'dscnn' || info.model_type === 'tcn';
            dscnnMelTime = info.mel_time || 98;
            const cfg = (info.multi_model && info.models) ? info.models : [info];
            models = await Promise.all(cfg.map(async m => ({
                name: m.wake_word,
                session: await (async () => {
                    if (zip) {
                        const f = zip.file(m.model_file);
                        if (!f) throw new Error(`${m.model_file} not found in ZIP`);
                        return await ort.InferenceSession.create(await f.async('arraybuffer'), { executionProviders: ['wasm'] });
                    }
                    const base = modelInfoUrl.substring(0, modelInfoUrl.lastIndexOf('/') + 1);
                    return await _loadOrtModel(base + m.model_file);
                })(),
                consFrames: m.cons_frames || 3,
            })));
            audioSamplesNeeded = dscnnMode
                ? dscnnMelTime * MEL_HOP + MEL_WIN
                : (76 + (Math.max(...models.map(m => m.embFrames || 1)) - 1) * 8) * MEL_HOP + MEL_WIN;
            _log(`[wakeword] ${models.length} model(s), dscnn=${dscnnMode}`);
        }

        // ═══════════════════════════
        // Inference
        // ═══════════════════════════

        async function predict(audioData) {
            if (!melSession || models.length === 0) return null;
            const melIn = new ort.Tensor('float32', audioData, [1, audioData.length]);
            const melOut = await melSession.run({ input: melIn });
            const mel = melOut[Object.keys(melOut)[0]].data;
            const frames = Math.floor(mel.length / N_MELS);
            const mel2d = new Float32Array(frames * N_MELS);
            for (let i = 0; i < frames * N_MELS; i++) mel2d[i] = mel[i] / 10 + 2;
            debug.inferCount++;

            const scores = [], words = [], cfList = [];
            if (dscnnMode) {
                const start = Math.max(0, frames - dscnnMelTime);
                const input = new Float32Array(dscnnMelTime * N_MELS);
                for (let f = 0; f < dscnnMelTime; f++) {
                    const s = start + f;
                    if (s >= 0 && s < frames) input.set(mel2d.subarray(s * N_MELS, (s + 1) * N_MELS), f * N_MELS);
                }
                for (const m of models) {
                    const out = await m.session.run({ input: new ort.Tensor('float32', input, [1, dscnnMelTime, N_MELS]) });
                    scores.push(out[Object.keys(out)[0]].data[0]);
                    words.push(m.name); cfList.push(m.consFrames);
                }
            } else { return null; }

            // Sigmoid → softmax
            const K = models.length;
            const logits = new Array(K + 1); let maxL = -Infinity;
            for (let i = 0; i < K; i++) {
                const p = Math.max(1e-6, Math.min(1 - 1e-6, scores[i]));
                logits[i] = Math.log(p / (1 - p)); if (logits[i] > maxL) maxL = logits[i];
            }
            logits[K] = 0;
            let sum = 0; const sm = new Array(K + 1);
            for (let i = 0; i <= K; i++) { sm[i] = Math.exp(logits[i] - maxL); sum += sm[i]; }
            let bestS = -1, bestW = null, bestC = 5, bestSig = 0, all = {};
            for (let i = 0; i < K; i++) { sm[i] /= sum; all[words[i]] = sm[i]; if (sm[i] > bestS) { bestS = sm[i]; bestW = words[i]; bestC = cfList[i]; bestSig = scores[i]; } }
            debug.lastScores = all;
            return { word: bestW, prob: bestS, sigmoid: bestSig, bg: sm[K] / sum, all, consFrames: bestC };
        }

        // ═══════════════════════════
        // Detection
        // ═══════════════════════════

        // ═══════════════════════════════════════════════
        // record() — matches Java DetectionLogic.record()
        // ═══════════════════════════════════════════════
        function record(prob, word, rms, now) {
            pHist[pHi] = prob; tHist[pHi] = now; pHi = (pHi + 1) % HIST;
            if (!word) bg = bg * 0.995 + prob * 0.005;
            rmsHist[ri] = rms; rmsTHist[ri] = now; ri = (ri + 1) % RMS_HIST;
        }

        // ═══════════════════════════════════════════════
        // evaluate() — matches Java DetectionLogic.evaluate()
        // ═══════════════════════════════════════════════
        let _frameN = 0, _lastProbAtPending = null;
        function evaluate(word, prob, rms, threshold, consFrames, now) {
            _frameN++;
            _lastBlock = '';
            let triggerWord = null, triggerProb = 0;

            // L5b: post-speech confirmation
            if (L5 && pendingWord != null) {
                const elapsed = now - pendingTime;
                if (elapsed >= POST_DELAY) {
                    const tailStart = now - POST_TAIL;
                    let postMin = Infinity, postN = 0;
                    for (let i = 0; i < RMS_HIST; i++) {
                        if (rmsTHist[i] > 0 && rmsTHist[i] >= tailStart && rmsTHist[i] <= now) {
                            const v = rmsHist[i]; if (v < postMin) postMin = v; postN++;
                        }
                    }
                    if (postN >= 3 && postMin < pendingPreMin * RETURN_RATIO) {
                        triggerWord = pendingWord; triggerProb = pendingProb;
                    } else {
                        console.log(`  L5b FAIL postMin=${postMin.toFixed(0)} need<${(pendingPreMin*RETURN_RATIO).toFixed(0)}`);
                        _lastProbAtPending = null;
                    }
                    pendingWord = null;
                }
            }

            // L1-L5
            if (triggerWord == null) {
                const hi = prob > threshold && word;
                if (!hi) { cons = 0; consWord = ''; return null; }  // threshold gate, always active
                // L1: consecutive frames with gap tolerance
                if (L1) {
                    if (hi && word === consWord) { cons++; consGap = 0; }
                    else if (hi) { consWord = word; cons = 1; consGap = 0; }
                    else if (cons > 0) { consGap++; if (consGap > MAX_GAP) { cons = 0; consWord = ''; consGap = 0; } }
                    if (cons < consFrames) { if (hi) console.log(`  L1 cons=${cons}/${consFrames}`); _lastBlock = 'L1'; return null; }
                }
                // L2: peak / background (peak is always computed, matches Android)
                let peak = 0;
                for (let i = 0; i < HIST; i++) {
                    if (tHist[i] > 0 && (now - tHist[i]) < PEAK_WIN) {
                        const p = pHist[i]; if (p > peak) peak = p;
                    }
                }
                if (L2 && peak <= bg * 3) { console.log(`  L2 BLOCK peak=${peak.toFixed(3)} bg=${bg.toFixed(4)}`); _lastBlock = 'L2'; return null; }
                // L3: cooldown
                if (L3 && (now - lastTrig) < CD_MS) { console.log(`  L3 cooldown ${now-lastTrig}ms`); _lastBlock = 'L3'; return null; }
                // L4a: burst cooldown
                if (L4 && now < blocked) { console.log(`  L4 blocked ${blocked-now}ms`); _lastBlock = 'L4'; return null; }
                // L5a: energy jump ratio
                if (L5) {
                    const preStart = now - PRE_WIN_END, preEnd = now - PRE_WIN_START;
                    let preMin = Infinity, preN = 0;
                    for (let i = 0; i < RMS_HIST; i++) {
                        if (rmsTHist[i] > 0 && rmsTHist[i] >= preStart && rmsTHist[i] <= preEnd) {
                            const v = rmsHist[i]; if (v < preMin) preMin = v; preN++;
                        }
                    }
                    if (preN >= 5 && preMin < 50 && rms < 80) { console.log(`  L5 BLOCK quiet: rms=${rms.toFixed(0)} preMin=${preMin.toFixed(0)}`); _lastBlock = 'L5:quiet'; return null; }
                    if (preN >= 5 && rms < preMin + l5delta) { console.log(`  L5 BLOCK delta: rms=${rms.toFixed(0)} preMin=${preMin.toFixed(0)} need>${(preMin+l5delta).toFixed(0)}`); _lastBlock = 'L5:delta'; return null; }
                    if (preN >= 5) { console.log(`  L5 JUMP rms=${rms.toFixed(0)} preMin=${preMin.toFixed(0)} delta=${l5delta}`); }
                    if (preN >= 5) {
                        pendingWord = word; pendingTime = now; pendingProb = peak; pendingPreMin = preMin;
                        _lastProbAtPending = prob;
                        cons = 0; consWord = ''; _lastBlock = 'L5:pending'; return null;
                    }
                }
                triggerWord = word; triggerProb = peak; _lastProbAtPending = null;
                console.log(`[DETECT] direct trigger word=${word} prob=${prob.toFixed(3)} rms=${rms.toFixed(0)}`);
            }

            // L4 burst gate (final)
            if (L4 && triggerWord != null) {
                bT[bi] = now; bW[bi] = triggerWord; bP[bi] = triggerProb; bi = (bi + 1) % BH;
                let bc = 0;
                for (let i = 0; i < BH; i++) {
                    if (bT[i] > 0 && (now - bT[i]) < BURST_WIN && triggerWord === bW[i] && bP[i] > 0.8) bc++;
                }
                if (bc >= BURST_N) {
                    blocked = now + BURST_BLOCK; cons = 0; consWord = ''; _lastBlock = 'L4:burst'; return null;
                }
            }

            if (triggerWord != null) {
                lastTrig = now; cons = 0; consWord = '';
                return triggerWord;
            }
            return null;
        }

        function reset() {
            cons = 0; consWord = ''; consGap = 0; lastTrig = 0; blocked = 0; bg = 0.001;
            pendingWord = null; pendingTime = 0; pendingProb = 0; pendingPreMin = 0;
            _lastBlock = ''; _lastProbAtPending = null;
            for (let i = 0; i < HIST; i++) { pHist[i] = 0; tHist[i] = 0; }
            for (let i = 0; i < RMS_HIST; i++) { rmsHist[i] = 0; rmsTHist[i] = 0; }
            ri = 0;
        }

        // ═══════════════════════════
        // Microphone
        // ═══════════════════════════

        function _rms(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / a.length); }
        function _resample(a, from) {
            if (from === SAMPLE_RATE) return a;
            const r = from / SAMPLE_RATE, nl = Math.round(a.length / r), o = new Float32Array(nl);
            for (let i = 0; i < nl; i++) {
                const src = i * r, lo = Math.floor(src), hi = Math.min(lo + 1, a.length - 1), f = src - lo;
                o[i] = a[lo] * (1 - f) + a[hi] * f;
            }
            return o;
        }

        async function start(onResult) {
            if (listening) { _log('[wakeword] Already running'); return; }
            stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const hw = audioCtx.sampleRate; debug.sampleRate = hw;
            _log(`[wakeword] AudioContext: ${hw}Hz`);
            const needed = Math.ceil((audioSamplesNeeded / SAMPLE_RATE) * hw);
            const stride = Math.round(0.05 * hw);
            ringBuf = new Float32Array(hw * 3); ringPos = 0; lastRingPos = 0; busy = false;
            listening = true; let collected = 0; reset();

            const src = audioCtx.createMediaStreamSource(stream);
            processorNode = audioCtx.createScriptProcessor(2048, 1, 1);

            async function run() {
                if (!listening || busy) return;
                const ns = (ringPos - lastRingPos + ringBuf.length) % ringBuf.length;
                if (ns < stride) return;
                busy = true; const pos = ringPos; lastRingPos = pos;
                const raw = new Float32Array(needed);
                for (let i = 0; i < needed; i++) raw[i] = ringBuf[(pos - needed + i + ringBuf.length) % ringBuf.length];
                try {
                    const chunk = _resample(raw, hw);
                    if (collected < needed) return;
                    const rms = _rms(chunk);
                    const result = await predict(chunk);
                    if (!listening || !result) return;
                    const now = Date.now();
                    // record() — matches Android: sigmoid for L2, word only when above threshold
                    const sig = result.sigmoid != null ? result.sigmoid : result.prob;
                    const w = (result.prob > cfgThreshold) ? result.word : '';
                    record(sig, w, rms, now);
                    // evaluate() — matches Android DetectionLogic.evaluate()
                    const d = evaluate(result.word, result.prob, rms, cfgThreshold, result.consFrames || 5, now);
                    const dispProb = (d && _lastProbAtPending) ? _lastProbAtPending : result.prob;
                    debug.lastBlock = _lastBlock; debug.bg = bg; debug.lastRms = rms;
                    if (d) onResult(d, dispProb, { bg: result.bg, all: result.all, rms, block: _lastBlock });
                } catch (e) { _warn('[wakeword] error', e.message); }
                finally { busy = false; run(); }
            }

            processorNode.onaudioprocess = (e) => {
                if (!listening) return;
                const input = e.inputBuffer.getChannelData(0);
                for (let i = 0; i < input.length; i++) { ringBuf[ringPos] = input[i] * 32767; ringPos = (ringPos + 1) % ringBuf.length; }
                collected += input.length; run();
            };
            src.connect(processorNode); processorNode.connect(audioCtx.destination);
            _log('[wakeword] Mic started');
        }

        function stop() {
            listening = false;
            if (processorNode) { processorNode.disconnect(); processorNode = null; }
            ringBuf = null; busy = false;
            if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
            if (audioCtx && audioCtx.state !== 'closed') { audioCtx.close(); audioCtx = null; }
        }

        // ═══════════════════════════
        // Public API
        // ═══════════════════════════

        return {
            load, start, stop, predict, reset,
            // Public detect: record() + evaluate() matching Android API
            detect: (word, prob, consFrames, threshold, cooldownMs, now, sigmoid) => {
                const rms = 50;  // fallback RMS when called externally
                const sig = (sigmoid != null) ? sigmoid : prob;
                record(sig, (prob > threshold) ? word : '', rms, now || Date.now());
                debug.lastBlock = _lastBlock; debug.bg = bg;
                return evaluate(word, prob, rms, threshold, consFrames, now || Date.now());
            },
            setThreshold: v => { cfgThreshold = Math.max(0.25, Math.min(0.95, v)); },
            setCooldown: v => { cfgCooldown = Math.max(500, v); },
            setDebug: v => _debugLog = v,
            setL1: v => L1 = v, setL2: v => L2 = v, setL3: v => L3 = v, setL4: v => L4 = v, setL5: v => L5 = v,
            setL5Delta: v => l5delta = Math.max(1, Math.min(300, v)),
            isLoaded: () => !!melSession && models.length > 0,
            getModels: () => models.map(m => ({ name: m.name, consFrames: m.consFrames })),
            debug,
        };
    },
};

})();

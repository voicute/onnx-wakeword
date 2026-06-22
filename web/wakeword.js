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

        // ---- Detection state ----
        let cons = 0, consWord = '';
        let lastTrig = 0, blocked = 0;
        let bgEma = 0.001;
        const PEAK_HIST = 128;
        const pHist = new Float32Array(PEAK_HIST);
        const tHist = new Float32Array(PEAK_HIST); let pHi = 0;

        // Layer toggles
        let L1 = true, L2 = false, L3 = false, L4 = false, L5 = false;
        let l5ratio = 3.0;  // L5 energy jump ratio, adjustable (2.0-8.0)
        let l5rms = 0;
        const RMS_HIST = 128;
        const rmsHist = new Float32Array(RMS_HIST);
        const rmsTHist = new Array(RMS_HIST).fill(0); let l5ri = 0;
        const bT = new Float32Array(8), bW = new Array(8); let bi = 0;

        let _debugLog = false;
        const _log = (...a) => { if (_debugLog) console.log(...a); };
        const _warn = (...a) => { if (_debugLog) console.warn(...a); };
        const debug = { sampleRate: 0, lastScores: {}, inferCount: 0, consCount: 0 };

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
            let bestS = -1, bestW = null, bestC = 5, all = {};
            for (let i = 0; i < K; i++) { sm[i] /= sum; all[words[i]] = sm[i]; if (sm[i] > bestS) { bestS = sm[i]; bestW = words[i]; bestC = cfList[i]; } }
            debug.lastScores = all;
            return { word: bestW, prob: bestS, bg: sm[K] / sum, all, consFrames: bestC };
        }

        // ═══════════════════════════
        // Detection
        // ═══════════════════════════

        function detect(word, prob, consFrames, threshold, cooldownMs, now) {
            // Track peak history and background EMA for ALL frames (before early return)
            // bgEma only updates on non-wake-word frames (matches Android: word==null → update)
            pHist[pHi] = prob; tHist[pHi] = now; pHi = (pHi + 1) % PEAK_HIST;
            if (!word || prob < threshold) bgEma = bgEma * 0.995 + prob * 0.005;

            if (!word || prob < threshold) { cons = 0; consWord = ''; return null; }
            if (now < blocked) { cons = 0; consWord = ''; return null; }
            debug.consCount = cons;

            // L5: energy jump ratio
            if (L5 && l5rms > 0) {
                const ps = now - 2000, pe = now - 500;
                let pMin = Infinity, pN = 0;
                for (let i = 0; i < RMS_HIST; i++) {
                    if (rmsTHist[i] > 0 && rmsTHist[i] >= ps && rmsTHist[i] <= pe) {
                        if (rmsHist[i] < pMin) pMin = rmsHist[i]; pN++;
                    }
                }
                const ratio = pN >= 5 ? l5rms / Math.max(pMin, 1) : -1;
                let block = pN >= 5 && l5rms < pMin * l5ratio;
                if (!block && pN >= 5 && pMin < 50 && l5rms < 80) block = true;
                if (_debugLog && (debug.inferCount % 5 === 0 || block)) {
                    _log(`[L5] rms=${l5rms.toFixed(0)} preMin=${pN>=5?pMin.toFixed(0):'?'} preN=${pN} ratio=${ratio>=0?ratio.toFixed(1):'?'} blocked=${block} word=${word} prob=${prob.toFixed(3)}`);
                }
                if (block) { cons = 0; consWord = ''; return null; }
            }

            // L1: consecutive frames
            if (L1) {
                const hi = prob > threshold && word;
                if (hi && word === consWord) cons++; else if (hi) { consWord = word; cons = 1; } else { cons = 0; consWord = ''; }
                if (cons < consFrames) return null;
            }

            // L2: peak/background ratio — matches Android DetectionLogic L2
            if (L2) {
                let peak = 0;
                for (let i = 0; i < PEAK_HIST; i++) {
                    if (tHist[i] > 0 && (now - tHist[i]) < 1500) {
                        if (pHist[i] > peak) peak = pHist[i];
                    }
                }
                if (peak <= bgEma * 3) { cons = 0; consWord = ''; return null; }
            }

            // L3: cooldown
            if (L3 && (now - lastTrig) < cooldownMs) { cons = 0; consWord = ''; return null; }

            // L4: burst
            if (L4) {
                bT[bi] = now; bW[bi] = word; bi = (bi + 1) % 8;
                let bc = 0; for (let i = 0; i < 8; i++) if (bT[i] > 0 && (now - bT[i]) < 3000 && word === bW[i]) bc++;
                if (bc >= 3) { blocked = now + 5000; cons = 0; consWord = ''; return null; }
            }

            cons = 0; consWord = ''; lastTrig = now;
            return word;
        }

        function reset() {
            cons = 0; consWord = ''; lastTrig = 0; blocked = 0; bgEma = 0.001;
            for (let i = 0; i < PEAK_HIST; i++) { pHist[i] = 0; tHist[i] = 0; }
            for (let i = 0; i < RMS_HIST; i++) { rmsHist[i] = 0; rmsTHist[i] = 0; }
            l5ri = 0; l5rms = 0;
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
                    l5rms = rms; rmsHist[l5ri] = rms; rmsTHist[l5ri] = Date.now(); l5ri = (l5ri + 1) % RMS_HIST;
                    const d = detect(result.word, result.prob, result.consFrames || 5, cfgThreshold, cfgCooldown, Date.now());
                    if (d) onResult(d, result.prob, { bg: result.bg, all: result.all, rms });
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
            load, start, stop, predict, detect, reset,
            setThreshold: v => { cfgThreshold = Math.max(0.3, Math.min(0.95, v)); },
            setCooldown: v => { cfgCooldown = Math.max(500, v); },
            setDebug: v => _debugLog = v,
            setL1: v => L1 = v, setL2: v => L2 = v, setL3: v => L3 = v, setL4: v => L4 = v, setL5: v => L5 = v,
            setL5Ratio: v => l5ratio = Math.max(2.0, Math.min(8.0, v)),
            isLoaded: () => !!melSession && models.length > 0,
            getModels: () => models.map(m => ({ name: m.name, consFrames: m.consFrames })),
            debug,
        };
    },
};

})();

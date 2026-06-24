"""
Voicute Wake Word Engine — Linux/Python Edition v9.0

Usage:
    from wakeword_engine import WakeWordEngine

    engine = WakeWordEngine()
    engine.load('models/model_info.json', 'models/melspectrogram.onnx')
    engine.start(lambda word, prob, info: print(f'Detected: {word} {prob:.0%}'))

Dependencies:
    pip install onnxruntime numpy pyaudio
"""

import json, struct, time, wave, os
import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
MEL_HOP = 160
MEL_WIN = 400
N_MELS = 32


class WakeWordEngine:
    def __init__(self):
        self.mel_sess = None
        self.models = []
        self.dscnn_mode = False
        self.dscnn_mel_time = 98
        self.audio_samples_needed = 0

        # Detection state
        self.cons = 0
        self.cons_word = ''
        self.last_trig = 0
        self.blocked = 0
        self.bg_ema = 0.001
        self.peak_hist = np.zeros(128, dtype=np.float32)
        self.time_hist = np.zeros(128, dtype=np.float64)
        self.phi = 0

        # Layer toggles
        self.L1 = True
        self.L2 = False
        self.L3 = False
        self.L4 = False
        self.L5 = False
        self.l5_rms = 0
        self.l5_ratio = 3.0
        self.rms_hist = np.zeros(128, dtype=np.float32)
        self.rms_t_hist = np.zeros(128, dtype=np.float64)
        self.l5_ri = 0
        self.burst_t = np.zeros(8, dtype=np.float64)
        self.burst_w = [''] * 8
        self.bi = 0

        self.threshold = 0.40
        self.cooldown_ms = 1500
        self._stream = None
        self._running = False

    # ═══════════════════════════
    # Model loading
    # ═══════════════════════════

    def load(self, model_info_path, mel_path):
        """Load models from model_info.json (or ZIP) and melspectrogram."""
        self.mel_sess = ort.InferenceSession(mel_path, providers=['CPUExecutionProvider'])

        # Check if ZIP
        with open(model_info_path, 'rb') as f:
            header = f.read(4)
        if header[:2] == b'PK':
            import zipfile
            with zipfile.ZipFile(model_info_path, 'r') as zf:
                info = json.loads(zf.read('model_info.json'))
                self._zip = zf
        else:
            info = json.load(open(model_info_path, 'r', encoding='utf-8'))
            self._zip = None

        self.dscnn_mode = info.get('model_type') == 'dscnn'
        self.dscnn_mel_time = info.get('mel_time', 98)
        cfg = info.get('models', [info]) if info.get('multi_model') else [info]

        base = os.path.dirname(model_info_path)
        self.models = []
        for m in cfg:
            if self._zip:
                data = self._zip.read(m['model_file'])
                sess = ort.InferenceSession(data, providers=['CPUExecutionProvider'])
            else:
                path = os.path.join(base, m['model_file'])
                sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            self.models.append({
                'name': m['wake_word'],
                'session': sess,
                'cons_frames': m.get('cons_frames', 3),
            })

        if self.dscnn_mode:
            self.audio_samples_needed = self.dscnn_mel_time * MEL_HOP + MEL_WIN
        else:
            max_frames = max(m.get('emb_frames', 1) for m in self.models)
            self.audio_samples_needed = (76 + (max_frames - 1) * 8) * MEL_HOP + MEL_WIN

    # ═══════════════════════════
    # Inference
    # ═══════════════════════════

    def predict(self, audio):
        """Run inference on float32 audio array (int16 range, 16kHz). Returns dict or None."""
        if self.mel_sess is None or not self.models:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < self.audio_samples_needed:
            audio = np.pad(audio, (0, self.audio_samples_needed - len(audio)))
        audio = audio[:self.audio_samples_needed]

        mel_out = self.mel_sess.run(None, {'input': audio.reshape(1, -1)})
        mel = mel_out[0]
        frames = mel.shape[2]
        mel2d = mel[0, 0] / 10.0 + 2.0

        scores, words, cf_list = [], [], []
        if self.dscnn_mode:
            start = max(0, frames - self.dscnn_mel_time)
            dscnn_in = np.zeros((1, self.dscnn_mel_time, N_MELS), dtype=np.float32)
            for f in range(self.dscnn_mel_time):
                src = start + f
                if 0 <= src < frames:
                    dscnn_in[0, f] = mel2d[src]
            for m in self.models:
                out = m['session'].run(None, {'input': dscnn_in})
                scores.append(float(out[0].flatten()[0]))
                words.append(m['name'])
                cf_list.append(m['cons_frames'])
        else:
            return None

        # Sigmoid → softmax
        K = len(self.models)
        logits = np.zeros(K + 1, dtype=np.float64)
        for i in range(K):
            p = max(1e-6, min(1 - 1e-6, scores[i]))
            logits[i] = np.log(p / (1 - p))
        logits[K] = 0
        sm = np.exp(logits - logits.max())
        sm /= sm.sum()
        best_i = np.argmax(sm[:K])
        return {
            'word': words[best_i], 'prob': float(sm[best_i]),
            'bg': float(sm[K]), 'all': {w: float(sm[i]) for i, w in enumerate(words)},
            'cons_frames': cf_list[best_i],
        }

    # ═══════════════════════════
    # Detection
    # ═══════════════════════════

    def detect(self, word, prob, cons_frames, now=None):
        """Run detection pipeline. Returns detected word or None."""
        if now is None:
            now = time.time() * 1000
        if not word or prob < self.threshold:
            self.cons = 0; self.cons_word = ''; return None
        if now < self.blocked:
            self.cons = 0; self.cons_word = ''; return None

        # L5: energy jump
        if self.L5 and self.l5_rms > 0:
            ps, pe = now - 2000, now - 500
            mask = (self.rms_t_hist > 0) & (self.rms_t_hist >= ps) & (self.rms_t_hist <= pe)
            pN = mask.sum()
            if pN >= 5:
                pMin = self.rms_hist[mask].min()
                ratio = self.l5_rms / max(pMin, 1)
                block = self.l5_rms < pMin * self.l5_ratio
                if not block and pMin < 50 and self.l5_rms < 80:
                    block = True
                if block:
                    self.cons = 0; self.cons_word = ''; return None

        # L1: consecutive frames
        if self.L1:
            hi = prob > self.threshold and word
            if hi and word == self.cons_word:
                self.cons += 1
            elif hi:
                self.cons_word = word; self.cons = 1
            else:
                self.cons = 0; self.cons_word = ''
            if self.cons < cons_frames:
                return None

        # L2: peak/background
        if self.L2:
            self.peak_hist[self.phi] = prob
            self.time_hist[self.phi] = now
            self.phi = (self.phi + 1) % 128
            if not word:
                self.bg_ema = self.bg_ema * 0.995 + prob * 0.005
            if prob <= self.bg_ema * 2.0:
                self.cons = 0; self.cons_word = ''; return None

        # L3: cooldown
        if self.L3 and (now - self.last_trig) < self.cooldown_ms:
            self.cons = 0; self.cons_word = ''; return None

        # L4: burst
        if self.L4:
            self.burst_t[self.bi] = now; self.burst_w[self.bi] = word
            self.bi = (self.bi + 1) % 8
            bc = sum(1 for i in range(8) if self.burst_t[i] > 0 and
                     (now - self.burst_t[i]) < 3000 and word == self.burst_w[i])
            if bc >= 3:
                self.blocked = now + 5000
                self.cons = 0; self.cons_word = ''; return None

        self.cons = 0; self.cons_word = ''
        self.last_trig = now
        return word

    # ═══════════════════════════
    # Microphone
    # ═══════════════════════════

    def start(self, on_detect=None, device=None):
        """Start microphone streaming. Requires pyaudio."""
        import pyaudio
        self._running = True
        self._reset()

        p = pyaudio.PyAudio()
        self._stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
            input=True, input_device_index=device,
            frames_per_buffer=self.audio_samples_needed,
            stream_callback=self._audio_callback(on_detect, p),
        )
        self._stream.start_stream()

    def _audio_callback(self, on_detect, pyaudio_obj):
        engine = self

        def callback(in_data, frame_count, time_info, status):
            if not engine._running:
                return (None, pyaudio.paComplete)
            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio ** 2))
            result = engine.predict(audio)
            if result is None:
                return (None, pyaudio.paContinue)

            engine.l5_rms = rms
            engine.rms_hist[engine.l5_ri] = rms
            engine.rms_t_hist[engine.l5_ri] = time.time() * 1000
            engine.l5_ri = (engine.l5_ri + 1) % 128

            detected = engine.detect(
                result['word'], result['prob'],
                result['cons_frames'],
                time.time() * 1000,
            )
            if detected and on_detect:
                on_detect(detected, result['prob'], {'bg': result['bg'], 'all': result['all'], 'rms': rms})
            return (None, pyaudio.paContinue)

        return callback

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    # ═══════════════════════════
    # Config
    # ═══════════════════════════

    def _reset(self):
        self.cons = 0; self.cons_word = ''
        self.last_trig = 0; self.blocked = 0; self.bg_ema = 0.001
        self.peak_hist.fill(0); self.time_hist.fill(0); self.phi = 0
        self.rms_hist.fill(0); self.rms_t_hist.fill(0); self.l5_ri = 0; self.l5_rms = 0

    def reset(self): self._reset()
    def set_threshold(self, v): self.threshold = max(0.3, min(0.95, v))
    def set_cooldown(self, ms): self.cooldown_ms = max(500, ms)
    def set_debug(self, v): pass  # stub for API compat
    def set_L1(self, v): self.L1 = v
    def set_L2(self, v): self.L2 = v
    def set_L3(self, v): self.L3 = v
    def set_L4(self, v): self.L4 = v
    def set_L5(self, v): self.L5 = v
    def set_L5_ratio(self, v): self.l5_ratio = max(2.0, min(8.0, v))
    def is_loaded(self): return self.mel_sess is not None and len(self.models) > 0
    def get_models(self): return [{'name': m['name'], 'cons_frames': m['cons_frames']} for m in self.models]


# ═══════════════════════════
# CLI demo
# ═══════════════════════════

if __name__ == '__main__':
    import sys
    eng = WakeWordEngine()
    print('Loading models...')
    eng.load(sys.argv[1] if len(sys.argv) > 1 else 'models/model_info.json',
             sys.argv[2] if len(sys.argv) > 2 else 'models/melspectrogram.onnx')
    print(f'Loaded: {eng.get_models()}')
    print('Listening... (Ctrl+C to stop)')
    try:
        eng.start(lambda w, p, i: print(f'\n>>> {w} ({p:.0%}) <<<\n'))
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        eng.stop()
        print('Stopped.')

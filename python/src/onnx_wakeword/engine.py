"""Voicute Wake Word Engine — offline wake-word / keyword-spotting on ONNX Runtime.

The engine is audio-library agnostic: :meth:`WakeWordEngine.predict` and
:meth:`WakeWordEngine.detect` work on plain numpy arrays.  Only
:meth:`WakeWordEngine.start` (microphone streaming) needs ``pyaudio``,
installed via ``pip install onnx-wakeword[audio]``.

Example:
    from onnx_wakeword import WakeWordEngine, get_demo_model

    engine = WakeWordEngine()
    engine.load(*get_demo_model())          # or your own model_info.json
    engine.set_L1(True)
    engine.start(lambda word, prob, info: print(word, prob))

Model formats: a ``model_info.json`` next to ``.onnx`` files, or a single
``.zip`` bundle containing both (as produced by voicute.com exports).
"""

import json, struct, time, wave, os
import numpy as np
import onnxruntime as ort

from .detection_logic import DetectionLogic

SAMPLE_RATE = 16000
MEL_HOP = 160
MEL_WIN = 400
RAW_MELS = 32   # mel spectrogram output channels (unchanged)
N_MELS = 34      # classifier input: 32 mel + 2 hidden (zero-padded at runtime)
HIDDEN_PAD = 2
MAX_GAP = 2  # L1 consecutive-frames gap tolerance, matches android DetectionLogic


class WakeWordEngine:
    def __init__(self):
        self.mel_sess = None
        self.models = []
        self.dscnn_mode = False
        self.dscnn_mel_time = 98
        self.dscnn_n_mels = N_MELS  # resolved from model input shape in load()
        self.is_multi_keyword = False
        self.multi_kw_session = None
        self.keywords = []
        self.audio_samples_needed = 0

        # Detection pipeline: single production-parity implementation (the
        # port of Android DetectionLogic.java). engine.detect() delegates
        # here; mic_test.py / infer.py use the same class directly.
        self.threshold = 0.40
        self.cooldown_ms = 1500
        self.l5_rms = 0   # RMS of the current window (feeds L5); set by the
                          # mic callback, or by the caller before detect()
        self._dl = DetectionLogic(thr=self.threshold, cons_frames=3)
        # Engine defaults: all detection layers OFF — the model itself
        # carries the false-trigger rejection, so out of the box the engine
        # is a plain threshold gate (single frame above threshold triggers).
        # Enable layers case-by-case for noisy deployments, e.g.
        # set_L1(True) (consecutive frames) or set_L3(True) (cooldown);
        # see README "Detection layers" for when each one helps.
        self._dl.l1, self._dl.l2, self._dl.l3, self._dl.l4, self._dl.l5 = \
            False, False, False, False, False

        self._stream = None
        self._running = False

    # ═══════════════════════════
    # Model loading
    # ═══════════════════════════

    def load(self, model_info_path, mel_path=None):
        """Load models from model_info.json (or ZIP).

        ``mel_path`` is optional: when omitted, the melspectrogram model is
        resolved automatically — first from the model bundle itself
        (``melspectrogram.onnx`` next to the JSON, or inside the ZIP), then
        falling back to the copy bundled with this package. Most users only
        need their keyword model; the shared mel module ships in the wheel.

        Both paths may be absolute or relative; model files referenced inside
        ``model_info.json`` are resolved relative to the JSON file's own
        directory (or inside the ZIP), never the current working directory.
        """
        model_info_path = str(model_info_path)
        mel_input = self._resolve_mel(model_info_path, mel_path)
        self.mel_sess = ort.InferenceSession(mel_input, providers=['CPUExecutionProvider'])

        # Check if ZIP
        with open(model_info_path, 'rb') as f:
            header = f.read(4)
        if header[:2] == b'PK':
            import zipfile
            # Keep the archive open — model .onnx bytes are read from it below.
            zf = zipfile.ZipFile(model_info_path, 'r')
            info = json.loads(zf.read('model_info.json'))
            self._zip = zf
        else:
            info = json.load(open(model_info_path, 'r', encoding='utf-8'))
            self._zip = None

        self.dscnn_mode = info.get('model_type') in ('dscnn', 'tcn', 'multi_keyword')
        self.dscnn_mel_time = info.get('mel_time', 98)

        # Multi-keyword: single model with N outputs
        if info.get('model_type') == 'multi_keyword':
            self.is_multi_keyword = True
            self.keywords = info['keywords']
            base = os.path.dirname(model_info_path)
            model_file = info.get('model_file', 'model.onnx')
            if self._zip:
                data = self._zip.read(model_file)
                self.multi_kw_session = ort.InferenceSession(data, providers=['CPUExecutionProvider'])
            else:
                self.multi_kw_session = ort.InferenceSession(os.path.join(base, model_file), providers=['CPUExecutionProvider'])
            self.models = [{'name': kw, 'cons_frames': info.get('cons_frames', 2)} for kw in self.keywords]
        # Legacy: multi-model (separate ONNX per keyword)
        else:
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
            # Resolve the classifier's real mel-channel width from its input
            # shape: current-gen tcn models take 32 cols; legacy DS-CNN took
            # 34 (32 mel + 2 hidden pad). Feeding the wrong width only fails
            # at first inference with an ORT shape error — resolve it here.
            sessions = ([self.multi_kw_session] if self.is_multi_keyword
                        else [m['session'] for m in self.models])
            self.dscnn_n_mels = self._resolve_input_cols(sessions[0], self.dscnn_mel_time)
            for s in sessions[1:]:
                if self._resolve_input_cols(s, self.dscnn_mel_time) != self.dscnn_n_mels:
                    raise ValueError('model bundle mixes different input mel widths')
            self.audio_samples_needed = self.dscnn_mel_time * MEL_HOP + MEL_WIN
        else:
            max_frames = max(m.get('emb_frames', 1) for m in self.models)
            self.audio_samples_needed = (76 + (max_frames - 1) * 8) * MEL_HOP + MEL_WIN

    @staticmethod
    def _resolve_mel(model_info_path, mel_path=None):
        """Resolve the melspectrogram model for a bundle. Priority: explicit
        mel_path > copy inside the bundle (dir or ZIP; forward compatibility
        if the mel ever changes) > the copy bundled with this package.
        Returns a filesystem path, or model bytes for a ZIP-embedded mel
        (ORT accepts both)."""
        from .demo import get_demo_model
        if mel_path is not None:
            return str(mel_path)
        with open(model_info_path, 'rb') as f:
            if f.read(2) == b'PK':
                import zipfile
                with zipfile.ZipFile(model_info_path, 'r') as zf:
                    if 'melspectrogram.onnx' in zf.namelist():
                        return zf.read('melspectrogram.onnx')
            else:
                local = os.path.join(os.path.dirname(model_info_path), 'melspectrogram.onnx')
                if os.path.exists(local):
                    return local
        return get_demo_model()[1]

    @staticmethod
    def _resolve_input_cols(session, expect_mel_time=None):
        """Validate mel_time and return the input's mel width
        ([batch, mel_time, cols])."""
        shape = session.get_inputs()[0].shape
        mt = shape[-2]
        if expect_mel_time is not None and isinstance(mt, int) and mt > 0 \
                and mt != expect_mel_time:
            raise ValueError(f'mel_time mismatch: config mel_time={expect_mel_time} '
                             f'but model input shape={shape}')
        last = shape[-1] if isinstance(shape[-1], int) and shape[-1] > 0 else RAW_MELS
        return last

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
            dscnn_in = np.zeros((1, self.dscnn_mel_time, self.dscnn_n_mels), dtype=np.float32)
            for f in range(self.dscnn_mel_time):
                src = start + f
                if 0 <= src < frames:
                    # Copy first RAW_MELS channels, last HIDDEN_PAD stay zero
                    dscnn_in[0, f, :RAW_MELS] = mel2d[src]
            # Multi-keyword: single model, single inference -> [1, N]
            if self.is_multi_keyword:
                out = self.multi_kw_session.run(None, {'input': dscnn_in})
                data = out[0].flatten()
                for i, kw in enumerate(self.keywords):
                    scores.append(float(data[i]))
                    words.append(kw)
                    cf_list.append(self.models[i]['cons_frames'])
            # Legacy: loop over multiple models
            else:
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
        """Run the detection pipeline (L1-L5, Android DetectionLogic parity).

        Returns the detected word or None. Feeds this frame into the shared
        DetectionLogic history first (like the Android record→evaluate flow),
        so L2 peak/background and L5 RMS history build up across calls.

        L5 uses ``self.l5_rms`` as the current window's RMS — the pyaudio
        callback sets it automatically; standalone callers should set it
        before calling (L5 off by default, so 0 is fine otherwise).
        """
        if now is None:
            now = time.time() * 1000
        self._dl.cons_frames = cons_frames
        hi_word = word if (prob > self._dl.thr and word) else ''
        self._dl.record(prob, hi_word, self.l5_rms, now)
        return self._dl.evaluate(word, prob, self.l5_rms, now)

    # ═══════════════════════════
    # Microphone
    # ═══════════════════════════

    def start(self, on_detect=None, device=None):
        """Start microphone streaming. Requires pyaudio
        (``pip install onnx-wakeword[audio]``)."""
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
        import pyaudio as _pa

        def callback(in_data, frame_count, time_info, status):
            if not engine._running:
                return (None, _pa.paComplete)
            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio ** 2))
            result = engine.predict(audio)
            if result is None:
                return (None, pyaudio.paContinue)

            engine.l5_rms = rms
            detected = engine.detect(
                result['word'], result['prob'],
                result['cons_frames'],
                time.time() * 1000,
            )
            if detected and on_detect:
                on_detect(detected, result['prob'], {'bg': result['bg'], 'all': result['all'], 'rms': rms})
            return (None, _pa.paContinue)

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
        self.l5_rms = 0
        self._dl.reset()

    def reset(self): self._reset()
    def set_threshold(self, v):
        self.threshold = max(0.3, min(0.95, v))
        self._dl.thr = self.threshold
    def set_cooldown(self, ms):
        self.cooldown_ms = max(500, ms)
        self._dl.CD_MS = self.cooldown_ms
    def set_debug(self, v): pass  # stub for API compat
    def set_L1(self, v): self._dl.l1 = bool(v)
    def set_L2(self, v): self._dl.l2 = bool(v)
    def set_L3(self, v): self._dl.l3 = bool(v)
    def set_L4(self, v): self._dl.l4 = bool(v)
    def set_L5(self, v): self._dl.l5 = bool(v)
    def set_L5_delta(self, v): self._dl.l5_delta = int(v)
    def set_L5_ratio(self, v): pass  # deprecated: DetectionLogic uses an absolute delta (set_L5_delta)
    def is_loaded(self): return self.mel_sess is not None and len(self.models) > 0
    def get_models(self): return [{'name': m['name'], 'cons_frames': m['cons_frames']} for m in self.models]

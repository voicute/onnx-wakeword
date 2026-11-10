"""``onnx-wakeword-demo`` command-line interface.

Two modes:

* Microphone (default) — needs the ``pyaudio`` extra.
* Offline WAV (``--wav file.wav``) — no audio library required; useful on
  headless machines and as a smoke test.

Both modes run the same engine + detection pipeline used in production
(Voicute Android / web).
"""

import argparse
import sys
import time
import wave

import numpy as np

from .engine import WakeWordEngine, SAMPLE_RATE
from .detection_logic import DetectionLogic
from .demo import get_demo_model

HOP = 640  # 40 ms processing hop for WAV mode


def _load_wav(path):
    """Read a WAV file as float32 int16-range mono samples. Resamples naive
    linearly if the sample rate differs from 16 kHz."""
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise SystemExit(f'unsupported sample width {width * 8} bits (need 16-bit PCM): {path}')
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        audio = audio[::ch]
    if sr != SAMPLE_RATE:
        import math
        new_n = int(len(audio) * SAMPLE_RATE / sr)
        idx = np.linspace(0, len(audio) - 1, new_n)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _run_wav(eng, dl, path, thr):
    audio = _load_wav(path)
    needed = eng.audio_samples_needed
    dur = len(audio) / SAMPLE_RATE
    print(f'File: {path}  ({dur:.1f}s, {len(eng.get_models())} model(s))')
    print(f'Words: {" | ".join(m["name"] for m in eng.get_models())}')
    print('Scanning...')
    detections = 0
    n_windows = max(0, (len(audio) - needed) // HOP + 1)
    for i in range(n_windows):
        chunk = audio[i * HOP:i * HOP + needed]
        r = eng.predict(chunk)
        if r is None:
            continue
        now_ms = int(time.time() * 1000)
        prob = r['prob']
        word = r['word'] if prob > thr else ''
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        dl.record(prob, word, rms, now_ms)
        result = dl.evaluate(word, prob, rms, now_ms)
        if result:
            t_sec = (i * HOP + needed) / SAMPLE_RATE
            print(f'>>> {result}  prob={prob:.3f}  @ {t_sec:.1f}s')
            detections += 1
    print(f'Done. {detections} detection(s).')


def _run_mic(eng, path_info, path_mel, args):
    try:
        eng.load(path_info, path_mel)
        print(f'Loaded: {eng.get_models()}')
        print('Listening... (Ctrl+C to stop)')
        eng.start(
            lambda w, p, i: print(f'\n>>> {w} ({p:.0%}) <<<\n'),
            device=args.device,
        )
        while True:
            time.sleep(0.1)
    except ImportError:
        print(
            'pyaudio is not installed — the microphone demo needs it.\n'
            '  pip install onnx-wakeword[audio]\n'
            'or run offline on a WAV file instead:\n'
            '  onnx-wakeword-demo --wav speech.wav',
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        eng.stop()
        print('\nStopped.')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='onnx-wakeword-demo',
        description='Voicute offline wake-word demo (bundled "Hey Jarvis" model by default)')
    p.add_argument('--info', help='path to model_info.json (default: bundled demo model)')
    p.add_argument('--mel', help='path to melspectrogram.onnx (default: bundled demo)')
    p.add_argument('--wav', help='run offline on this WAV file instead of the microphone')
    p.add_argument('--device', type=int, default=None, help='microphone input device index')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--cons', type=int, default=3, help='L1 consecutive frames to trigger')
    p.add_argument('--all', action='store_true', help='enable all detection layers L1-L5')
    p.add_argument('--list-devices', action='store_true', help='list audio input devices')
    args = p.parse_args(argv)

    if args.list_devices:
        try:
            import pyaudio
        except ImportError:
            print('Listing devices requires pyaudio: pip install onnx-wakeword[audio]')
            return 1
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get('maxInputChannels', 0) > 0:
                print(f"  [{i}] {d['name']}  ({d['maxInputChannels']} ch, {int(d['defaultSampleRate'])} Hz)")
        return 0

    demo_info, demo_mel = get_demo_model()
    info = args.info or demo_info
    mel = args.mel or demo_mel

    eng = WakeWordEngine()
    eng.set_threshold(args.threshold)

    if args.wav:
        eng.load(info, mel)
        dl = DetectionLogic(thr=args.threshold, cons_frames=args.cons)
        dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = True, args.all, args.all, args.all, args.all
        _run_wav(eng, dl, args.wav, args.threshold)
        return 0
    return _run_mic(eng, info, mel, args)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Voicute Wake Word — streaming mic inference driven by the onnx_wakeword
package (sounddevice flavor; wakeword_engine.py uses pyaudio).

Usage:
    pip install onnx-wakeword[sounddevice]
    python infer.py --model-dir ../models/
    python infer.py --info ../models/en/model_info.json --threshold 0.6

Works with any model bundle: a directory containing model_info.json (or a
multi_model / multi_keyword JSON) plus melspectrogram.onnx. With no
arguments it loads the bundled English demo model.
"""

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd

try:
    from onnx_wakeword import WakeWordEngine, DetectionLogic
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from onnx_wakeword import WakeWordEngine, DetectionLogic

SAMPLE_RATE = 16000
HOP = 640  # process every 40 ms, sliding window


def find_info(model_dir):
    info = os.path.join(model_dir, 'model_info.json')
    if os.path.exists(info):
        return info
    sys.exit(f'no model_info.json in {model_dir}')


def main():
    p = argparse.ArgumentParser(description='Voicute Wake Word Inference (package API)')
    p.add_argument('--model-dir', default='../models/', help='Directory with model_info.json + melspectrogram.onnx')
    p.add_argument('--info', help='Path to model_info.json (overrides --model-dir)')
    p.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    p.add_argument('--cons', type=int, default=3, help='L1 consecutive frames')
    p.add_argument('--cooldown', type=float, default=3.5, help='Cooldown in seconds')
    p.add_argument('--list-devices', action='store_true', help='List audio devices')
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    info_path = args.info or find_info(args.model_dir)
    mel_path = os.path.join(os.path.dirname(info_path), 'melspectrogram.onnx')

    eng = WakeWordEngine()
    eng.load(info_path, mel_path)
    eng.set_threshold(args.threshold)
    eng.set_cooldown(int(args.cooldown * 1000))

    dl = DetectionLogic(thr=args.threshold, cons_frames=args.cons)
    dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = True, False, True, False, False

    print(f'Loaded: {eng.get_models()}')
    print(f'Listening... threshold={args.threshold} cooldown={args.cooldown}s')

    ring = np.zeros(max(SAMPLE_RATE * 4, eng.audio_samples_needed + SAMPLE_RATE), dtype=np.float32)
    pos = 0

    def cb(indata, frames, time_info, status):
        nonlocal pos
        n = len(indata)
        if pos + n > len(ring):
            ring[:pos - n] = ring[n - pos:]
            pos -= n
        ring[pos:pos + n] = indata[:, 0] * 32767
        pos += n
        if pos < eng.audio_samples_needed:
            return

        chunk = ring[pos - eng.audio_samples_needed:pos].copy()
        r = eng.predict(chunk)
        if r is None:
            return
        prob = r['prob']
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now_ms = int(time.time() * 1000)
        word = r['word'] if prob > args.threshold else ''

        dl.record(prob, word, rms, now_ms)
        result = dl.evaluate(word, prob, rms, now_ms)
        if result:
            print(f"\n>>> {result} (prob={prob:.3f} rms={rms:.0f})\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=cb,
                            dtype='float32', blocksize=HOP):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nDone.')


if __name__ == '__main__':
    main()

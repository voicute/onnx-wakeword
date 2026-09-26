"""Quick mic test — drives the real WakeWordEngine (wakeword_engine.py), so
mel_time / input width / wake word all come from model_info.json exactly like
production. No model parsing logic here.

Usage:
  python mic_test.py                        # loads models/model_info.json
  python mic_test.py --info ../models/zh/model_info.json
  python mic_test.py --model manbo          # finds the model_info.json referencing manbo*.onnx
  python mic_test.py --all                  # L1-L5 all on
  python mic_test.py --thr 0.6              # higher threshold
"""
import sys, os, time, argparse, glob, json
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_logic import DetectionLogic
from wakeword_engine import WakeWordEngine

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
HOP, SR = 640, 16000


def list_bundles():
    """All model_info.json bundles under models/ with their wake words."""
    out = []
    for p in sorted(glob.glob(os.path.join(MODEL_DIR, '**', 'model_info.json'),
                              recursive=True)):
        try:
            with open(p, encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            continue
        words = cfg.get('keywords') or \
            [m.get('wake_word', '?') for m in cfg.get('models', [])] or \
            [cfg.get('wake_word', '?')]
        out.append((os.path.relpath(p, MODEL_DIR), ', '.join(map(str, words))))
    return out


def find_info(args):
    """Resolve which model_info.json to load."""
    if args.info:
        if not os.path.exists(args.info):
            sys.exit(f'model_info not found: {args.info}')
        return args.info
    if args.model:
        hits = []
        for p in glob.glob(os.path.join(MODEL_DIR, '**', 'model_info.json'),
                           recursive=True):
            try:
                with open(p, encoding='utf-8') as f:
                    cfg = json.load(f)
            except Exception:
                continue
            refs = [cfg.get('model_file', '')] + \
                   [m.get('model_file', '') for m in cfg.get('models', [])]
            if any(args.model in os.path.basename(r) and '_test' not in r
                   for r in refs if r):
                hits.append(p)
        if not hits:
            sys.exit(f'no model_info.json references "{args.model}". '
                     f'Available bundles:\n' +
                     '\n'.join(f'  {p}  ({w})' for p, w in list_bundles()))
        return hits[0]
    dflt = os.path.join(MODEL_DIR, 'model_info.json')
    if os.path.exists(dflt):
        return dflt
    sys.exit('no models/model_info.json and no --info/--model given. '
             'Available bundles:\n' +
             '\n'.join(f'  {p}  ({w})' for p, w in list_bundles()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--info', help='Path to model_info.json (default: models/model_info.json)')
    p.add_argument('--model', help='Find the model_info.json whose model file matches this name')
    p.add_argument('--thr', type=float, default=0.5)
    p.add_argument('--cons', type=int, default=2)
    p.add_argument('--all', action='store_true', help='Enable all L1-L5')
    p.add_argument('--l1', type=int, default=1, help='L1 on/off')
    p.add_argument('--l2', type=int, default=None)
    p.add_argument('--l3', type=int, default=None)
    p.add_argument('--l4', type=int, default=None)
    p.add_argument('--l5', type=int, default=None)
    p.add_argument('--l5-delta', type=int, default=1200, help='L5 delta: curRms > preMin + delta')
    p.add_argument('--list-devices', action='store_true')
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Defaults: L1+L3 (safe baseline). --all → L1-L5 all on
    if args.all:
        l1, l2, l3, l4, l5 = 1, 1, 1, 1, 1
    else:
        l1 = args.l1
        l2 = 0 if args.l2 is None else args.l2
        l3 = 1 if args.l3 is None else args.l3
        l4 = 0 if args.l4 is None else args.l4
        l5 = 0 if args.l5 is None else args.l5

    info_path = find_info(args)
    mel_path = os.path.join(os.path.dirname(info_path), 'melspectrogram.onnx')
    if not os.path.exists(mel_path):
        mel_path = os.path.join(MODEL_DIR, 'melspectrogram.onnx')

    eng = WakeWordEngine()
    eng.load(info_path, mel_path)
    if not eng.is_loaded():
        sys.exit(f'engine failed to load {info_path}: {eng.getErrorMessage() if hasattr(eng, "getErrorMessage") else "see log"}')
    needed = eng.audio_samples_needed
    words = [m['name'] for m in eng.get_models()]

    dl = DetectionLogic(thr=args.thr, cons_frames=args.cons)
    dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = bool(l1), bool(l2), bool(l3), bool(l4), bool(l5)
    if l5:
        dl.l5_delta = args.l5_delta

    layers = ''.join([f'L{i+1}' for i, v in enumerate([l1,l2,l3,l4,l5]) if v])
    print(f'Bundle: {info_path}')
    print(f'Words: {" | ".join(words)}  mel_time={eng.dscnn_mel_time} '
          f'n_mels={eng.dscnn_n_mels} audio_win={needed}')
    print(f'Layers: {layers}  thr={args.thr}  cons={args.cons}  L5={dl.l5_delta}')
    print(f'Say the wake word... (Ctrl+C to stop)')
    print()

    ring = np.zeros(max(SR * 4, needed + SR), dtype=np.float32)
    pos = 0

    def cb(indata, frames, info, status):
        nonlocal pos
        n = len(indata)
        if pos + n > len(ring):
            ring[:pos-n] = ring[n-pos:]
            pos -= n
        ring[pos:pos+n] = indata[:, 0] * 32767
        pos += n
        if pos < needed:
            return

        chunk = ring[pos-needed:pos].copy()
        r = eng.predict(chunk)
        if r is None:
            return
        prob = r['prob']
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now_ms = int(time.time() * 1000)
        word = r['word'] if prob > args.thr else ''

        dl.record(prob, word, rms, now_ms)
        result = dl.evaluate(word, prob, rms, now_ms)

        if prob > 0.3:
            bar = '#' * int(prob * 40)
            tag = f'[{dl.cons}/{args.cons}]' if dl.cons > 0 else ''
            print(f'  [{bar:<40}] {prob:.3f} {tag} {r["word"]} rms={rms:.0f}  ', end='\r')
        if result:
            print(f'\n>>> {result} (prob={prob:.3f} rms={rms:.0f})\n')

    try:
        with sd.InputStream(samplerate=SR, channels=1, callback=cb, dtype='float32', blocksize=HOP):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nDone.')

if __name__ == '__main__':
    main()

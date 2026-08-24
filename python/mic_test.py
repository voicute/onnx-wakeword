"""Quick mic test — standalone, no modification to existing code.

Usage:
  python mic_test.py                        # L1+L3 only, hey_limi
  python mic_test.py --all                  # L1-L5 all on
  python mic_test.py --model nihaodiannao   # different model by name
  python mic_test.py --path ../models/zh/manbo.onnx  # full path
  python mic_test.py --thr 0.6              # higher threshold
"""
import sys, os, time, argparse, numpy as np
import onnxruntime as ort, sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_logic import DetectionLogic

MEL_TIME, N_MELS = 98, 32
AUDIO_WIN = (MEL_TIME - 1) * 160 + 512 + 160
HOP, SR = 640, 16000

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='xiaona', help='Model name (xiaona, manbo, gugugaga, etc.)')
    p.add_argument('--path', help='Full path to .onnx model file (overrides --model)')
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

    # Model name → wake word mapping
    WORD_MAP = {'xiaona': '小娜', 'manbo_voice_model': '曼波', 'manbo': '曼波', 'nihaodiannao': '你好电脑',
                'kaishibofang': '开始播放', 'gugugaga': '咕咕嘎嘎', 'laifu': '来福'}
    wake_word = WORD_MAP.get(args.model, args.model)

    # Model path: --path overrides --model search
    if args.path:
        model_path = args.path
    else:
        MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_path = os.path.join(MODEL_DIR, f'{args.model}.onnx')
        # Search models/ root first, then subdirectories (zh/, en/, de/, fr/)
        if not os.path.exists(model_path):
            for subdir in ('zh', 'en', 'de', 'fr', 'ja'):
                candidate = os.path.join(MODEL_DIR, subdir, f'{args.model}.onnx')
                if os.path.exists(candidate):
                    model_path = candidate
                    break
                break
    mel_path = os.path.join(MODEL_DIR, 'melspectrogram.onnx')

    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        return

    mel = ort.InferenceSession(mel_path, providers=['CPUExecutionProvider'])
    model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    dl = DetectionLogic(thr=args.thr, cons_frames=args.cons)
    dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = bool(l1), bool(l2), bool(l3), bool(l4), bool(l5)
    if l5:
        dl.l5_delta = args.l5_delta

    layers = ''.join([f'L{i+1}' for i, v in enumerate([l1,l2,l3,l4,l5]) if v])
    print(f'Model: {model_path}')
    print(f'Layers: {layers}  thr={args.thr}  cons={args.cons}  L5={dl.l5_delta}')
    print(f'Say the wake word... (Ctrl+C to stop)')
    print()

    ring = np.zeros(SR * 4, dtype=np.float32)
    pos = 0

    def cb(indata, frames, info, status):
        nonlocal pos
        n = len(indata)
        if pos + n > len(ring):
            ring[:pos-n] = ring[n-pos:]
            pos -= n
        ring[pos:pos+n] = indata[:, 0] * 32767
        pos += n
        if pos < AUDIO_WIN:
            return

        chunk = ring[pos-AUDIO_WIN:pos].copy()
        mel_out = mel.run(None, {'input': chunk.reshape(1, -1).astype(np.float32)})[0]
        fr = mel_out.shape[2]
        mel_data = mel_out[0, 0] / 10.0 + 2.0
        ms = max(0, fr - MEL_TIME)
        tcn_in = np.zeros((1, MEL_TIME, N_MELS), dtype=np.float32)
        for f in range(MEL_TIME):
            s = ms + f
            if s < fr: tcn_in[0, f, :] = mel_data[s, :]

        prob = float(model.run(None, {'input': tcn_in})[0][0, 0])
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now_ms = int(time.time() * 1000)
        word = wake_word if prob > args.thr else ''

        dl.record(prob, word, rms, now_ms)
        result = dl.evaluate(word, prob, rms, now_ms)

        if prob > 0.3:
            bar = '#' * int(prob * 40)
            tag = f'[{dl.cons}/{args.cons}]' if dl.cons > 0 else ''
            print(f'  [{bar:<40}] {prob:.3f} {tag} rms={rms:.0f}  ', end='\r')
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

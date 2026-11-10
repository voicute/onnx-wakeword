"""Backwards-compatible launcher — the implementation now lives in the
``onnx_wakeword`` package (``src/onnx_wakeword/engine.py``).

Running this file directly still works, with or without ``pip install``:

    python wakeword_engine.py [model_info.json] [melspectrogram.onnx]

With no arguments it uses the bundled English demo model ("Hey Jarvis").
"""
import os, sys, time

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, 'src')
if os.path.isdir(os.path.join(_src, 'onnx_wakeword')) and _src not in sys.path:
    sys.path.insert(0, _src)  # allow running without installing the package

from onnx_wakeword import WakeWordEngine, get_demo_model  # noqa: E402


if __name__ == '__main__':
    if len(sys.argv) > 1:
        info = sys.argv[1]
        mel = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(info), 'melspectrogram.onnx')
        if not os.path.exists(mel):
            mel = os.path.join(_here, 'src', 'onnx_wakeword', 'resources', 'melspectrogram.onnx')
    else:
        info, mel = get_demo_model()
    eng = WakeWordEngine()
    print('Loading models...')
    eng.load(info, mel)
    print(f'Loaded: {eng.get_models()}')
    print('Listening... (Ctrl+C to stop)')
    try:
        eng.start(lambda w, p, i: print(f'\n>>> {w} ({p:.0%}) <<<\n'))
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        eng.stop()
        print('Stopped.')

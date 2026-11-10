"""Microphone demo — prints every wake-word detection.

    pip install "onnx-wakeword[audio]"
    python examples/mic_demo.py

Say "Hey Jarvis" (bundled English demo model).
"""
from onnx_wakeword import WakeWordEngine, get_demo_model


def main():
    engine = WakeWordEngine()
    engine.load(*get_demo_model())
    engine.set_L1(True)          # consecutive-frames gate (default on)
    engine.set_L3(True)          # 1.5 s cooldown against double-triggers
    print('Loaded:', engine.get_models())
    print('Listening... say "Hey Jarvis" (Ctrl+C to stop)')

    def on_detect(word, probability, info):
        print(f'\n>>> {word} ({probability:.0%})  details={info}')

    try:
        engine.start(on_detect)
        while True:
            import time
            time.sleep(0.2)
    except KeyboardInterrupt:
        engine.stop()
        print('Stopped.')


if __name__ == '__main__':
    main()

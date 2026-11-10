"""Minimal example — core inference on a numpy array, no microphone needed.

    pip install onnx-wakeword
    python examples/minimal.py

This is the lowest-level entry point: predict() takes any float32 audio
(int16 range, 16 kHz) at least audio_samples_needed long and returns the
best keyword with its softmax probability. Nothing here touches an audio
library, so it runs on headless servers and in CI.
"""
import numpy as np

from onnx_wakeword import WakeWordEngine, get_demo_model


def main():
    engine = WakeWordEngine()
    engine.load(*get_demo_model())   # bundled "Hey Jarvis" model
    print('Models:', engine.get_models())
    print(f'Audio window: {engine.audio_samples_needed} samples '
          f'({engine.audio_samples_needed / 16000:.2f}s @ 16 kHz)')

    # Generate a 1-second buffer (in production this comes from your mic
    # or file stream — int16 range, mono, 16 kHz).
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000) * 100).astype(np.float32)

    result = engine.predict(audio)
    print('predict():', result)

    # feed() predict() results into the detection pipeline to get
    # production-grade triggering behavior (thresholds, L1-L5 gates):
    engine.set_threshold(0.5)
    detected = engine.detect(result['word'], result['prob'], result['cons_frames'])
    print('detect():', detected)


if __name__ == '__main__':
    main()

"""onnx-wakeword — offline wake-word / keyword-spotting engine.

Core inference needs only numpy + onnxruntime; no audio library is imported
by the package itself.  Microphone streaming via ``engine.start()`` requires
the optional ``pyaudio`` extra:

    pip install onnx-wakeword[audio]

Quick start with the bundled English demo model ("Hey Jarvis"):

    from onnx_wakeword import WakeWordEngine, get_demo_model

    engine = WakeWordEngine()
    engine.load(*get_demo_model())
    engine.set_L1(True)
    engine.start(lambda word, prob, info: print(word, prob))

Or try it from your terminal:

    onnx-wakeword-demo
"""

from .engine import WakeWordEngine, SAMPLE_RATE
from .detection_logic import DetectionLogic
from .demo import get_demo_model

__version__ = "0.1.0"

__all__ = [
    "WakeWordEngine",
    "DetectionLogic",
    "get_demo_model",
    "SAMPLE_RATE",
    "__version__",
]

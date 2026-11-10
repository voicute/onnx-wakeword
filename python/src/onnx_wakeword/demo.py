"""Bundled demo model access.

The wheel ships one small English demo model ("Hey Jarvis", the
false-trigger-optimized R1 edition) plus the shared ``melspectrogram.onnx``
preprocessing module (~1.2 MB total).  Larger / multilingual models live in
the GitHub repo and at https://www.voicute.com — they are not bundled to
keep the wheel small.
"""

import os

_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')


def get_demo_model():
    """Return ``(model_info_path, mel_path)`` for the bundled demo model.

    Both are absolute filesystem paths inside the installed package, so they
    work regardless of the current working directory:

        from onnx_wakeword import WakeWordEngine, get_demo_model

        engine = WakeWordEngine()
        engine.load(*get_demo_model())
    """
    return (
        os.path.join(_RESOURCES_DIR, 'model_info.json'),
        os.path.join(_RESOURCES_DIR, 'melspectrogram.onnx'),
    )

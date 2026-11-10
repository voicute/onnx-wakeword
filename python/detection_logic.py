"""Backwards-compatible shim — the implementation now lives in the
``onnx_wakeword`` package (``src/onnx_wakeword/detection_logic.py``)."""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, 'src')
if os.path.isdir(os.path.join(_src, 'onnx_wakeword')) and _src not in sys.path:
    sys.path.insert(0, _src)  # allow running without installing the package

from onnx_wakeword import DetectionLogic  # noqa: E402,F401

if __name__ == '__main__':
    print('detection_logic.py is a library module - import DetectionLogic from it,')
    print('or use mic_test.py / the onnx_wakeword package.')

"""Package import and API surface tests — must pass with no audio libraries installed."""
import sys


def test_version():
    import onnx_wakeword
    assert isinstance(onnx_wakeword.__version__, str)
    assert onnx_wakeword.__version__.count('.') >= 1


def test_api_surface():
    import onnx_wakeword
    assert hasattr(onnx_wakeword, 'WakeWordEngine')
    assert hasattr(onnx_wakeword, 'DetectionLogic')
    assert hasattr(onnx_wakeword, 'get_demo_model')
    assert 'WakeWordEngine' in onnx_wakeword.__all__
    assert 'DetectionLogic' in onnx_wakeword.__all__
    assert 'get_demo_model' in onnx_wakeword.__all__


def test_core_import_does_not_pull_audio_libs():
    """Core inference must not require pyaudio or sounddevice."""
    import onnx_wakeword  # noqa: F401
    assert 'pyaudio' not in sys.modules
    assert 'sounddevice' not in sys.modules


def test_engine_public_methods():
    from onnx_wakeword import WakeWordEngine
    eng = WakeWordEngine()
    for m in ('load', 'predict', 'detect', 'start', 'stop', 'reset',
              'set_threshold', 'set_cooldown', 'set_L1', 'set_L2', 'set_L3',
              'set_L4', 'set_L5', 'set_L5_ratio', 'is_loaded', 'get_models'):
        assert callable(getattr(eng, m)), f'missing method: {m}'

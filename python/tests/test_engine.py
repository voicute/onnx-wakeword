"""Engine loading + inference tests against the bundled demo model.

No microphone and no audio library involved — pure ONNX inference.
"""
import json
import os
import zipfile

import numpy as np
import pytest

from onnx_wakeword import WakeWordEngine, get_demo_model


@pytest.fixture(scope='module')
def engine():
    eng = WakeWordEngine()
    info, mel = get_demo_model()
    eng.load(info, mel)
    return eng


@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(42)


def test_demo_paths_exist():
    info, mel = get_demo_model()
    assert os.path.isabs(info)
    assert os.path.isabs(mel)
    assert os.path.exists(info)
    assert os.path.exists(mel)
    # cwd-independence: paths are absolute inside the installed package
    assert os.path.basename(info) == 'model_info.json'
    assert os.path.basename(mel) == 'melspectrogram.onnx'


def test_load_demo_model(engine):
    assert engine.is_loaded()
    models = engine.get_models()
    assert models == [{'name': 'Hey Jarvis', 'cons_frames': 3}]
    assert engine.dscnn_mel_time == 98
    assert engine.dscnn_n_mels == 32
    assert engine.audio_samples_needed == 98 * 160 + 400  # mel_time*hop + win


def test_predict_returns_dict(engine, rng):
    audio = (rng.standard_normal(engine.audio_samples_needed) * 100).astype(np.float32)
    r = engine.predict(audio)
    assert r is not None
    assert r['word'] == 'Hey Jarvis'
    assert 0.0 <= r['prob'] <= 1.0
    assert 0.0 <= r['bg'] <= 1.0
    assert set(r['all'].keys()) == {'Hey Jarvis'}
    assert r['cons_frames'] == 3
    assert r['prob'] + r['bg'] == pytest.approx(1.0, abs=1e-6)


def test_predict_short_audio_is_padded(engine, rng):
    """A too-short buffer must be zero-padded, not crash."""
    audio = (rng.standard_normal(1600) * 100).astype(np.float32)
    r = engine.predict(audio)
    assert r is not None
    assert 'word' in r


def test_predict_does_not_mutate_input(engine, rng):
    audio = (rng.standard_normal(engine.audio_samples_needed) * 100).astype(np.float32)
    before = audio.copy()
    engine.predict(audio)
    assert np.array_equal(audio, before)


def test_unloaded_engine_predict_returns_none():
    eng = WakeWordEngine()
    assert not eng.is_loaded()
    assert eng.predict(np.zeros(16000, dtype=np.float32)) is None


def test_multi_model_order(tmp_path):
    """Legacy multi_model JSON: keyword order must be preserved."""
    src_info, src_mel = get_demo_model()
    model_file = 'heyjarvis_r1.onnx'

    cfg = {
        'model_type': 'tcn',
        'mel_time': 98,
        'multi_model': True,
        'models': [
            {'wake_word': 'Alpha', 'model_file': model_file, 'cons_frames': 2},
            {'wake_word': 'Beta', 'model_file': model_file, 'cons_frames': 3},
        ],
    }
    d = tmp_path / 'bundle'
    d.mkdir()
    (d / 'model_info.json').write_text(json.dumps(cfg), encoding='utf-8')
    import shutil
    shutil.copy(src_mel, d / 'melspectrogram.onnx')
    shutil.copy(os.path.join(os.path.dirname(src_info), model_file), d / model_file)

    eng = WakeWordEngine()
    eng.load(str(d / 'model_info.json'), str(d / 'melspectrogram.onnx'))
    assert [m['name'] for m in eng.get_models()] == ['Alpha', 'Beta']
    r = eng.predict(np.zeros(eng.audio_samples_needed, dtype=np.float32))
    assert r['word'] in ('Alpha', 'Beta')
    assert set(r['all'].keys()) == {'Alpha', 'Beta'}


def test_zip_bundle_load(tmp_path):
    """A single ZIP containing model_info.json + model must load like a directory."""
    src_info, src_mel = get_demo_model()
    zpath = tmp_path / 'demo.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        zf.write(src_info, 'model_info.json')
        zf.write(src_mel, 'melspectrogram.onnx')
        zf.write(os.path.join(os.path.dirname(src_info), 'heyjarvis_r1.onnx'), 'heyjarvis_r1.onnx')

    eng = WakeWordEngine()
    eng.load(str(zpath), str(src_mel))
    assert eng.is_loaded()
    assert eng.get_models()[0]['name'] == 'Hey Jarvis'


def test_load_without_mel_auto_resolves(tmp_path):
    """Keyword-model-only bundles work: mel falls back to the package copy."""
    import shutil
    src_info, _ = get_demo_model()
    model_file = 'heyjarvis_r1.onnx'
    d = tmp_path / 'model_only'
    d.mkdir()
    shutil.copy(os.path.join(os.path.dirname(src_info), model_file), d / model_file)
    with open(src_info, encoding='utf-8') as f:
        cfg = json.load(f)
    (d / 'model_info.json').write_text(json.dumps(cfg), encoding='utf-8')

    eng = WakeWordEngine()
    eng.load(str(d / 'model_info.json'))          # no mel_path at all
    assert eng.is_loaded()
    assert eng.get_models()[0]['name'] == 'Hey Jarvis'
    r = eng.predict(np.zeros(eng.audio_samples_needed, dtype=np.float32))
    assert r is not None


def test_load_zip_without_mel_auto_resolves(tmp_path):
    """ZIP with only model_info.json + model: mel comes from the package."""
    import shutil
    import zipfile
    src_info, src_mel = get_demo_model()
    zpath = tmp_path / 'model_only.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        zf.write(src_info, 'model_info.json')
        zf.write(os.path.join(os.path.dirname(src_info), 'heyjarvis_r1.onnx'), 'heyjarvis_r1.onnx')

    eng = WakeWordEngine()
    eng.load(str(zpath))                          # no mel_path at all
    assert eng.is_loaded()


def test_load_zip_with_embedded_mel(tmp_path):
    """ZIP containing its own melspectrogram.onnx uses it (bundle-local wins)."""
    import zipfile
    src_info, src_mel = get_demo_model()
    zpath = tmp_path / 'full.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        zf.write(src_info, 'model_info.json')
        zf.write(src_mel, 'melspectrogram.onnx')
        zf.write(os.path.join(os.path.dirname(src_info), 'heyjarvis_r1.onnx'), 'heyjarvis_r1.onnx')

    eng = WakeWordEngine()
    eng.load(str(zpath))
    assert eng.is_loaded()
    assert eng.get_models()[0]['name'] == 'Hey Jarvis'


def test_mel_time_mismatch_fails_fast(tmp_path):
    """A wrong mel_time in the JSON must raise at load(), not at first inference."""
    src_info, src_mel = get_demo_model()
    with open(src_info, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['mel_time'] = 150
    d = tmp_path / 'bad'
    d.mkdir()
    (d / 'model_info.json').write_text(json.dumps(cfg), encoding='utf-8')
    import shutil
    shutil.copy(src_mel, d / 'melspectrogram.onnx')
    shutil.copy(os.path.join(os.path.dirname(src_info), 'heyjarvis_r1.onnx'), d / 'heyjarvis_r1.onnx')

    eng = WakeWordEngine()
    with pytest.raises(ValueError, match='mel_time'):
        eng.load(str(d / 'model_info.json'), str(d / 'melspectrogram.onnx'))

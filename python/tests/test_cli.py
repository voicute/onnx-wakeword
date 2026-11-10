"""CLI tests — the ``onnx-wakeword-demo`` entry point in WAV mode (no mic, no pyaudio)."""
import os
import wave

import numpy as np

from onnx_wakeword import cli, get_demo_model


def _make_wav(path, seconds=2.0, seed=7):
    """Synthetic 16 kHz mono 16-bit WAV (noise — won't trigger, that's the point)."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(16000 * seconds)) * 200).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.tobytes())
    return path


def test_wav_mode_runs_on_noise(tmp_path, capsys):
    wav = _make_wav(tmp_path / 'noise.wav')
    rc = cli.main(['--wav', str(wav)])
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Done.' in out
    assert '0 detection(s)' in out


def test_wav_mode_with_custom_model(tmp_path, capsys):
    wav = _make_wav(tmp_path / 'noise2.wav', seed=9)
    info, mel = get_demo_model()
    rc = cli.main(['--wav', str(wav), '--info', info, '--mel', mel, '--all'])
    assert rc == 0
    assert 'Done.' in capsys.readouterr().out


def test_wav_mode_accepts_non_16k(tmp_path, capsys):
    """A 8 kHz file gets resampled, not rejected."""
    rng = np.random.default_rng(3)
    audio = (rng.standard_normal(8000) * 200).astype(np.int16)
    path = tmp_path / 'low_rate.wav'
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(audio.tobytes())
    rc = cli.main(['--wav', str(path)])
    assert rc == 0
    assert 'Done.' in capsys.readouterr().out

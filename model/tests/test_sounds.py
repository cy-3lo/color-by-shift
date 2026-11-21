"""The synthesised sounds of the game (tools/make_sounds.py)."""

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import make_sounds  # noqa: E402


def test_pluck_has_its_pitch_and_dies_away():
    rate = 22050
    x = make_sounds.pluck(rate, f0=220.0, seconds=0.6)
    assert np.isfinite(x).all() and len(x) == round(0.6 * rate)
    spec = np.abs(np.fft.rfft(x[: rate // 4]))
    f = np.fft.rfftfreq(rate // 4, 1 / rate)
    assert abs(f[spec.argmax()] - 220.0) < 10.0  # fundamental strongest
    early, late = np.abs(x[: rate // 10]).mean(), np.abs(x[-rate // 10 :]).mean()
    assert late < 0.3 * early


def test_tada_and_wav_file(tmp_path):
    rate = 22050
    x = make_sounds.tada(rate)
    assert np.isfinite(x).all() and len(x) == round(2.2 * rate)
    make_sounds.write(tmp_path / "t.wav", x, rate)
    with wave.open(str(tmp_path / "t.wav")) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        assert w.getframerate() == rate and w.getnchannels() == 1
    assert abs(np.abs(pcm).max() / 32767 - 0.8) < 0.01

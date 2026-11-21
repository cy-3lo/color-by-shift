"""Sounds of the game, synthesised (no samples, no licence question).

    python tools/make_sounds.py [--out game/audio] [--rate 44100]

pluck.wav: a soft plucked string (Karplus-Strong: a burst of low-passed noise circulating in a
delay line one period long, averaged at each pass, so the high partials die first), on A3; the
game plays it at a pitch set by the group of the move and a loudness set by the number of
pairs that swap. tada.wav: an arpeggio C5 E5 G5 C6 of mallet tones (a few inharmonic partials,
each with its own decay), ending on the chord, with a short echo for room.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pluck(
    rate: int, f0: float = 220.0, seconds: float = 0.9, damping: float = 0.996, seed: int = 3
) -> np.ndarray:
    """Karplus-Strong string: one period of noise, low-passed for a softer attack."""
    rng = np.random.default_rng(seed)
    period = round(rate / f0)
    burst = rng.uniform(-1.0, 1.0, period)
    for _ in range(3):  # three passes of a two-point average: a darker, rounder attack
        burst = 0.5 * (burst + np.roll(burst, 1))
    n = round(seconds * rate)
    out = np.zeros(n)
    buf = burst.copy()
    for i in range(n):
        k = i % period
        out[i] = buf[k]
        buf[k] = damping * 0.5 * (buf[k] + buf[k - 1])
    fade = np.ones(n)
    tail = round(0.08 * rate)
    fade[-tail:] = np.linspace(1.0, 0.0, tail)
    return out * fade


def mallet(rate: int, f: float, seconds: float) -> np.ndarray:
    """A struck bar: partials 1, 3.9 and 9.2 times f, the higher ones shorter."""
    t = np.arange(round(seconds * rate)) / rate
    tone = (
        np.sin(2 * np.pi * f * t) * np.exp(-t / 0.9)
        + 0.25 * np.sin(2 * np.pi * 3.9 * f * t) * np.exp(-t / 0.18)
        + 0.08 * np.sin(2 * np.pi * 9.2 * f * t) * np.exp(-t / 0.05)
    )
    attack = np.minimum(1.0, t / 0.004)
    return tone * attack


def tada(rate: int) -> np.ndarray:
    notes = [(0.0, 523.25), (0.09, 659.25), (0.18, 783.99), (0.32, 1046.5)]
    chord = [(0.32, 523.25), (0.32, 659.25), (0.32, 783.99)]
    n = round(2.2 * rate)
    out = np.zeros(n)
    for start, f in notes + chord:
        k = round(start * rate)
        tone = mallet(rate, f, 2.2 - start)[: n - k]
        out[k : k + len(tone)] += tone * (0.6 if (start, f) in chord else 1.0)
    for delay, gain in ((0.11, 0.3), (0.23, 0.15)):  # a little room
        d = round(delay * rate)
        out[d:] += gain * out[:-d].copy()
    return out


def write(path: Path, x: np.ndarray, rate: int, peak: float = 0.8) -> None:
    x = x / np.abs(x).max() * peak
    pcm = np.round(x * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "game" / "audio"))
    ap.add_argument("--rate", type=int, default=44100)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "pluck.wav", pluck(a.rate), a.rate)
    write(out / "tada.wav", tada(a.rate), a.rate)
    print("wrote", out / "pluck.wav", out / "tada.wav")


if __name__ == "__main__":
    main()

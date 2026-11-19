"""Goal and failure detection.

Goal types (durations in sim time units):
  {"type": "extinct", "color": c}                         colour c has count 0
  {"type": "hold", "bands": [[lo, hi], ...], "duration": T} every fraction within its band
                                                           continuously for T
  {"type": "reach", "target": [...], "tol": e, "duration": T}  max |x - target| <= e
                                                           continuously for T
  {"type": "size", "range": [lo, hi], "duration": T}     patch size S (cells, see
                                                           observables.py) within [lo, hi]
                                                           continuously for T; with
                                                           "color": c, the patches of c only
Fractions and counts are over the free cells (terrain: holes and fixed cells excluded).
Each goal latches once met; the level is won when all goals have latched.
Failure conditions use the same types and fail the level as soon as one is met
(e.g. {"type": "extinct", "color": 1} during a hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Cond:
    spec: dict
    since: float | None = None  # time the condition became continuously true
    done: bool = False

    def instant(self, x: np.ndarray, counts: np.ndarray, obs: dict | None = None) -> bool:
        s = self.spec
        if s["type"] == "extinct":
            return counts[s["color"]] == 0
        if s["type"] == "hold":
            return all(lo <= xi <= hi for xi, (lo, hi) in zip(x, s["bands"], strict=True))
        if s["type"] == "reach":
            return float(np.max(np.abs(x - np.asarray(s["target"])))) <= s["tol"]
        if s["type"] == "size":
            if obs is None:
                return False
            v = obs["size"] if s.get("color") is None else obs["size_c"][s["color"]]
            lo, hi = s["range"]
            return lo <= v <= hi
        raise ValueError(f"unknown goal type {s['type']!r}")

    def update(self, t: float, x: np.ndarray, counts: np.ndarray, obs: dict | None) -> bool:
        if self.done:
            return True
        if not self.instant(x, counts, obs):
            self.since = None
            return False
        if self.since is None:
            self.since = t
        if t - self.since >= float(self.spec.get("duration", 0.0)):
            self.done = True
        return self.done

    def progress(self, t: float) -> float:
        """Fraction of the required duration achieved (0..1)."""
        if self.done:
            return 1.0
        if self.since is None:
            return 0.0
        d = float(self.spec.get("duration", 0.0))
        return 1.0 if d == 0 else min(1.0, (t - self.since) / d)


@dataclass
class GoalTracker:
    goals: list[dict]
    fail: list[dict] = field(default_factory=list)
    status: str = "running"  # running | won | failed

    def __post_init__(self) -> None:
        self._goals = [_Cond(g) for g in self.goals]
        self._fails = [_Cond(f) for f in self.fail]

    @property
    def needs_size(self) -> bool:
        """Whether any goal or failure condition reads the patch size."""
        return any(c.spec["type"] == "size" for c in self._goals + self._fails)

    def update(self, t: float, counts: np.ndarray, obs: dict | None = None) -> str:
        """obs: {"size": S, "size_c": per-colour S} when needs_size (observables.py)."""
        if self.status != "running":
            return self.status
        counts = np.asarray(counts)
        x = counts / max(1, counts.sum())
        if any(f.update(t, x, counts, obs) for f in self._fails):
            self.status = "failed"
            return self.status
        met = [g.update(t, x, counts, obs) for g in self._goals]
        if met and all(met):
            self.status = "won"
        return self.status

    def progress(self, t: float) -> list[float]:
        return [g.progress(t) for g in self._goals]

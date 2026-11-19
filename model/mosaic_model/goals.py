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
  {"type": "contact", "colors": [a, b]}                   some world edge joins a cell of
                                                           colour a and one of colour b
                                                           (e.g. red reaches a green target);
                                                           with "all": true, every cell of
                                                           colour b must have been touched
                                                           (touches are remembered)
  {"type": "zone", "zone": "G", "color": c, "range": [lo, hi], "duration": T}
                                                           fraction of colour c among the
                                                           free cells of zone G (map zones)
                                                           within [lo, hi] for T; [0, 0] =
                                                           c extinct in the zone
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
    touched: set = field(default_factory=set)  # contact "all": b cells touched so far
    n_targets: int = 0

    def instant(self, x: np.ndarray, counts: np.ndarray, obs: dict | None = None) -> bool:
        s = self.spec
        if s["type"] == "extinct":
            return counts[s["color"]] == 0
        if s["type"] == "hold":
            return all(lo <= xi <= hi for xi, (lo, hi) in zip(x, s["bands"], strict=True))
        if s["type"] == "reach":
            return float(np.max(np.abs(x - np.asarray(s["target"])))) <= s["tol"]
        if s["type"] == "zone":
            if obs is None:
                return False
            v = zone_fraction(obs, s["zone"], s["color"])
            lo, hi = s["range"]
            return lo <= v <= hi
        if s["type"] == "contact":
            if obs is None:
                return False
            a, b = s["colors"]
            col, e = obs["col"], obs["edges"]
            ca, cb = col[e[:, 0]], col[e[:, 1]]
            if not s.get("all"):
                return bool(np.any(((ca == a) & (cb == b)) | ((ca == b) & (cb == a))))
            hit = np.concatenate([e[(ca == a) & (cb == b), 1], e[(ca == b) & (cb == a), 0]])
            self.touched.update(hit.tolist())
            self.n_targets = max(self.n_targets, int(np.sum(col == b)))
            return len(self.touched) >= max(1, self.n_targets)
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
        if self.spec.get("all") and self.n_targets:
            return len(self.touched) / self.n_targets
        if self.since is None:
            return 0.0
        d = float(self.spec.get("duration", 0.0))
        return 1.0 if d == 0 else min(1.0, (t - self.since) / d)


def zone_fraction(obs: dict, zone: str, colour: int) -> float:
    """Fraction of colour among the free cells of a zone (obs from observables.observe,
    with "zones" filled in by the tracker)."""
    cells = obs["zones"][zone]
    cells = cells[obs["free"][cells]]
    if len(cells) == 0:
        return 0.0
    return float(np.mean(obs["col"][cells] == colour))


@dataclass
class GoalTracker:
    goals: list[dict]
    fail: list[dict] = field(default_factory=list)
    status: str = "running"  # running | won | failed
    zones: dict = field(default_factory=dict)  # zone name -> cell indices

    def __post_init__(self) -> None:
        self._goals = [_Cond(g) for g in self.goals]
        self._fails = [_Cond(f) for f in self.fail]

    @property
    def needs_size(self) -> bool:
        """Whether any goal or failure condition reads the patch size."""
        return any(c.spec["type"] == "size" for c in self._goals + self._fails)

    @property
    def needs_obs(self) -> bool:
        """Whether update() needs the obs dict (spatial goals)."""
        spatial = ("size", "contact", "zone")
        return any(c.spec["type"] in spatial for c in self._goals + self._fails)

    def update(self, t: float, counts: np.ndarray, obs: dict | None = None) -> str:
        """obs (observables.observe): {"size": S, "size_c": per-colour S, "col": colours,
        "edges": world edges}, needed by size and contact goals."""
        if self.status != "running":
            return self.status
        counts = np.asarray(counts)
        x = counts / max(1, counts.sum())
        if obs is not None and "zones" not in obs:
            obs = {**obs, "zones": self.zones}
        if any(f.update(t, x, counts, obs) for f in self._fails):
            self.status = "failed"
            return self.status
        met = [g.update(t, x, counts, obs) for g in self._goals]
        if met and all(met):
            self.status = "won"
        return self.status

    def progress(self, t: float) -> list[float]:
        return [g.progress(t) for g in self._goals]

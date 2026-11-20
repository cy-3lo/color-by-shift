"""Reversible mosaic: directional swaps applied to every pair at once.

A reaction is a swap of two colours across the edges of one direction, named after the
side the two cells share on a pointy-top hex grid:

    "|"   horizontal neighbours        (axial step (1, 0))
    "/"   upper-left / lower-right     (axial step (0, 1))
    "\\"  lower-left / upper-right     (axial step (1, -1))

Along a direction every cell has two neighbours, so the pairs of a direction split into two
alternations of disjoint pairs (the first cell's coordinate along the direction, mod 2).
Applying a reaction swaps every pair of alternation 0 that holds its two colours, all at
once, then every such pair of alternation 1. Whether a pair swaps depends only on the two
colours it holds, which the swap does not change, so each half is its own inverse, and the
reaction is undone by doing alternation 1 first, then 0.

A reaction may also be limited to one alternation (Swap.alt = 0 or 1; levels with
"single": true use alternation 0 everywhere). Each cell then has one partner per direction
and moves by one place at most; such a reaction is its own inverse.

A program is an ordered list of reactions. Time t counts reactions: a forward step applies
program[t mod n] and adds 1 to t; a backward step subtracts 1 from t and undoes
program[t mod n]. No random numbers: the dynamics is deterministic and exactly reversible,
whatever the program. Undoing a scramble needs the same reactions in the reverse order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .graph import Graph, hex_rect

DIRECTIONS = {"|": (1, 0), "/": (0, 1), "\\": (1, -1)}
DIRECTION_NAMES = list(DIRECTIONS)


@dataclass(frozen=True)
class Swap:
    direction: str
    a: int
    b: int
    alt: int | None = None  # None: both alternations (0 then 1); 0 or 1: that one only

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.alt not in (None, 0, 1):
            raise ValueError("alt must be None, 0 or 1")


class ShiftGrid:
    """The pairs of a graph, by direction and alternation."""

    def __init__(self, g: Graph):
        self.graph = g
        e = np.asarray(g.edges)
        ax = np.asarray(g.axial)
        d = ax[e[:, 1]] - ax[e[:, 0]]
        flip = (d[:, 0] < 0) | ((d[:, 0] == 0) & (d[:, 1] < 0))
        p = np.where(flip, e[:, 1], e[:, 0])
        q = np.where(flip, e[:, 0], e[:, 1])
        d = ax[q] - ax[p]
        self.pairs: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        seen = np.zeros(len(e), bool)
        for name, step in DIRECTIONS.items():
            m = (d == step).all(axis=1)
            seen |= m
            along = ax[p][:, 1] if step[0] == 0 else ax[p][:, 0]
            for alt in (0, 1):
                k = m & (along % 2 == alt)
                self.pairs[(name, alt)] = (p[k], q[k])
        if not seen.all():
            raise ValueError("ShiftGrid needs an open grid (no glued sides)")

    def offset(self, direction: str) -> np.ndarray:
        """Centre-to-centre vector of a pair of this direction (unit circumradius, y down)."""
        p, q = self.pairs[(direction, 0)]
        return np.asarray(self.graph.centers)[q[0]] - np.asarray(self.graph.centers)[p[0]]

    def half(self, col: np.ndarray, s: Swap, alt: int) -> tuple[np.ndarray, int]:
        """Swap every pair of one alternation that holds the colours of s: (new, count)."""
        p, q = self.pairs[(s.direction, alt)]
        cp, cq = col[p], col[q]
        m = ((cp == s.a) & (cq == s.b)) | ((cp == s.b) & (cq == s.a))
        if s.a == s.b:
            m[:] = False
        out = col.copy()
        out[p[m]] = cq[m]
        out[q[m]] = cp[m]
        return out, int(m.sum())

    def apply(self, col: np.ndarray, s: Swap, inverse: bool = False) -> np.ndarray:
        alts = (s.alt,) if s.alt is not None else (1, 0) if inverse else (0, 1)
        for alt in alts:
            col, _ = self.half(col, s, alt)
        return col

    def step(self, col: np.ndarray, program: list[Swap], t: int, forward: bool = True):
        """One reaction forward or backward: (new colours, new t). Empty program: no change."""
        if not program:
            return col, t
        if forward:
            return self.apply(col, program[t % len(program)]), t + 1
        t -= 1
        return self.apply(col, program[t % len(program)], inverse=True), t

    def run(self, col: np.ndarray, program: list[Swap], steps: int, t: int = 0):
        """`steps` reactions forward (negative: backward) from time t: (colours, t)."""
        for _ in range(abs(steps)):
            col, t = self.step(col, program, t, steps > 0)
        return col, t


# ---- pictures and levels ---------------------------------------------------------------------
def flag(kind: str, g: Graph, n: int) -> np.ndarray:
    """Simple flag-like pictures with n colours (generic patterns, no particular flag):
    bands_v, bands_h (n equal bands), disc (colour 1 on 0), cross (colour 1 on 0, off-centre
    like a Nordic cross), diagonal (two halves; with n = 3 a band along the diagonal)."""
    x, y = np.asarray(g.centers)[:, 0], np.asarray(g.centers)[:, 1]
    u = (x - x.min()) / (x.max() - x.min())
    v = (y - y.min()) / (y.max() - y.min())
    if kind == "bands_v":
        return np.minimum((u * n).astype(int), n - 1)
    if kind == "bands_h":
        return np.minimum((v * n).astype(int), n - 1)
    if kind == "disc":
        r = np.hypot(x - x.mean(), y - y.mean())
        return (r < 0.3 * (y.max() - y.min())).astype(int) * min(1, n - 1)
    if kind == "cross":
        return ((np.abs(u - 0.36) < 0.08) | (np.abs(v - 0.5) < 0.12)).astype(int)
    if kind == "diagonal":
        s = u - v
        if n >= 3:
            return np.where(s > 0.12, 0, np.where(s < -0.12, 2, 1))
        return (s < 0).astype(int)
    raise ValueError(f"unknown flag {kind!r}")


@dataclass
class ShiftLevel:
    id: str
    graph: Graph
    colours: list[str]  # "#rrggbb" per colour
    target: np.ndarray  # the picture to recover
    program: list[Swap]  # hidden program that scrambled it
    cycles: int  # times the whole program was applied
    slots: int  # reactions offered to the player at the start
    alt: int | None = None  # alternation of every reaction of the level (None: both)
    title: str = ""
    grid: ShiftGrid = field(init=False)

    def __post_init__(self) -> None:
        self.grid = ShiftGrid(self.graph)

    @property
    def n(self) -> int:
        return len(self.colours)

    def scrambled(self) -> np.ndarray:
        return self.grid.run(self.target, self.program, self.cycles * len(self.program))[0]


def load_shift_level(path: str | Path) -> ShiftLevel:
    """Level JSON: {"id", "title", "grid": {"w", "h"}, "colours": ["#rrggbb", ...],
    "picture": {"flag": kind} or {"rows": ["0011...", ...]} (odd-r rows of colour digits),
    "program": [{"dir": "|", "pair": [0, 1]}, ...], "cycles": T, optional "slots", optional
    "single": true (every reaction, the player's too, uses alternation 0 only)}."""
    d = json.loads(Path(path).read_text())
    w, h = int(d["grid"]["w"]), int(d["grid"]["h"])
    g = hex_rect(w, h)
    n = len(d["colours"])
    pic = d["picture"]
    if "rows" in pic:
        rows = pic["rows"]
        if len(rows) != h or any(len(r) != w for r in rows):
            raise ValueError(f"{path}: picture rows must be {h} strings of {w} digits")
        target = np.array([int(c) for r in rows for c in r])
    else:
        target = flag(pic["flag"], g, n)
    alt = 0 if d.get("single") else None
    prog = [Swap(r["dir"], int(r["pair"][0]), int(r["pair"][1]), alt) for r in d["program"]]
    if target.max() >= n or any(max(s.a, s.b) >= n for s in prog):
        raise ValueError(f"{path}: colour outside the palette")
    return ShiftLevel(
        d["id"],
        g,
        list(d["colours"]),
        target.astype(np.int64),
        prog,
        int(d.get("cycles", 1)),
        int(d.get("slots", len(prog))),
        alt,
        d.get("title", ""),
    )

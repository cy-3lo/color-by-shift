"""Cell graphs: cells, edges (pairs of neighbouring cells) and vertices (triples of cells
meeting at a point, listed clockwise as seen on screen with the y axis pointing down).

All worlds are made of pointy-top hexagons. Cells carry axial coordinates (q, r); the centre
of a hex of unit circumradius is (sqrt3 * (q + r / 2), 1.5 * r). A world is a set of cells
plus, if periodic, the translations that glue its sides together.

World shapes (see make_graph):
  hex      n x n "odd-r" offset block (a rhombus-like rectangle), periodic or open.
           Cell index r * n + c; matches the Godot benchmark grid (game/sim/hex_grid.gd).
  hexagon  regular hexagon of side R (3R^2 - 3R + 1 cells), periodic (opposite sides glued,
           a torus with sixfold symmetry) or open.

Each cell owns at most three edges (E, SW, SE) and two vertices (its lower-right and bottom
corners). On a torus E = 3N and V = 2N; with open sides, elements that would leave the world
are dropped (cells on the border have fewer neighbours).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SQRT3 = 3.0**0.5

# axial directions used to build elements
E_DIR, SE_DIR, SW_DIR = (1, 0), (0, 1), (-1, 1)


@dataclass(frozen=True)
class Graph:
    """Cell graph with full adjacency.

    Attributes:
        n_cells: number of cells.
        edges: (E, 2) int array of neighbouring cell pairs.
        vertices: (V, 3) int array of cell triples, clockwise on screen (y down).
        axial: (n_cells, 2) int axial coordinates of the drawn position of each cell.
        centers: (n_cells, 2) cell centres for a hex of unit circumradius (y down).
        shifts: (k, 2) pixel translations that glue the world (empty if open).
        kind: "hex" or "hexagon"; periodic: whether sides are glued; n: size parameter.
    """

    n_cells: int
    edges: np.ndarray
    vertices: np.ndarray
    axial: np.ndarray
    centers: np.ndarray
    shifts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    shifts_axial: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    kind: str = "hex"
    periodic: bool = True
    n: int = 0

    def neighbours(self) -> list[list[int]]:
        nb: list[list[int]] = [[] for _ in range(self.n_cells)]
        for a, b in self.edges.tolist():
            nb[a].append(b)
            nb[b].append(a)
        return nb

    def displacement(self, a: int, b: int) -> np.ndarray:
        """Vector from centre a to centre b (minimal image if periodic)."""
        d = self.centers[b] - self.centers[a]
        if len(self.shifts) == 0:
            return d
        s1, s2 = self.shifts
        cands = [d - i * s1 - j * s2 for i in (-1, 0, 1) for j in (-1, 0, 1)]
        return min(cands, key=lambda c: float(c @ c))

    def label(self) -> str:
        return f"{self.kind}{self.n}{'' if self.periodic else '-open'}"


def _axial_xy(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.stack([SQRT3 * (q + 0.5 * r), 1.5 * r], axis=-1)


def _build(
    cells: list[tuple[int, int]], shifts: list[tuple[int, int]], kind: str, n: int, periodic: bool
) -> Graph:
    index = {c: i for i, c in enumerate(cells)}
    s_all = [(0, 0)]
    if periodic:
        s1, s2 = shifts
        s_all = [
            (i * s1[0] + j * s2[0], i * s1[1] + j * s2[1]) for i in (-1, 0, 1) for j in (-1, 0, 1)
        ]

    def find(q: int, r: int) -> int:
        for dq, dr in s_all:
            k = index.get((q + dq, r + dr))
            if k is not None:
                return k
        return -1

    edges, verts = [], []
    for i, (q, r) in enumerate(cells):
        e = find(q + E_DIR[0], r + E_DIR[1])
        se = find(q + SE_DIR[0], r + SE_DIR[1])
        sw = find(q + SW_DIR[0], r + SW_DIR[1])
        # same order as the original hex torus: E, SW, SE
        for j in (e, sw, se):
            if j >= 0:
                edges.append((i, j))
        # lower-right corner (i, E, SE) and bottom corner (i, SE, SW), both clockwise
        if e >= 0 and se >= 0:
            verts.append((i, e, se))
        if se >= 0 and sw >= 0:
            verts.append((i, se, sw))
    ax = np.array(cells, dtype=np.int64)
    sh = (
        _axial_xy(np.array([s[0] for s in shifts], float), np.array([s[1] for s in shifts], float))
        if periodic
        else np.zeros((0, 2))
    )
    return Graph(
        n_cells=len(cells),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        vertices=np.array(verts, dtype=np.int64).reshape(-1, 3),
        axial=ax,
        centers=_axial_xy(ax[:, 0].astype(float), ax[:, 1].astype(float)),
        shifts=sh,
        shifts_axial=np.array(shifts if periodic else [], dtype=np.int64).reshape(-1, 2),
        kind=kind,
        periodic=periodic,
        n=n,
    )


def hex_torus(n: int, periodic: bool = True) -> Graph:
    """n x n odd-r offset block of hexes (n even, n >= 4), periodic unless periodic=False."""
    if n < 4 or n % 2:
        raise ValueError("hex_torus: n must be even and >= 4")
    cells = [(c - (r - (r & 1)) // 2, r) for r in range(n) for c in range(n)]
    # gluing: column + n is the axial shift (n, 0); row + n (n even) is (-n/2, n)
    return _build(cells, [(n, 0), (-n // 2, n)], "hex", n, periodic)


def hex_rect(w: int, h: int, periodic: bool = False) -> Graph:
    """w x h odd-r offset block of hexes (cell index r * w + c), as drawn from an ASCII map
    (odd rows shifted half a cell to the right). Periodic needs h even."""
    if w < 2 or h < 2 or (periodic and h % 2):
        raise ValueError("hex_rect: w, h >= 2 (h even if periodic)")
    cells = [(c - (r - (r & 1)) // 2, r) for r in range(h) for c in range(w)]
    return _build(cells, [(w, 0), (-h // 2, h)], "hex", w, periodic)


def hex_hexagon(side: int, periodic: bool = True) -> Graph:
    """Regular hexagon of `side` cells per side (3 side^2 - 3 side + 1 cells).

    Periodic: opposite sides are glued by the translations (2N+1, -N) and (N, N+1),
    N = side - 1 (a hexagonal torus; each of the six neighbouring copies sits at distance
    2N + 1)."""
    if side < 3:
        raise ValueError("hex_hexagon: side must be >= 3")
    m = side - 1
    cells = [(q, r) for r in range(-m, m + 1) for q in range(-m, m + 1) if abs(q + r) <= m]
    return _build(cells, [(2 * m + 1, -m), (m, m + 1)], "hexagon", side, periodic)


def make_graph(spec: dict) -> Graph:
    """Build a graph from a level's "grid" entry, e.g. {"type": "hex", "n": 64},
    {"type": "hex", "w": 48, "h": 34, "periodic": false},
    {"type": "hexagon", "side": 37}, optionally "periodic": false."""
    periodic = bool(spec.get("periodic", True))
    if spec.get("type") == "hex" and "w" in spec:
        return hex_rect(int(spec["w"]), int(spec["h"]), periodic)
    if spec.get("type") == "hex":
        return hex_torus(int(spec["n"]), periodic)
    if spec.get("type") == "hexagon":
        return hex_hexagon(int(spec["side"]), periodic)
    raise ValueError(f"unsupported grid type: {spec.get('type')!r}")

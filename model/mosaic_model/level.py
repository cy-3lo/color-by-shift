"""JSON level loader (format shared with Godot, see CLAUDE.md, Levels).

Optional terrain, in axial cell coordinates (q, r) of the level's world:
    "terrain": {"holes": [[q, r], ...], "fixed": [[q, r, colour], ...]}
Holes are walls (not part of the world); fixed cells keep their colour forever.

Drawn start (instead of random fractions): an ASCII map, one string per row of an odd-r hex
block (odd rows drawn half a cell to the right), and a legend from character to cell:
    "map": {"rows": ["#####", "#R.y#", ...],
            "legend": {"#": "wall", ".": 0, "y": 4, "R": [1, "fixed"]}}
A legend value "random" draws the cell's colour from start "fractions" (seed start "seed");
{"random": [f0, f1, ...]} uses its own fractions for that character.
Zones (for zone goals): "map": {..., "zones": {"G": "g", "M": {"rect": [r0, r1, c0, c1]}}}:
a string names the cells drawn with any of its characters, a rect the cells in rows r0..r1
and columns c0..c1 (inclusive).
The grid is then {"type": "hex", "w": len(row), "h": len(rows), "periodic": false} unless
the level gives one of that size. The map replaces "terrain" and start "fractions".
Optional "colours" (hex strings) and "names" override the viewer palette for this level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .goals import GoalTracker
from .graph import Graph, make_graph
from .reactions import ReactionSet, reactions_from_json
from .sim import Sim


@dataclass
class Level:
    id: str
    grid: dict
    palette: int
    reactions: list[dict]
    locked: list[list]
    start: dict
    goals: list[dict]
    fail: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    terrain: dict = field(default_factory=dict)
    map: dict = field(default_factory=dict)
    colours: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    title: str = ""
    path: Path | None = None

    def reaction_set(self) -> ReactionSet:
        return ReactionSet(
            reactions_from_json(self.reactions),
            self.palette,
            {(int(b[0]), str(b[1]), int(b[2])) for b in self.locked},
        )

    def graph(self) -> Graph:
        return make_graph(self.grid)

    def __post_init__(self) -> None:
        if self.map and not self.grid:
            rows = self.map["rows"]
            self.grid = {"type": "hex", "w": len(rows[0]), "h": len(rows), "periodic": False}

    def tracker(self, g: Graph | None = None) -> GoalTracker:
        zones = map_zones(g if g is not None else self.graph(), self.map)
        return GoalTracker(self.goals, self.fail, zones=zones)


def load_level(path: str | Path) -> Level:
    path = Path(path)
    d = json.loads(path.read_text())
    return Level(
        id=d["id"],
        grid=d.get("grid", {}),
        palette=int(d["palette"]),
        reactions=d["reactions"],
        locked=d.get("locked", []),
        start=d.get("start", {}),
        goals=d.get("goals", []),
        fail=d.get("fail", []),
        meta=d.get("meta", {}),
        terrain=d.get("terrain", {}),
        map=d.get("map", {}),
        colours=d.get("colours", []),
        names=d.get("names", []),
        title=d.get("title", ""),
        path=path,
    )


@dataclass
class Terrain:
    """Per-cell masks: hole, fixed, and the colour of each fixed cell (0 elsewhere)."""

    hole: np.ndarray
    fixed: np.ndarray
    colour: np.ndarray

    @classmethod
    def empty(cls, n_cells: int) -> Terrain:
        z = np.zeros(n_cells, bool)
        return cls(z, z.copy(), np.zeros(n_cells, np.int64))

    def apply(self, colours: np.ndarray) -> np.ndarray:
        """Colours with the fixed cells set to their colour (holes get colour 0)."""
        c = np.array(colours, dtype=np.int64)
        c[self.fixed] = self.colour[self.fixed]
        c[self.hole] = 0
        return c


def axial_index(g: Graph) -> dict[tuple[int, int], int]:
    return {(int(q), int(r)): i for i, (q, r) in enumerate(g.axial.tolist())}


def terrain_from_json(g: Graph, spec: dict | None, palette: int) -> Terrain:
    """Terrain masks from a level's "terrain" entry; cells outside the world are an error."""
    t = Terrain.empty(g.n_cells)
    if not spec:
        return t
    idx = axial_index(g)

    def cell(q: int, r: int) -> int:
        try:
            return idx[(int(q), int(r))]
        except KeyError:
            raise ValueError(f"terrain cell ({q}, {r}) is not in the world") from None

    for q, r in spec.get("holes", []):
        t.hole[cell(q, r)] = True
    for q, r, c in spec.get("fixed", []):
        if not 0 <= int(c) < palette:
            raise ValueError(f"fixed colour {c} outside palette {palette}")
        i = cell(q, r)
        t.fixed[i] = True
        t.colour[i] = int(c)
    t.fixed &= ~t.hole
    return t


def terrain_to_json(g: Graph, hole: np.ndarray, fixed: np.ndarray, colours) -> dict:
    """Inverse of terrain_from_json (empty dict if there is no terrain)."""
    ax = g.axial.tolist()
    col = np.asarray(colours)
    out: dict = {}
    if np.any(hole):
        out["holes"] = [ax[i] for i in np.flatnonzero(hole)]
    fx = np.asarray(fixed) & ~np.asarray(hole)
    if np.any(fx):
        out["fixed"] = [[*ax[i], int(col[i])] for i in np.flatnonzero(fx)]
    return out


def map_start(
    g: Graph, spec: dict, palette: int, start: dict | None = None
) -> tuple[np.ndarray, Terrain, dict]:
    """(colours, terrain, tags) from an ASCII map. tags maps every legend character to the
    indices of its cells (e.g. room labels for bots and tools). Cells whose legend value is
    "random" get colours drawn from start["fractions"] with start["seed"]."""
    rows = spec["rows"]
    h, w = len(rows), len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("map rows must all have the same length")
    if g.kind != "hex" or g.n_cells != w * h or g.n != w:
        raise ValueError(f"map is {w} x {h} but the grid is {g.label()} ({g.n_cells} cells)")
    t = Terrain.empty(g.n_cells)
    col = np.zeros(g.n_cells, np.int64)
    tags: dict[str, list[int]] = {}
    rnd: list[int] = []
    own: dict[str, list[int]] = {}
    legend = spec["legend"]
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            i = r * w + c
            tags.setdefault(ch, []).append(i)
            try:
                v = legend[ch]
            except KeyError:
                raise ValueError(f"map character {ch!r} not in the legend") from None
            if v == "wall":
                t.hole[i] = True
                continue
            if v == "random":
                rnd.append(i)
                continue
            if isinstance(v, dict):
                own.setdefault(ch, []).append(i)
                continue
            fixed = isinstance(v, list)
            k = int(v[0] if fixed else v)
            if not 0 <= k < palette:
                raise ValueError(f"map colour {k} outside palette {palette}")
            col[i] = k
            if fixed:
                t.fixed[i] = True
                t.colour[i] = k
    if rnd:
        if not start or "fractions" not in start:
            raise ValueError('map uses "random" cells but start has no "fractions"')
        col[rnd] = initial_colours(len(rnd), start["fractions"], int(start.get("seed", 0)))
    seed0 = int((start or {}).get("seed", 0))
    for k, (ch, cells) in enumerate(sorted(own.items())):
        col[cells] = initial_colours(len(cells), legend[ch]["random"], seed0 + 7919 * (k + 1))
    return col, t, {k: np.array(v) for k, v in tags.items()}


def map_zones(g: Graph, spec: dict) -> dict[str, np.ndarray]:
    """Zone name -> cell indices, from "map": {"zones": {"G": "g"}} (empty if none)."""
    if not spec or "zones" not in spec:
        return {}
    w = len(spec["rows"][0])
    out: dict[str, np.ndarray] = {}
    h = len(spec["rows"])
    for name, z in spec["zones"].items():
        if isinstance(z, dict):
            r0, r1, c0, c1 = z["rect"]
            sel = [r * w + c for r in range(max(0, r0), min(h, r1 + 1)) for c in range(c0, c1 + 1)]
            sel = [i for i in sel if i % w <= min(c1, w - 1)]
        else:
            sel = [
                r * w + c
                for r, line in enumerate(spec["rows"])
                for c, ch in enumerate(line)
                if ch in z
            ]
        out[name] = np.array(sel, dtype=np.int64)
    return out


def initial_colours(n_cells: int, fractions: list[float], seed: int) -> np.ndarray:
    """Independent random colours with the given probabilities."""
    rng = np.random.default_rng(seed)
    p = np.asarray(fractions, dtype=float)
    return rng.choice(len(p), size=n_cells, p=p / p.sum())


def make_sim(
    level: Level,
    seed: int | None = None,
    shuffled: bool = False,
    burn_in: bool = True,
    graph: Graph | None = None,
) -> Sim:
    """Sim at the level's start: random colours, then burn_in_sweeps of the initial
    arrangement. Time and firing counters are reset after the burn-in."""
    seed = int(level.start.get("seed", 0)) if seed is None else seed
    g = graph if graph is not None else level.graph()
    if level.map:
        st = dict(level.start)
        st["seed"] = seed
        cols, ter, _ = map_start(g, level.map, level.palette, st)
    else:
        ter = terrain_from_json(g, level.terrain, level.palette)
        cols = ter.apply(initial_colours(g.n_cells, level.start["fractions"], seed))
    sim = Sim(
        g,
        level.reaction_set(),
        cols,
        seed=seed + 1,
        shuffled=shuffled,
        holes=ter.hole,
        fixed=ter.fixed,
    )
    if burn_in:
        sim.sweep(int(level.start.get("burn_in_sweeps", 0)))
    sim.t = 0.0
    sim.sweeps = 0
    sim.fires = [0] * len(sim.fires)
    return sim

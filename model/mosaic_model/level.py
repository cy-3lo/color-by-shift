"""JSON level loader (format shared with Godot, see CLAUDE.md, Levels).

Optional terrain, in axial cell coordinates (q, r) of the level's world:
    "terrain": {"holes": [[q, r], ...], "fixed": [[q, r, colour], ...]}
Holes are walls (not part of the world); fixed cells keep their colour forever.
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

    def tracker(self) -> GoalTracker:
        return GoalTracker(self.goals, self.fail)


def load_level(path: str | Path) -> Level:
    path = Path(path)
    d = json.loads(path.read_text())
    return Level(
        id=d["id"],
        grid=d["grid"],
        palette=int(d["palette"]),
        reactions=d["reactions"],
        locked=d.get("locked", []),
        start=d["start"],
        goals=d.get("goals", []),
        fail=d.get("fail", []),
        meta=d.get("meta", {}),
        terrain=d.get("terrain", {}),
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

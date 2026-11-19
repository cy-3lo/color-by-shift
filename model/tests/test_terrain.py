"""Terrain (holes, fixed cells), patch sizes and the size goal."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_model import _random_set

from mosaic_model import (
    GoalTracker,
    Reaction,
    ReactionSet,
    Sim,
    hex_torus,
    initial_colours,
    integrate,
    load_level,
    make_sim,
    patch_sizes,
    rhs,
    run_timeline,
    sim_patch_sizes,
)
from mosaic_model.level import terrain_from_json, terrain_to_json
from mosaic_model.observables import interface_density


def _mask(n: int, idx) -> np.ndarray:
    m = np.zeros(n, bool)
    m[list(idx)] = True
    return m


# ---- holes ---------------------------------------------------------------------------------
def test_holes_remove_their_elements() -> None:
    g = hex_torus(16)
    rs = ReactionSet([Reaction("vertex", [0, 1, 2], [1, 2, 0], 1.0)], 3)
    i, j = 5 * 16 + 5, 5 * 16 + 6  # two neighbours in one row
    sim = Sim(g, rs, np.zeros(g.n_cells, int), seed=0, holes=_mask(g.n_cells, [i]))
    assert len(sim.edge_arr) == 3 * g.n_cells - 6
    assert len(sim.vert_arr) == 2 * g.n_cells - 6
    assert not np.isin(sim.edge_arr, [i]).any() and not np.isin(sim.vert_arr, [i]).any()
    sim.set_terrain(holes=_mask(g.n_cells, [i, j]))
    assert len(sim.edge_arr) == 3 * g.n_cells - 11  # the shared edge counted once
    assert len(sim.vert_arr) == 2 * g.n_cells - 10  # two vertices shared
    assert sim.n_elig == len(sim.vert_arr)
    assert len(sim.free_idx) == g.n_cells - 2


# ---- fixed cells ---------------------------------------------------------------------------
@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_fixed_cells_and_holes_never_change(seed: int, shuffled: bool) -> None:
    rng = np.random.default_rng(seed)
    rs = _random_set(rng, 3)
    g = hex_torus(16)
    n = g.n_cells
    fixed = rng.random(n) < 0.2
    holes = rng.random(n) < 0.1
    cols = rng.integers(0, 3, n)
    sim = Sim(g, rs, cols, seed=seed, shuffled=shuffled, holes=holes, fixed=fixed)
    keep = sim.colours.copy()
    sim.sweep(30)
    still = fixed | holes
    assert (sim.colours[still] == keep[still]).all()
    assert len(sim.free_idx) == int((~fixed & ~holes).sum())


def test_exchanges_stay_conservative_next_to_fixed_cells() -> None:
    """A|B -> B|A and a vertex rotation with fixed cells: rejected whole, so the free counts
    and the fixed colours are both conserved."""
    rs = ReactionSet(
        [
            Reaction("edge", [0, 1], [1, 0], 1.0),
            Reaction("edge", [1, 2], [2, 1], 0.7),
            Reaction("vertex", [0, 1, 2], [2, 0, 1], 1.0, chiral=False),
        ],
        3,
    )
    g = hex_torus(32)
    rng = np.random.default_rng(1)
    sim = Sim(
        g,
        rs,
        rng.integers(0, 3, g.n_cells),
        seed=2,
        fixed=rng.random(g.n_cells) < 0.3,
        holes=rng.random(g.n_cells) < 0.05,
    )
    c0 = sim.counts().copy()
    f0 = sim.fixed_fractions().copy()
    for _ in range(10):
        sim.sweep()
        assert (sim.counts() == c0).all()
    assert np.allclose(sim.fixed_fractions(), f0) and sum(sim.fires) > 0


def test_fixed_source_invades_only_through_a_reaction() -> None:
    """A fixed cell of colour 1 next to colour 0: with 1|0 -> 1|1 colour 1 grows from it,
    with the reaction switched to 2|0 -> 2|2 (no 2 present) nothing happens."""
    g = hex_torus(16)
    fixed = _mask(g.n_cells, [0])
    cols = np.zeros(g.n_cells, int)
    cols[0] = 1
    on = ReactionSet([Reaction("edge", [1, 0], [1, 1], 1.0)], 3)
    sim = Sim(g, on, cols, seed=0, fixed=fixed)
    sim.sweep(200)
    assert sim.counts()[1] == g.n_cells - 1  # voter-like invasion from a zealot: total
    off = ReactionSet([Reaction("edge", [2, 0], [2, 2], 1.0)], 3)
    sim = Sim(g, off, cols, seed=0, fixed=fixed)
    sim.sweep(50)
    assert sim.counts()[1] == 0


def test_shuffled_with_fixed_cells_and_holes_matches_mean_field() -> None:
    rs = ReactionSet(
        [
            Reaction("cell", [0], [1], 0.4),
            Reaction("edge", [1, 2], [2, 2], 0.6),
            Reaction("edge", [2, 0], [0, 0], 0.5),
            Reaction("vertex", [2, 2, 1], [2, 0, 1], 0.3),
        ],
        3,
    )
    g = hex_torus(128)
    rng = np.random.default_rng(5)
    n = g.n_cells
    fixed = rng.random(n) < 0.15
    holes = rng.random(n) < 0.05
    cols = initial_colours(n, [0.5, 0.2, 0.3], 3)
    cols[fixed] = rng.choice([0, 2], int(fixed.sum()), p=[0.3, 0.7])
    sim = Sim(g, rs, cols, seed=4, shuffled=True, holes=holes, fixed=fixed)
    ts, fr = run_timeline(sim, 30, record_every=2)
    n_open, n_free = len(sim.open_idx), len(sim.free_idx)
    y0 = fr[0] * n_free / n_open  # free part as a fraction of world cells
    f = rhs(rs, fixed=sim.fixed_fractions(), weights=sim.element_weights())
    mf = integrate(f, y0, ts) * n_open / n_free
    assert ts[-1] > 20
    assert np.abs(fr - mf).max() < 0.02
    # the plain mean field (fixed cells ignored) is visibly wrong here
    plain = integrate(rhs(rs), fr[0], ts)
    assert np.abs(fr - plain).max() > 0.05


# ---- level files ---------------------------------------------------------------------------
def test_terrain_json_round_trip(tmp_path) -> None:
    g = hex_torus(8)
    spec = {"holes": [[0, 0], [3, 2]], "fixed": [[1, 1, 2], [-1, 5, 0]]}
    t = terrain_from_json(g, spec, 3)
    assert t.hole.sum() == 2 and t.fixed.sum() == 2
    back = terrain_to_json(g, t.hole, t.fixed, t.colour)
    assert sorted(map(tuple, back["holes"])) == sorted(map(tuple, spec["holes"]))
    assert sorted(map(tuple, back["fixed"])) == sorted(map(tuple, spec["fixed"]))
    with pytest.raises(ValueError):
        terrain_from_json(g, {"holes": [[100, 100]]}, 3)
    level = {
        "id": "X",
        "grid": {"type": "hex", "n": 8},
        "palette": 3,
        "reactions": [{"shape": "edge", "in": [0, 1], "out": [0, 0], "rate": 1.0}],
        "start": {"seed": 1, "fractions": [0.5, 0.5, 0.0], "burn_in_sweeps": 5},
        "terrain": spec,
    }
    p = tmp_path / "X.json"
    p.write_text(json.dumps(level))
    sim = make_sim(load_level(p))
    assert sim.hole.sum() == 2 and sim.fixed.sum() == 2
    assert (sim.colours[t.fixed] == t.colour[t.fixed]).all()


# ---- patch sizes ---------------------------------------------------------------------------
def test_patch_sizes_on_simple_mosaics() -> None:
    g = hex_torus(8)
    n = g.n_cells
    allc = np.arange(n)
    assert patch_sizes(np.zeros(n, int), g.edges, allc, 2)[0] == n
    rows = (np.arange(n) // 8) % 2  # alternating rows: each row is a ring of 8 cells
    s, sc = patch_sizes(rows, g.edges, allc, 2)
    assert s == 8 and (sc == 8).all()
    assert interface_density(rows, g.edges) == pytest.approx(2 / 3)
    # one colour cut into two bands of 3 rows by two rows of holes
    rs = ReactionSet([Reaction("edge", [0, 1], [0, 0], 1.0)], 2)
    holes = np.isin(np.arange(n) // 8, [0, 4])
    sim = Sim(g, rs, np.zeros(n, int), seed=0, holes=holes)
    s, sc = sim_patch_sizes(sim)
    assert s == 24 and sc[0] == 24 and sc[1] == 0


def test_size_goal_and_fail() -> None:
    tr = GoalTracker([{"type": "size", "range": [50, 1e9], "duration": 1.0}])
    assert tr.needs_size
    c = np.array([5, 5])
    assert tr.update(0.0, c, {"size": 10.0, "size_c": np.array([10.0, 10.0])}) == "running"
    tr.update(1.0, c, {"size": 60.0, "size_c": np.array([60.0, 60.0])})
    assert tr.update(2.5, c, {"size": 80.0, "size_c": np.array([80.0, 80.0])}) == "won"
    tr = GoalTracker(
        [{"type": "extinct", "color": 1}],
        [{"type": "size", "color": 0, "range": [0, 3]}],
    )
    assert tr.update(0.0, c, {"size": 4.0, "size_c": np.array([2.0, 9.0])}) == "failed"
    assert not GoalTracker([{"type": "extinct", "color": 1}]).needs_size

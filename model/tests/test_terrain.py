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


# ---- drawn starts and contact goals ----------------------------------------------------------
def test_map_start_and_contact_goal(tmp_path) -> None:
    level = {
        "id": "M",
        "palette": 3,
        "map": {
            "rows": ["#####", "#R..#", "#.yG#", "#####"],
            "legend": {"#": "wall", ".": 0, "y": 1, "R": [1, "fixed"], "G": [2, "fixed"]},
        },
        "reactions": [{"shape": "edge", "in": [1, 0], "out": [1, 1], "rate": 1.0}],
        "goals": [{"type": "contact", "colors": [1, 2]}],
    }
    p = tmp_path / "M.json"
    p.write_text(json.dumps(level))
    lv = load_level(p)
    assert lv.grid == {"type": "hex", "w": 5, "h": 4, "periodic": False}
    sim = make_sim(lv)
    assert sim.hole.sum() == 14 and sim.fixed.sum() == 2
    assert sim.colours[1 * 5 + 1] == 1 and sim.colours[2 * 5 + 3] == 2
    assert list(sim.counts()) == [3, 1, 0]  # free cells: three white, one yellow (colour 1)
    from mosaic_model.observables import observe

    tr = GoalTracker(lv.goals)
    assert tr.needs_obs
    # the yellow cell (colour 1) already touches green: contact is by colour
    assert tr.update(0.0, sim.counts(), observe(sim, size=False)) == "won"
    bad = dict(level, map=dict(level["map"], rows=["####", "#R.#"]))
    p.write_text(json.dumps(bad | {"grid": {"type": "hex", "n": 8}}))
    with pytest.raises(ValueError):
        make_sim(load_level(p))


def test_maze_level_loads() -> None:
    from pathlib import Path

    lv = load_level(Path(__file__).resolve().parents[2] / "levels" / "sandbox" / "X04_maze.json")
    sim = make_sim(lv)
    assert not sim.graph.periodic and sim.fixed.sum() > 0 and sim.hole.sum() > 0
    rs = sim.reactions
    free = [b for b in rs.blocks() if b not in rs.locked]
    assert sorted(free) == [(1, "in", 1), (2, "in", 0)]


def test_contact_all_remembers_touched_targets() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    tr = GoalTracker([{"type": "contact", "colors": [1, 3], "all": True}])
    col = np.array([3, 1, 0, 0, 3])  # red touches the first green only
    assert tr.update(0.0, np.array([2, 1]), {"col": col, "edges": edges}) == "running"
    assert tr.progress(0.0) == [0.5]
    col = np.array([3, 0, 0, 1, 3])  # red moved on: the first touch is remembered
    assert tr.update(1.0, np.array([2, 1]), {"col": col, "edges": edges}) == "won"


def test_vine_follows_the_wall_and_mirrors_with_one_swap() -> None:
    """(R, W, S) -> (R, R, S): red grows along the top wall in one direction only; swapping
    the two input blocks reverses it."""
    from mosaic_model.graph import hex_rect
    from mosaic_model.level import map_start

    rows = ["S" * 20, "S" + "." * 8 + "R" + "." * 9 + "S"] + ["S" + "." * 18 + "S"] * 3
    rows += ["S" * 20]
    legend = {"S": [2, "fixed"], ".": 0, "R": [1, "fixed"]}
    g = hex_rect(20, 6)
    col, ter, _ = map_start(g, {"rows": rows, "legend": legend}, 3)
    for swap, side in ((False, "left"), (True, "right")):
        rs = ReactionSet([Reaction("vertex", [1, 0, 2], [1, 1, 2], 1.0)], 3)
        sim = Sim(g, rs, col, seed=3, holes=ter.hole, fixed=ter.fixed)
        if swap:
            sim.swap((0, "in", 0), (0, "in", 1))
        sim.advance(5.0)
        red = np.flatnonzero(sim.colours[20:40] == 1)
        assert len(red) > 2 and (red.max() == 9 if side == "left" else red.min() == 9)


def test_zone_goal_rect_zones_and_random_fills(tmp_path) -> None:
    rows = ["<<<<>>>>"] * 4
    level = {
        "id": "Z",
        "palette": 3,
        "map": {
            "rows": rows,
            "legend": {"<": {"random": [0.0, 1.0, 0.0]}, ">": {"random": [0.0, 0.0, 1.0]}},
            "zones": {"L": {"rect": [0, 3, 0, 3]}, "R": ">"},
        },
        "reactions": [{"shape": "edge", "in": [1, 2], "out": [1, 1], "rate": 1.0}],
        "start": {"seed": 1},
        "goals": [{"type": "zone", "zone": "R", "color": 1, "range": [0.5, 1.0]}],
        "fail": [{"type": "zone", "zone": "L", "color": 1, "range": [0.0, 0.0]}],
    }
    p = tmp_path / "Z.json"
    p.write_text(json.dumps(level))
    lv = load_level(p)
    sim = make_sim(lv)
    assert (sim.colours[[r * 8 + c for r in range(4) for c in range(4)]] == 1).all()
    assert (sim.colours[[r * 8 + c for r in range(4) for c in range(4, 8)]] == 2).all()
    tr = lv.tracker(sim.graph)
    assert sorted(tr.zones) == ["L", "R"] and len(tr.zones["L"]) == 16
    from mosaic_model.observables import observe

    assert tr.update(0.0, sim.counts(), observe(sim, size=False)) == "running"
    sim.advance(30.0)  # red invades blue
    assert tr.update(sim.t, sim.counts(), observe(sim, size=False)) == "won"

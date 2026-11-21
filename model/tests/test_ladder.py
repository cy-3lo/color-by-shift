"""Shortest solutions of foam levels and Mondrian-like compositions."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from mosaic_model.foam import load_foam_level, make_foam, random_scramble
from mosaic_model.foam_solver import all_moves, proven_scramble, shortest
from mosaic_model.mondrian import composition, polygons, touching

ROOT = Path(__file__).resolve().parents[2]


def brute(foam, start, target, n, depth):
    """Shortest length and number of shortest solutions by trying every list of moves."""
    moves = all_moves(foam, n)
    for d in range(depth + 1):
        count = 0
        for seq in itertools.product(moves, repeat=d):
            col, ok = start, True
            for mv in seq:
                new = foam.move(col, *mv)
                if (new == col).all():  # a move that changes nothing is not a move
                    ok = False
                    break
                col = new
            count += ok and bool((col == target).all())
        if count:
            return d, count
    return None, 0


def test_solver_agrees_with_brute_force() -> None:
    foam = make_foam(14, 1.0, 2, relax=2, width=120)
    rng = np.random.default_rng(5)
    for trial in range(6):
        target = rng.integers(0, 3, foam.n_cells)
        scr = random_scramble(foam, target, 3, 3, np.random.default_rng(trial), min_pairs=1)
        start = target
        for mv in scr:
            start = foam.move(start, *mv)
        sol = shortest(foam, start, target, 3, max_depth=3)
        assert (sol.length, sol.count) == brute(foam, start, target, 3, 3)
        col = start
        for mv in sol.path:
            col = foam.move(col, *mv)
        assert (col == target).all() and len(sol.path) == sol.length


def test_solver_reports_what_it_proved() -> None:
    foam = make_foam(40, 1.0, 1, relax=2, width=160)
    target = np.arange(foam.n_cells) % 3
    assert shortest(foam, target, target, 3).length == 0
    scr, sol = proven_scramble(foam, target, 3, 4, seed=3)
    start = target
    for mv in scr:
        start = foam.move(start, *mv)
    short = shortest(foam, start, target, 3, max_depth=3)
    assert sol.length == 4 and short.length is None and short.proved == 3


@pytest.mark.parametrize("lid", ["F01", "F02", "F03", "F05", "F08", "P01"])
def test_shipped_levels_are_never_longer_than_their_scramble(lid: str) -> None:
    lv = load_foam_level(ROOT / "levels" / "foam" / f"{lid}.json")
    sol = shortest(lv.foam, lv.scrambled(), lv.target, lv.n, max_depth=lv.par)
    assert sol.length is not None and sol.length <= lv.par


@pytest.mark.parametrize("n_rects, n_colours", [(2, 2), (5, 3), (10, 5)])
def test_composition_tiles_the_box_and_follows_into_the_foam(n_rects, n_colours) -> None:
    aspect = 4 / 3
    rects, cols = composition(n_rects, n_colours, aspect, seed=7)
    assert len(rects) == n_rects and sorted(set(cols)) == list(range(n_colours))
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects) == pytest.approx(aspect)
    assert all(min(x1 - x0, y1 - y0) >= 0.16 - 1e-9 for x0, y0, x1, y1 in rects)
    assert len(touching(rects)) >= n_rects - 1
    assert composition(n_rects, n_colours, aspect, seed=7) == (rects, cols)
    foam = make_foam(12 * n_rects, aspect, 1, relax=3, width=240, shapes=polygons(rects))
    assert foam.n_regions == n_rects
    for k, (x0, y0, x1, y1) in enumerate(rects):  # every site lies in its rectangle
        s = foam.sites[foam.site_region == k]
        assert len(s) >= 3
        assert (s[:, 0] >= x0).all() and (s[:, 0] <= x1).all()
        assert (s[:, 1] >= y0).all() and (s[:, 1] <= y1).all()

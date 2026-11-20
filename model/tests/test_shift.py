"""Reversible directional swaps (mosaic_model.shift)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mosaic_model.graph import hex_rect
from mosaic_model.shift import DIRECTIONS, ShiftGrid, Swap, flag, load_shift_level

ROOT = Path(__file__).resolve().parents[2]


def test_pairs_are_disjoint_and_cover_every_edge() -> None:
    g = hex_rect(12, 9)
    sg = ShiftGrid(g)
    total = 0
    for (name, _alt), (p, q) in sg.pairs.items():
        assert len(np.unique(np.concatenate([p, q]))) == 2 * len(p)  # no cell twice
        assert (g.axial[q] - g.axial[p] == DIRECTIONS[name]).all()
        total += len(p)
    assert total == len(g.edges)
    assert abs(sg.offset("|")[1]) < 1e-9 and sg.offset("/")[1] > 0 > sg.offset("\\")[1]


@pytest.mark.parametrize("n", [2, 3, 5])
def test_any_program_is_undone_in_reverse_order(n: int) -> None:
    rng = np.random.default_rng(n)
    g = hex_rect(18, 12)
    sg = ShiftGrid(g)
    col = rng.integers(0, n, g.n_cells)
    prog = [
        Swap(str(rng.choice(list(DIRECTIONS))), int(rng.integers(n)), int(rng.integers(n)))
        for _ in range(7)
    ]
    end, t = sg.run(col, prog, 40)
    assert t == 40 and (end != col).any()
    assert (np.bincount(end, minlength=n) == np.bincount(col, minlength=n)).all()  # conserved
    back, t = sg.run(end, prog, -40, t)
    assert t == 0 and (back == col).all()


def test_half_step_is_its_own_inverse_and_order_matters() -> None:
    g = hex_rect(12, 9)
    sg = ShiftGrid(g)
    col = flag("disc", g, 2)
    s = Swap("|", 0, 1)
    once, moved = sg.half(col, s, 0)
    assert moved > 0 and (sg.half(once, s, 0)[0] == col).all()
    assert (sg.apply(sg.apply(col, s), s, inverse=True) == col).all()
    assert (sg.apply(sg.apply(col, s), s) != col).any()  # forward twice is not the way back
    # with 3 colours a swap depends on what the cells hold, and the order matters
    col3 = np.random.default_rng(0).integers(0, 3, g.n_cells)
    a, b = Swap("|", 0, 1), Swap("\\", 1, 2)
    end, _ = sg.run(col3, [a, b], 2)
    assert (sg.run(end, [a, b], -2, 2)[0] == col3).all()
    assert (sg.run(end, [b, a], -2, 2)[0] != col3).any()  # wrong order does not recover it


def test_two_colours_are_four_sliding_sublattices() -> None:
    """With 2 colours every pair swaps (visibly or not), so a reaction moves each cell two
    steps along its direction, forwards or backwards by the parity of its coordinate, and
    keeps that parity: away from the border the reactions commute and order is irrelevant."""
    g = hex_rect(18, 12)
    sg = ShiftGrid(g)
    col = flag("disc", g, 2)  # far from the border for two steps
    for x, y in (("|", "/"), ("|", "\\"), ("/", "\\")):
        a, b = Swap(x, 0, 1), Swap(y, 0, 1)
        assert (sg.run(col, [a, b], 2)[0] == sg.run(col, [b, a], 2)[0]).sum() >= g.n_cells - 8
    ax = np.asarray(g.axial)
    one = np.zeros(g.n_cells, int)
    k = int(np.argmin(np.abs(g.centers - g.centers.mean(axis=0)).sum(axis=1)))
    one[k] = 1
    out = sg.apply(one, Swap("|", 0, 1))
    (m,) = np.nonzero(out)
    assert len(m) == 1 and abs(int(ax[m[0], 0] - ax[k, 0])) == 2 and ax[m[0], 1] == ax[k, 1]


def test_swap_only_touches_its_two_colours_and_direction() -> None:
    g = hex_rect(12, 9)
    sg = ShiftGrid(g)
    col = flag("bands_v", g, 3)
    out = sg.apply(col, Swap("|", 0, 1))
    assert ((col == 2) == (out == 2)).all() and (out != col).any()
    rows = flag("bands_h", g, 2)
    assert (sg.apply(rows, Swap("|", 0, 1)) == rows).all()  # equal colours along the rows
    assert (sg.apply(col, Swap("|", 1, 1)) == col).all()
    with pytest.raises(ValueError):
        Swap("-", 0, 1)


def test_shift_levels_load_and_are_solved_by_their_own_program() -> None:
    files = sorted((ROOT / "levels" / "shift").glob("*.json"))
    assert files
    for f in files:
        lv = load_shift_level(f)
        start = lv.scrambled()
        assert (start != lv.target).any(), f.name
        back, _ = lv.grid.run(start, lv.program, -lv.cycles * len(lv.program))
        assert (back == lv.target).all(), f.name
        assert set(np.unique(lv.target)) <= set(range(lv.n))

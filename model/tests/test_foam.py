"""Reversible swaps on a foam (mosaic_model.foam)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mosaic_model.foam import (
    Foam,
    foam_picture,
    load_foam,
    load_foam_level,
    make_foam,
    nearest_site,
    random_scramble,
    region_map,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def foam() -> Foam:
    return make_foam(60, 1.5, 3, relax=3, width=240)


def test_foam_groups_are_matchings_and_cover_every_edge(foam: Foam) -> None:
    assert foam.n_cells == 60 and len(foam.edges) > 100
    assert (foam.edges[:, 0] < foam.edges[:, 1]).all()
    seen = 0
    for u, v in foam.pairs:
        assert len(np.unique(np.concatenate([u, v]))) == 2 * len(u)  # no cell twice
        seen += len(u)
    assert seen == len(foam.edges)  # every edge in exactly one group
    deg = np.bincount(foam.edges.ravel(), minlength=foam.n_cells)
    assert deg.min() >= 1 and deg.max() <= foam.n_groups <= deg.max() + 2
    u, v = foam.edges[0]
    assert foam.group_of(v, u) == foam.group[0] and foam.group_of(u, u) is None


def test_foam_is_deterministic_and_survives_json(foam: Foam, tmp_path) -> None:
    again = make_foam(60, 1.5, 3, relax=3, width=240)
    assert np.allclose(again.sites, foam.sites) and (again.group == foam.group).all()
    import json

    p = tmp_path / "foam.json"
    p.write_text(json.dumps(foam.to_json()))
    back = load_foam(p)
    assert (back.edges == foam.edges).all() and np.allclose(back.sites, foam.sites, atol=1e-4)
    lab = nearest_site(foam.sites, 90, 60, 1.5)
    assert lab.shape == (60, 90) and set(np.unique(lab)) == set(range(60))


@pytest.mark.parametrize("n", [2, 3, 5])
def test_foam_moves_undo_themselves_and_sequences_reverse(foam: Foam, n: int) -> None:
    rng = np.random.default_rng(n)
    col = rng.integers(0, n, foam.n_cells)
    moves = [
        (int(rng.integers(foam.n_groups)), int(rng.integers(n)), int(rng.integers(n)))
        for _ in range(30)
    ]
    c = col
    for m in moves:
        new = foam.move(c, *m)
        assert (foam.move(new, *m) == c).all()  # its own inverse
        assert (np.bincount(new, minlength=n) == np.bincount(c, minlength=n)).all()
        c = new
    assert (c != col).any()
    for m in moves[::-1]:
        c = foam.move(c, *m)
    assert (c == col).all()
    assert (foam.move(col, 0, 1, 1) == col).all()  # same colour twice: nothing


def test_two_cells_sharing_a_group_are_refused() -> None:
    sites = np.array([[0.2, 0.5], [0.7, 0.5], [1.2, 0.5]])
    with pytest.raises(ValueError):
        Foam(1.5, sites, np.array([[0, 1], [1, 2]]), [0, 0])


def test_random_scramble_changes_the_picture_at_every_move(foam: Foam) -> None:
    target = foam_picture("bands_v", foam, 3)
    scr = random_scramble(foam, target, 3, 4, np.random.default_rng(0))
    col = target
    for a, b in zip(scr, scr[1:], strict=False):
        assert a != b
    for m in scr:
        assert len(foam.swapping(col, *m)[0]) >= 2
        col = foam.move(col, *m)


def test_foam_levels_load_and_are_solved_by_the_reversed_scramble() -> None:
    files = sorted((ROOT / "levels" / "foam").glob("*.json"))
    assert files
    for f in files:
        lv = load_foam_level(f, ROOT)
        col = lv.scrambled()
        assert (col != lv.target).any(), f.name
        for m in lv.scramble[::-1]:
            col = lv.foam.move(col, *m)
        assert (col == lv.target).all(), f.name
        assert lv.par == len(lv.scramble) and set(np.unique(lv.target)) <= set(range(lv.n))


def test_foam_follows_drawn_shapes() -> None:
    """A pentagon in the box: no cell crosses its outline, so the outline is made of cell
    sides; both regions are paved; the picture is one colour per region."""
    import math

    penta = [
        [
            0.75 + 0.38 * math.cos(math.radians(72 * k - 90)),
            0.5 + 0.38 * math.sin(math.radians(72 * k - 90)),
        ]
        for k in range(5)
    ]
    f = make_foam(70, 1.5, 5, relax=3, width=240, shapes=[penta])
    assert f.n_regions == 2 and set(np.unique(f.site_region)) == {0, 1}
    lab = f.raster(300, 200)
    reg = region_map(f.shapes, 300, 200, 1.5)
    assert (f.site_region[lab] == reg).all()  # every pixel belongs to a cell of its region
    inside = (reg == 1).mean()
    assert 0.2 < inside < 0.3  # area of the pentagon: 0.343 / 1.5
    assert set(np.unique(lab)) == set(range(f.n_cells))  # no empty cell
    for u, v in f.pairs:
        assert len(np.unique(np.concatenate([u, v]))) == 2 * len(u)
    cross = (f.site_region[f.edges[:, 0]] != f.site_region[f.edges[:, 1]]).sum()
    assert cross >= 10  # cells of the two regions are neighbours across the outline
    with pytest.raises(ValueError):
        nearest_site(f.sites, 60, 40, 1.5, np.zeros(f.n_cells, int), f.shapes)  # empty region

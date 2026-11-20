"""Painting toy: pictures, rule presets and GIF export (toy/pictures.py, toy/painting_toy.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "toy"))

pictures = pytest.importorskip("pictures")  # needs Pillow (optional dependency "tools")

from mosaic_model import ReactionSet, Sim  # noqa: E402


@pytest.mark.parametrize("n", range(2, 9))
def test_presets_build_and_stay_in_palette(n: int) -> None:
    for name in pictures.PRESETS:
        rx = pictures.preset(name, n)
        assert all(r.rate == 1.0 for r in rx)
        assert all(0 <= c < n for r in rx for c in [*r.inp, *r.out])
        ReactionSet(rx, n)  # lookup tables build
        assert name in pictures.DESCRIPTIONS


def test_parse_rules() -> None:
    rx = pictures.parse_rules("01>00 ; 3>1 ; 012>120", 3)
    assert [r.shape for r in rx] == ["edge", "cell", "vertex"]
    assert rx[1].inp == [0] and rx[2].out == [1, 2, 0]  # 3 wraps to 0
    assert pictures.rule_text(rx) == "01>00 ; 0>1 ; 012>120"
    with pytest.raises(ValueError):
        pictures.parse_rules("01>0", 3)


def test_grid_and_generated_pictures() -> None:
    g = pictures.grid_for(1.5, 6000)
    assert abs(g.n_cells - 6000) < 300
    col = pictures.generated("regular", g, 3, 1)
    e = g.edges
    assert (col[e[:, 0]] != col[e[:, 1]]).all()  # the 3-colouring of the hex lattice
    col = pictures.generated("regular", g, 5, 1)
    assert (col[e[:, 0]] != col[e[:, 1]]).all()  # 5 colours: neighbours differ by 1 or 2
    for kind in ("random", "stripes", "rings", "destijl"):
        c = pictures.generated(kind, g, 4, 1)
        assert c.shape == (g.n_cells,) and c.min() >= 0 and c.max() <= 3
    d = pictures.generated("destijl", g, 4, 1)
    assert (d == 0).any() and (d == 3).mean() > 0.3  # dark lines, mostly light fills


def test_kmeans_palette_is_dark_to_light() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(v, 5, (200, 3)) for v in (200, 30, 120)])
    pal, lab = pictures.kmeans(x, 3, 1)
    s = pal.astype(int).sum(axis=1)
    assert (np.diff(s) > 0).all()
    assert (lab[200:400] == 0).all() and (lab[:200] == 2).all()


def test_painting_loads_and_runs() -> None:
    entry = {"key": "great_wave", "file": "assets/paintings/great_wave.jpg", "title": "w"}
    cp = {
        "clusters": 12,
        "contrast_share": 0.02,
        "accent_share": 0.004,
        "generated": "candy",
        "default": "contrast",
    }
    assert pictures.choice(entry, cp) == "contrast"
    assert pictures.choice({**entry, "palette": "accent"}, cp) == "accent"
    assert pictures.choice({"key": "r", "generated": "random"}, cp) == "candy"
    p = pictures.load_picture(entry, 4, 3000, 0.5, 1, cp, "present")
    assert p.palette.shape == (4, 3) and p.colours.max() == 3
    c = pictures.load_picture(entry, 4, 3000, 0.5, 1, cp, "contrast")
    assert c.palette.shape == (4, 3) and set(np.unique(c.colours)) == {0, 1, 2, 3}
    assert pictures.min_delta_e(c.palette) > pictures.min_delta_e(p.palette)
    own = pictures.load_picture({**entry, "share": 0.3}, 4, 3000, 0.5, 1, cp, "contrast")
    assert (own.palette != c.palette).any()  # a picture's own share threshold is used
    lum = [pictures.lum(v) for v in c.palette]
    assert lum == sorted(lum)  # dark to light
    crop = pictures.load_picture(
        {**entry, "crop": [0.0, 0.0, 0.5, 1.0]}, 4, 3000, 0.5, 1, cp, "present"
    )
    assert crop.graph.n_cells != p.graph.n_cells or (crop.colours != p.colours).any()
    sim = Sim(p.graph, ReactionSet(pictures.preset("spirals", 4), 4), p.colours, 1)
    sim.advance(2.0)
    assert sim.t > 1.9


def test_contrast_palette_keeps_rare_distinct_colours() -> None:
    """900 greys in three close shades and 100 red samples: the present choice spends its
    three colours on the greys or merges them; the contrast choice keeps the red."""
    rng = np.random.default_rng(0)
    greys = [rng.normal(v, 2, (300, 3)) for v in (100, 120, 140)]
    red = rng.normal(0, 2, (100, 3)) + np.array([200, 40, 40])
    x = np.clip(np.vstack([*greys, red]), 0, 255)
    pal, lab, de = pictures.contrast_palette(x, 3, 1, k=8, min_share=0.02)
    assert pal.shape == (3, 3) and de > 15
    assert any(c[0] > 150 and c[1] < 90 for c in pal.astype(int))  # the red is kept
    assert len(set(lab[900:])) == 1 and (lab[:300] != lab[900]).all()
    lab_white = pictures.to_lab(np.array([255, 255, 255]))
    assert np.allclose(lab_white, [100, 0, 0], atol=0.1)  # Lab of white


def test_write_gif(tmp_path) -> None:
    from PIL import Image

    import painting_toy

    g = pictures.grid_for(1.0, 400)
    pal = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255]], np.uint8)
    frames = [pictures.generated("random", g, 3, s) for s in range(5)]
    from mosaic_model.raster import hex_pixel_map

    pm = hex_pixel_map(g, 3.0)
    gc = {"frame_ms": 50, "hold_start_ms": 100, "hold_end_ms": 200}
    p = tmp_path / "m.gif"
    painting_toy.write_gif(p, pm, frames, pal, gc)
    im = Image.open(p)
    assert im.n_frames == 5 and im.size == (pm.shape[1], pm.shape[0])
    im.seek(4)
    rgb = np.asarray(im.convert("RGB"))
    cell = pm[pm.shape[0] // 2, pm.shape[1] // 2]
    assert tuple(rgb[pm.shape[0] // 2, pm.shape[1] // 2]) == tuple(pal[frames[4][cell]])


def test_every_toy_picture_has_its_file_and_a_valid_colour_choice() -> None:
    import tomllib

    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    keys = [p["key"] for p in cfg["pictures"]]
    assert len(keys) == len(set(keys))
    for p in cfg["pictures"]:
        if "file" in p:
            assert (ROOT / p["file"]).exists(), p["file"]
            assert pictures.choice(p, cfg["palettes"]) in pictures.PAINTING_OPTIONS
            assert 0 < p.get("share", 0.02) < 1


def test_fit_aspect_and_common_format() -> None:
    from PIL import Image

    wide = pictures.fit_aspect(Image.new("RGB", (300, 100)), 4 / 3)
    assert wide.size == (133, 100)
    tall = pictures.fit_aspect(Image.new("RGB", (120, 300)), 4 / 3, (0.5, 0.0))
    assert tall.size == (120, 90)
    cp = {"clusters": 12, "contrast_share": 0.02, "accent_share": 0.004, "generated": "candy"}
    entry = {"key": "great_wave", "file": "assets/paintings/great_wave.jpg", "title": "w"}
    a = pictures.load_picture(entry, 4, 3000, 0.5, 1, cp, "present", 4 / 3)
    b = pictures.load_picture(
        {"key": "r", "generated": "rings"}, 4, 3000, 0.5, 1, cp, "candy", 4 / 3
    )
    assert a.graph.n_cells == b.graph.n_cells  # every picture on the same grid

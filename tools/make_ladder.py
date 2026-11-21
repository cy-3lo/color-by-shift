"""Build the candidate levels of the ladder, prove their par, and draw contact sheets.

    python tools/make_ladder.py [--only M3a Q05 ...] [--no-build]

Reads levels/candidates/ladder.toml. For each candidate writes grids/candidates/foam_<id>.json
and levels/candidates/<id>.json, with a scramble that the search from both ends
(mosaic_model.foam_solver) proves cannot be undone in fewer moves. Then writes in
runs/ladder/<stamp>/: table.md (cells, groups, par, number of shortest solutions, share of
the orders of one solution that also solve, moves available at the start, cells out of
place), mondrian.png and paintings.png (scrambled picture | picture to recover), and
params.json (the ladder file, the git hash). --no-build only measures the existing files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))
sys.path.insert(0, str(ROOT / "tools"))

import make_painting_level  # noqa: E402
import palette as palette_mod  # noqa: E402
from mosaic_model.foam import FoamLevel, load_foam_level, make_foam  # noqa: E402
from mosaic_model.foam_solver import all_moves, proven_scramble, shortest  # noqa: E402
from mosaic_model.mondrian import lined_composition  # noqa: E402

GRIDS = ROOT / "grids" / "candidates"
LEVELS = ROOT / "levels" / "candidates"


def build_mondrian(spec: dict, lv: dict) -> None:
    polys, cols = lined_composition(
        lv["rects"], lv["colours"], spec["aspect"], lv["seed"], spec["bar"], spec["min_side"]
    )
    # enough cells in the lines for them to be about as long as the other cells are wide
    lines = spec["aspect"] - sum((p[2][0] - p[0][0]) * (p[2][1] - p[0][1]) for p in polys)
    in_lines = round(lines / spec["bar"] / (spec["aspect"] / lv["cells"]) ** 0.5)
    foam = make_foam(
        lv["cells"], spec["aspect"], lv["seed"], relax=8, shapes=polys,
        region_sites={0: in_lines},
    )  # fmt: skip
    target = np.array(cols)[foam.site_region]
    scr, _ = proven_scramble(foam, target, lv["colours"], lv["moves"], lv["seed"] + 1000)
    GRIDS.mkdir(parents=True, exist_ok=True)
    rel = f"grids/candidates/foam_{lv['id']}.json"
    (ROOT / rel).write_text(json.dumps(foam.to_json() | {"seed": lv["seed"]}, indent=1) + "\n")
    level = {
        "id": lv["id"],
        "title": "",
        "foam": rel,
        "colours": spec["palette"][: lv["colours"]],
        "picture": {"regions": cols},
        "scramble": [{"group": g, "pair": [a, b]} for g, a, b in scr],
    }
    (LEVELS / f"{lv['id']}.json").write_text(json.dumps(level, indent=1) + "\n")


def build_painting(toy_cfg: dict, lv: dict) -> None:
    seed = lv.get("seed", 1)
    foam, level, _, target, _, _, _ = make_painting_level.build(
        toy_cfg, lv["picture"], lv["id"], lv.get("cells", 600), lv.get("moves", 6),
        seed=seed, grids=GRIDS, levels=LEVELS, palette=lv.get("palette"), share=lv.get("share"),
    )  # fmt: skip
    scr, _ = proven_scramble(foam, target, len(level["colours"]), lv.get("moves", 6), seed + 1000)
    level["scramble"] = [{"group": g, "pair": [a, b]} for g, a, b in scr]
    (LEVELS / f"{lv['id']}.json").write_text(json.dumps(level, indent=1) + "\n")


def measure(level: FoamLevel) -> dict:
    foam, start = level.foam, level.scrambled()
    sol = shortest(foam, start, level.target, level.n, max_depth=level.par)
    # orders of the found solution that also solve: 1 = any order, 1 / par! = one order only
    orders = sorted(set(itertools.permutations(sol.path)))  # a move may occur twice
    good = 0
    for seq in orders:
        col = start
        for mv in seq:
            col = foam.move(col, *mv)
        good += bool((col == level.target).all())
    deg = np.bincount(foam.edges.ravel(), minlength=foam.n_cells)
    return {
        "id": level.id,
        "colours": level.n,
        "regions": foam.n_regions,
        "cells": foam.n_cells,
        "groups": foam.n_groups,
        "max_neighbours": int(deg.max()),
        "par": level.par,
        "shortest": sol.length,
        "solutions": sol.count,
        "orders_ok": f"{good}/{len(orders)}",
        "moves_at_start": sum(
            len(foam.swapping(start, *m)[0]) > 0 for m in all_moves(foam, level.n)
        ),
        "cells_wrong": int((start != level.target).sum()),
    }


def tile(level: FoamLevel, w: int) -> Image.Image:
    """Scrambled picture and picture to recover, side by side, with the cell sides."""
    h = round(w / level.foam.aspect)
    cells = level.foam.raster(w, h)
    line = np.zeros((h, w), bool)
    line[:, :-1] |= cells[:, :-1] != cells[:, 1:]
    line[:-1, :] |= cells[:-1, :] != cells[1:, :]
    pal = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in level.colours])
    out = Image.new("RGB", (2 * w + 6, h), (28, 28, 34))
    for k, col in enumerate((level.scrambled(), level.target)):
        rgb = pal[col][cells]
        rgb[line] = (16, 16, 20)
        out.paste(Image.fromarray(rgb.astype(np.uint8)), (k * (w + 6), 0))
    return out


def sheet(levels: list[FoamLevel], rows: list[dict], per_row: int, w: int, path: Path) -> None:
    if not levels:
        return
    tiles = [tile(lv, w) for lv in levels]
    tw, th = tiles[0].size
    pad, label = 14, 22
    n_rows = -(-len(tiles) // per_row)
    img = Image.new(
        "RGB", (per_row * (tw + pad) + pad, n_rows * (th + label + pad) + pad), (20, 20, 24)
    )
    draw = ImageDraw.Draw(img)
    for k, (t, r) in enumerate(zip(tiles, rows, strict=True)):
        x = pad + (k % per_row) * (tw + pad)
        y = pad + (k // per_row) * (th + label + pad)
        text = (
            f"{r['id']}   {r['cells']} cells, {r['groups']} groups, par {r['par']}, "
            f"{r['solutions']} shortest, orders {r['orders_ok']}"
        )
        draw.text((x, y + 4), text, fill=(236, 233, 224))
        img.paste(t, (x, y + label))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="candidate ids")
    ap.add_argument("--no-build", action="store_true", help="measure the existing files")
    a = ap.parse_args()
    text = (LEVELS / "ladder.toml").read_text()
    spec = tomllib.loads(text)
    toy_cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    wanted = lambda lv: not a.only or lv["id"] in a.only  # noqa: E731
    mond = [lv for lv in spec.get("mondrian", []) if wanted(lv)]
    paint = [lv for lv in spec.get("painting", []) if wanted(lv)]

    rgb = [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in spec.get("palette", [])]
    for n in range(2, len(rgb) + 1):
        worst = min(palette_mod.min_delta_e(rgb[:n], view) for view in range(3))
        print(f"palette, first {n} colours: smallest Delta E {worst:.1f} (worst of 3 visions)")

    if not a.no_build:
        for lv in mond:
            t = time.time()
            build_mondrian(spec, lv)
            print(f"built {lv['id']} ({time.time() - t:.1f} s)")
        for lv in paint:
            t = time.time()
            build_painting(toy_cfg, lv)
            print(f"built {lv['id']} {lv['picture']} ({time.time() - t:.1f} s)")

    out = ROOT / "runs" / "ladder" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    (out / "params.json").write_text(json.dumps({"git": git, "argv": sys.argv, "ladder": spec}))
    rows_all = []
    for name, group, per_row, w in (("mondrian", mond, 3, 300), ("paintings", paint, 2, 440)):
        levels = [load_foam_level(LEVELS / f"{lv['id']}.json") for lv in group]
        rows = [measure(lv) for lv in levels]
        for r, lv in zip(rows, group, strict=True):
            r["picture"] = lv.get("picture", f"{lv.get('rects')} rectangles")
        sheet(levels, rows, per_row, w, out / f"{name}.png")
        rows_all += rows
    keys = list(rows_all[0]) if rows_all else []
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows_all]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

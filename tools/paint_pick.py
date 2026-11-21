"""Page to pick paintings for the foam game: colours and number of cells (for an Artifact).

    python tools/paint_pick.py [--cells 250 400 600] [--tile 260] [--only KEY ...]

Paintings: assets/rawpixel/pictures.toml and the [[pictures]] of toy/toy.toml. Each one is
cut to 4:3 and quantised to the toy's 5 colours in up to four ways ("present" = the most
present colours; "contrast" with a minimum share of 5 %, 2 % or 0.4 % of the picture), its
colour map split into blocks with smooth outlines as tools/make_painting_level.py does,
and a foam built inside the blocks for every number of cells. Each tile is the picture to
recover as it would be shown in the game. Painters with flat shades come first (FIRST).
Writes runs/paintpick/<stamp>/index.html and cards_<k>.json; the page stores the picks in
the artifact's database (collection "painting_picks").
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))
sys.path.insert(0, str(ROOT / "tools"))

import pictures  # noqa: E402
from make_painting_level import components, majority, merge_small, smooth_blocks  # noqa: E402
from mosaic_model.foam import make_foam  # noqa: E402

LINE = (16, 16, 20)
SHARES = [0.05, 0.02, 0.004]
FIRST = ["gris", "picasso", "matisse", "klee", "mondrian"]


def data_uri(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": 80} if fmt == "JPEG" else {"optimize": True}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def trim(img: Image.Image, tol: int = 28) -> Image.Image:
    """Cuts a paper margin: the box of the pixels that differ from the corner colour."""
    small = img.copy()
    small.thumbnail((400, 400))
    small = small.filter(ImageFilter.GaussianBlur(2))
    bg = Image.new("RGB", small.size, small.getpixel((3, 3)))
    box = ImageChops.difference(small, bg).convert("L").point(lambda v: 255 * (v > tol)).getbbox()
    if box is None:
        return img
    s = img.size[0] / small.size[0]
    return img.crop(tuple(round(v * s) for v in box))


def open_cut(entry: dict, aspect: float) -> Image.Image:
    img = Image.open(ROOT / entry["file"])
    img.draft("RGB", (1600, 1600))  # large scans: decoded at a reduced size
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    if entry.get("crop"):
        x0, y0, x1, y1 = entry["crop"]
        W, H = img.size
        img = img.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
    if entry.get("trim"):
        img = trim(img)
    return pictures.fit_aspect(img, aspect, tuple(entry.get("anchor", (0.5, 0.5))))


def card(job: tuple) -> dict:
    entry, cfg, cells_list, tile, res, up, min_area = job
    sc, cp = cfg["sim"], cfg["palettes"]
    n, aspect = sc["colours"], sc["aspect"]
    img = open_cut(entry, aspect)
    w0, h0 = res, round(res / aspect)
    small = img.filter(ImageFilter.GaussianBlur(0.5 * img.size[0] / w0)).resize((w0, h0))
    x = np.asarray(small, dtype=float).reshape(-1, 3)
    var = [("present", None, *pictures.kmeans(x, n, sc["seed"]))]
    for sh in SHARES:
        pal, lab, _ = pictures.contrast_palette(x, n, sc["seed"], cp["clusters"], sh)
        if not any((pal == v[2]).all() for v in var):
            var.append((f"contrast {sh * 100:g} %", sh, pal, lab))
    th, tw = round(tile / aspect), tile
    versions = []
    for name, share, pal, lab in var:
        col = np.kron(lab.reshape(h0, w0), np.ones((up, up), dtype=np.int64))
        for _ in range(2):
            col = majority(col, n, up // 2 + 1)
        col = merge_small(col, int(min_area * col.size))
        regions, region_colour = smooth_blocks(components(col), col, 640)
        n_reg = len(region_colour)
        tiles = []
        for cells in cells_list:
            foam = make_foam(
                cells, aspect, 1, relax=4, width=480, regions=regions.astype(np.uint16),
                min_sites=1,
            )  # fmt: skip
            lab_px = foam.raster(tw, th)
            rgb = pal[region_colour[foam.site_region]][lab_px].astype(np.uint8)
            line = np.zeros((th, tw), bool)
            line[:, :-1] |= lab_px[:, :-1] != lab_px[:, 1:]
            line[:-1, :] |= lab_px[:-1, :] != lab_px[1:, :]
            rgb[line] = LINE
            tiles.append(
                {
                    "cells": int(foam.n_cells),
                    "asked": cells,
                    "groups": int(foam.n_groups),
                    "img": data_uri(Image.fromarray(rgb).quantize(n + 1), "PNG"),
                }
            )
        versions.append(
            {
                "name": name,
                "share": share,
                "de": round(pictures.min_delta_e(pal)),
                "blocks": n_reg,
                "colours": ["#" + bytes(int(v) for v in c).hex() for c in pal],
                "tiles": tiles,
            }
        )
    orig = img.copy()
    orig.thumbnail((tile, tile))
    artist = entry.get("artist", "unknown")
    rank = next((k for k, a in enumerate(FIRST) if a in artist.lower()), len(FIRST))
    return {
        "key": entry["key"],
        "file": entry["file"],
        "artist": artist,
        "title": entry.get("title", ""),
        "date": entry.get("date", ""),
        "new": entry["file"].startswith("assets/rawpixel"),
        "check": bool(entry.get("check")),
        "rights": entry.get("rights", ""),
        "rank": rank,
        "orig": data_uri(orig, "JPEG"),
        "versions": versions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=int, nargs="+", default=[250, 400, 600])
    ap.add_argument("--tile", type=int, default=260)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--up", type=int, default=4)
    ap.add_argument("--min-area", type=float, default=0.003)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    new = tomllib.loads((ROOT / "assets" / "rawpixel" / "pictures.toml").read_text())["pictures"]
    entries = [e for e in new + cfg["pictures"] if "file" in e and not e.get("skip")]
    if a.only:
        entries = [e for e in entries if e["key"] in a.only]
    jobs = [(e, cfg, a.cells, a.tile, a.res, a.up, a.min_area) for e in entries]
    cards = []
    with ProcessPoolExecutor(a.jobs) as pool:  # raises if a worker dies (a Pool would hang)
        for c in pool.map(card, jobs):
            cards.append(c)
            print(len(cards), c["artist"][:28], "|", c["title"][:44], flush=True)
    cards.sort(key=lambda c: (c["rank"], not c["new"], c["artist"], c["title"]))
    out = ROOT / "runs" / "paintpick" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    parts, cur, size = [], [], 0
    for c in cards:  # files of at most about 4 MB
        s = json.dumps(c)
        if cur and size + len(s) > 4_000_000:
            parts.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s)
    parts.append(cur)
    for k, p in enumerate(parts):
        (out / f"cards_{k}.json").write_text("[\n" + ",\n".join(p) + "\n]\n")
    html = (ROOT / "tools" / "paint_pick.html").read_text().replace("__PARTS__", str(len(parts)))
    (out / "index.html").write_text(html)
    total = sum(len(s) for p in parts for s in p) / 1e6
    print(f"{len(cards)} paintings -> {out.relative_to(ROOT)} ({len(parts)} files, {total:.1f} MB)")


if __name__ == "__main__":
    main()

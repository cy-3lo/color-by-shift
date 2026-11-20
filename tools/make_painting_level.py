"""Turn a painting of the toy into a click-to-shift level on a foam that follows its outlines.

    python tools/make_painting_level.py --picture great_wave --id P01 [--cells 600] [--moves 6]

The painting (a [[pictures]] entry of toy/toy.toml: file, crop, anchor, colour choice) is cut
to 4:3, reduced to --res pixels wide and quantised to the toy's 5 colours. The colour map is
enlarged --up times, its staircase rounded by a majority filter, and split into connected
blocks of one colour; blocks smaller than --min-area of the picture are merged into the
neighbour they touch most. The foam is generated inside these blocks (mosaic_model.foam), so
their outlines are cell sides. Writes grids/foam_<id>.json, grids/foam_<id>_regions.npy,
levels/foam/<id>.json (with the painting as the reward) and a preview PNG in
runs/painting_level/. Existing files are kept unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

import pictures  # noqa: E402
from mosaic_model.foam import make_foam, random_scramble  # noqa: E402


def majority(col: np.ndarray, n: int, radius: int) -> np.ndarray:
    """Each pixel takes the most frequent colour of the (2 radius + 1)^2 square around it."""
    h, w = col.shape
    pad = np.pad(col, radius, mode="edge")
    votes = np.zeros((n, h, w), dtype=np.int32)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            win = pad[dy : dy + h, dx : dx + w]
            for c in range(n):
                votes[c] += win == c
    return votes.argmax(axis=0)


def components(col: np.ndarray) -> np.ndarray:
    """Connected blocks of one colour (4 neighbours): a label per pixel, 0, 1, 2 ..."""
    h, w = col.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    flat = col.tolist()
    k = 0
    for y0 in range(h):
        for x0 in range(w):
            if lab[y0, x0] >= 0:
                continue
            c = flat[y0][x0]
            lab[y0, x0] = k
            todo = deque([(y0, x0)])
            while todo:
                y, x = todo.popleft()
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and lab[yy, xx] < 0 and flat[yy][xx] == c:
                        lab[yy, xx] = k
                        todo.append((yy, xx))
            k += 1
    return lab


def merge_small(col: np.ndarray, min_px: int) -> np.ndarray:
    """Recolour every block smaller than min_px with the colour of the neighbouring block it
    touches most, smallest blocks first, until none is left."""
    col = col.copy()
    while True:
        lab = components(col)
        size = np.bincount(lab.ravel())
        small = np.nonzero(size < min_px)[0]
        if len(small) == 0:
            return col
        pairs = []
        for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
            m = a != b
            pairs += [np.stack([a[m], b[m]], 1), np.stack([b[m], a[m]], 1)]
        uniq, cnt = np.unique(np.concatenate(pairs), axis=0, return_counts=True)
        colour_of = np.zeros(len(size), dtype=np.int64)
        colour_of[lab.ravel()] = col.ravel()
        new = colour_of.copy()
        for k in small[np.argsort(size[small])][: max(1, len(small) // 2)]:
            nb = uniq[:, 0] == k
            if nb.any():
                new[k] = colour_of[uniq[nb, 1][cnt[nb].argmax()]]
        col = new[lab]


def main() -> None:
    cfg = tomllib.loads((ROOT / "toy" / "toy.toml").read_text())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picture", required=True, help="key of a [[pictures]] entry")
    ap.add_argument("--id", required=True, help="level id, e.g. P01")
    ap.add_argument("--cells", type=int, default=600)
    ap.add_argument("--moves", type=int, default=6)
    ap.add_argument("--res", type=int, default=128, help="width of the quantised picture")
    ap.add_argument("--up", type=int, default=4, help="enlargement before rounding")
    ap.add_argument("--min-area", type=float, default=0.003, help="smallest block, share")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    grid = ROOT / "grids" / f"foam_{a.id}.json"
    if grid.exists() and not a.force:
        raise SystemExit(f"{grid} exists: use --force")
    entry = next(p for p in cfg["pictures"] if p["key"] == a.picture)
    sc, cp = cfg["sim"], cfg["palettes"]
    n, aspect = sc["colours"], sc["aspect"]
    img = Image.open(ROOT / entry["file"]).convert("RGB")
    if entry.get("crop"):
        x0, y0, x1, y1 = entry["crop"]
        W, H = img.size
        img = img.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
    img = pictures.fit_aspect(img, aspect, entry.get("anchor", (0.5, 0.5)))
    w0, h0 = a.res, round(a.res / aspect)
    small = img.filter(ImageFilter.GaussianBlur(0.5 * img.size[0] / w0)).resize((w0, h0))
    x = np.asarray(small, dtype=float).reshape(-1, 3)
    choice = pictures.choice(entry, cp)
    if choice == "present":
        pal, lab = pictures.kmeans(x, n, sc["seed"])
    else:
        share = entry.get("share", cp[f"{choice}_share"])
        pal, lab, _ = pictures.contrast_palette(x, n, sc["seed"], cp["clusters"], share)
    col = np.kron(lab.reshape(h0, w0), np.ones((a.up, a.up), dtype=np.int64))
    for _ in range(2):
        col = majority(col, n, a.up // 2 + 1)
    col = merge_small(col, int(a.min_area * col.size))
    regions = components(col)
    n_reg = int(regions.max()) + 1
    region_colour = np.zeros(n_reg, dtype=np.int64)
    region_colour[regions.ravel()] = col.ravel()
    foam = make_foam(
        a.cells, aspect, a.seed, relax=5, width=640, regions=regions.astype(np.uint16), min_sites=1
    )
    target = region_colour[foam.site_region]
    scr = random_scramble(foam, target, n, a.moves, np.random.default_rng(a.seed + 1000))
    np.save(grid.with_name(f"foam_{a.id}_regions.npy"), regions.astype(np.uint16))
    data = foam.to_json() | {"seed": a.seed, "regions_file": f"foam_{a.id}_regions.npy"}
    grid.write_text(json.dumps(data, indent=1) + "\n")
    level = {
        "id": a.id,
        "title": "",
        "foam": f"grids/foam_{a.id}.json",
        "colours": ["#" + bytes(int(v) for v in c).hex() for c in pal],
        "picture": {"regions": region_colour.tolist()},
        "scramble": [{"group": g, "pair": [p, q]} for g, p, q in scr],
        "reward": {
            "image": entry["file"],
            "crop": entry.get("crop"),
            "anchor": list(entry.get("anchor", (0.5, 0.5))),
            "artist": entry.get("artist", ""),
            "title": entry.get("title", ""),
        },
    }
    (ROOT / "levels" / "foam" / f"{a.id}.json").write_text(json.dumps(level, indent=1) + "\n")
    # preview: the blocks in their colours with the cell sides, next to the scrambled start
    out = ROOT / "runs" / "painting_level"
    out.mkdir(parents=True, exist_ok=True)
    pw, ph = 800, round(800 / aspect)
    cells = foam.raster(pw, ph)
    line = np.zeros((ph, pw), bool)
    line[:, :-1] |= cells[:, :-1] != cells[:, 1:]
    line[:-1, :] |= cells[:-1, :] != cells[1:, :]
    sheet = Image.new("RGB", (2 * pw + 10, ph), (28, 28, 34))
    start = target
    for g, p, q in scr:
        start = foam.move(start, g, p, q)
    for k, c in enumerate((target, start)):
        rgb = pal[c][cells]
        rgb[line] = (16, 16, 20)
        sheet.paste(Image.fromarray(rgb.astype(np.uint8)), (k * (pw + 10), 0))
    sheet.save(out / f"{a.id}_preview.png")
    deg = np.bincount(foam.edges.ravel(), minlength=foam.n_cells)
    sizes = np.bincount(foam.site_region, minlength=n_reg)
    print(f"{a.id}: {entry['key']} ({choice}), {n_reg} blocks, {foam.n_cells} cells", end=" ")
    print(f"({sizes.min()}-{sizes.max()} per block), {len(foam.edges)} edges,", end=" ")
    print(f"{deg.min()}-{deg.max()} neighbours, {foam.n_groups} groups")
    print("scramble:", ", ".join(f"g{g} {p}|{q}" for g, p, q in scr))
    print("preview:", (out / f"{a.id}_preview.png").relative_to(ROOT))


if __name__ == "__main__":
    main()

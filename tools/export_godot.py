"""Export the foam levels to the Godot project (game/data), which only loads data.

    python tools/export_godot.py [--src levels/candidates] [--levels F01 P01 ...] [--width 1280]

Per level (<src>/<id>.json and its foam; src defaults to levels/foam):
  game/data/levels/<id>.json        sites, edges, groups, points on every side, colours, target,
                                    start, par, reward
  game/data/levels/<id>.labels.gz   cell index of every pixel, uint16 little-endian, row by
                                    row, gzip (2.4 MB raw, about 60 kB packed)
  game/data/levels/<id>.shapes.gz   outline of every cell (48 points), for the moving cells
  game/data/paintings/<id>.jpg      the reward scan cut to the level's window (painting levels)
  game/tests/golden/<id>.json       colours after each move of the reversed scramble
and game/data/index.json, the ordered list of levels, and game/data/credits.json (painter,
title, date, collection and licence of each painting, in level order). Files are rewritten at
every run: they are derived data, the sources are levels/ and grids/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHAPE_POINTS = 48  # points on the outline of a cell
sys.path.insert(0, str(ROOT / "toy"))

from mosaic_model.foam import FoamLevel, load_foam_level, outlines  # noqa: E402


def side_points(labels: np.ndarray, edges: np.ndarray, aspect: float, k: int = 8) -> list:
    """Up to k points on the side shared by the two cells of every edge (foam coordinates),
    taken from the raster: used to find the side nearest to the pointer. An edge whose cells
    do not touch on this raster gets no point: it cannot be pointed at, but still swaps."""
    h, w = labels.shape
    lab = labels.astype(np.int64)
    n = int(lab.max()) + 1
    keys, xs, ys = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        p, q = lab[: h - dy, : w - dx], lab[dy:, dx:]
        yy, xx = np.nonzero(p != q)
        keys.append(np.minimum(p[yy, xx], q[yy, xx]) * n + np.maximum(p[yy, xx], q[yy, xx]))
        xs.append((xx + 0.5 + 0.5 * dx) / w * aspect)
        ys.append((yy + 0.5 + 0.5 * dy) / h)
    key, x, y = (np.concatenate(v) for v in (keys, xs, ys))
    order = np.argsort(key, kind="stable")
    key, x, y = key[order], x[order], y[order]
    out = []
    for u, v in edges:
        i0, i1 = np.searchsorted(key, [u * n + v, u * n + v + 1])
        if i1 == i0:  # neighbours on the coarser raster of the generator only: no side here
            out.append([])
            continue
        pick = np.unique(np.linspace(i0, i1 - 1, k).round().astype(int))
        out.append([c for i in pick for c in (round(float(x[i]), 4), round(float(y[i]), 4))])
    return out


def source_date(image: str) -> str:
    """Date of a painting as written in assets/paintings/SOURCES.md after its title
    ("- file.jpg: Painter, *Title*, 1890-92, ..."); "" if there is none."""
    text = (ROOT / "assets" / "paintings" / "SOURCES.md").read_text()
    m = re.search(rf"(?ms)^- {re.escape(Path(image).name)}:(.*?)(?=^- |\Z)", text)
    if not m:
        return ""
    entry = " ".join(m.group(1).split())
    d = re.search(r"\*(?: \([^)]*\))?,\s*((?:c\. |ca\. )?\d{4}(?:[–-]\d{2,4})?)", entry)
    return d.group(1).replace("ca. ", "c. ") if d else ""


COLLECTIONS = [  # name as written in SOURCES.md -> name in the credits
    ("Cleveland Museum of Art", "The Cleveland Museum of Art"),
    ("Metropolitan Museum of Art", "The Metropolitan Museum of Art"),
    ("National Gallery of Art", "National Gallery of Art, Washington"),
    ("rawpixel", "rawpixel"),
]


def source_credit(image: str) -> str:
    """Collection and licence of a painting from its entry in assets/paintings/SOURCES.md,
    such as "National Gallery of Art, Washington, CC0"; "" if there is no entry."""
    text = (ROOT / "assets" / "paintings" / "SOURCES.md").read_text()
    m = re.search(rf"(?ms)^- {re.escape(Path(image).name)}:(.*?)(?=^- |\Z)", text)
    if not m:
        return ""
    entry = " ".join(m.group(1).split())
    where = next((name for key, name in COLLECTIONS if key in entry), "")
    licence = "CC0" if "CC0" in entry else "public domain" if "open access" in entry.lower() else ""
    return ", ".join(x for x in (where, licence) if x)


def level_data(level: FoamLevel, width: int) -> tuple[dict, np.ndarray, list[list[int]]]:
    """The level as plain data, its raster of cell indices, and the golden states."""
    foam = level.foam
    height = round(width / foam.aspect)
    labels = foam.raster(width, height).astype("<u2")
    start = level.scrambled()
    states, col = [], start
    for g, a, b in reversed(level.scramble):
        col = foam.move(col, g, a, b)
        states.append(col.tolist())
    if not (col == level.target).all():
        raise ValueError(f"{level.id}: the reversed scramble does not give the picture")
    reward = level.reward or {}
    data = {
        "id": level.id,
        "aspect": foam.aspect,
        "sites": [[round(float(x), 5), round(float(y), 5)] for x, y in foam.sites],
        "edges": foam.edges.tolist(),
        "group": [int(g) for g in foam.group],
        "sides": side_points(labels, foam.edges, foam.aspect),
        "colours": level.colours,
        "target": level.target.tolist(),
        "start": start.tolist(),
        "par": level.par,
        # the reversed scramble, for the hints: one shortest way from the start to the picture
        "solution": [
            [int(g), int(min(a, b)), int(max(a, b))] for g, a, b in reversed(level.scramble)
        ],
        "labels": {"file": f"{level.id}.labels.gz", "width": width, "height": height},
        "shapes": {"file": f"{level.id}.shapes.gz", "points": SHAPE_POINTS},
        "painting": f"{level.id}.jpg" if reward.get("image") else "",
        "artist": reward.get("artist", ""),
        "date": source_date(reward["image"]) if reward.get("image") else "",
        "title": reward.get("title", "") or level.title,
    }
    golden = {
        "id": level.id,
        "start": start.tolist(),
        "moves": [[g, a, b] for g, a, b in reversed(level.scramble)],
        "states": states,
    }
    return data, labels, golden


def painting(level: FoamLevel, width: int):
    """The reward scan: the level's crop, the largest window of the foam's format, resized."""
    from PIL import Image

    import pictures

    rw = level.reward
    img = Image.open(ROOT / rw["image"]).convert("RGB")
    if rw.get("crop"):
        x0, y0, x1, y1 = rw["crop"]
        W, H = img.size
        img = img.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
    img = pictures.fit_aspect(img, level.foam.aspect, tuple(rw.get("anchor") or (0.5, 0.5)))
    return img.resize((width, round(width / level.foam.aspect)), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--levels",
        nargs="*",
        help="level ids in order (default: the order of <src>/ladder.toml, else sorted)",
    )
    ap.add_argument("--width", type=int, default=1280, help="width of the raster of cells")
    ap.add_argument("--src", default="levels/foam", help="folder of level JSON files")
    args = ap.parse_args()

    src = ROOT / args.src
    ids = args.levels or sorted(p.stem for p in src.glob("*.json"))
    if not args.levels and (src / "ladder.toml").exists():  # the order of the ladder file
        spec = tomllib.loads((src / "ladder.toml").read_text())
        ids = [lv["id"] for lv in spec.get("mondrian", []) + spec.get("painting", [])]
    out = ROOT / "game" / "data"
    gold = ROOT / "game" / "tests" / "golden"
    for d in (out / "levels", out / "paintings", gold):
        for old in d.glob("*") if d.exists() else []:  # derived data: start clean
            if old.suffix in (".json", ".gz", ".jpg"):
                old.unlink()
        d.mkdir(parents=True, exist_ok=True)

    index, credits = [], []
    for lid in ids:
        level = load_foam_level(src / f"{lid}.json")
        data, labels, golden = level_data(level, args.width)
        (out / "levels" / f"{lid}.json").write_text(json.dumps(data, separators=(",", ":")))
        packed = gzip.compress(labels.tobytes(), mtime=0)
        (out / "levels" / f"{lid}.labels.gz").write_bytes(packed)
        # outline of every cell: SHAPE_POINTS points, x then y, uint16 (0..65535 = the picture)
        shape = outlines(labels, level.foam.n_cells, SHAPE_POINTS) / labels.shape[::-1]
        packed = gzip.compress(np.round(shape * 65535).astype("<u2").tobytes(), mtime=0)
        (out / "levels" / f"{lid}.shapes.gz").write_bytes(packed)
        (gold / f"{lid}.json").write_text(json.dumps(golden, separators=(",", ":")))
        if data["painting"]:
            painting(level, args.width).save(out / "paintings" / data["painting"], quality=88)
        index.append({"id": lid, "kind": "painting" if data["painting"] else "learning"})
        if data["painting"]:
            image = (level.reward or {})["image"]
            credits.append(
                {k: data[k] for k in ("artist", "title", "date")} | {"source": source_credit(image)}
            )
        hidden = sum(not s for s in data["sides"])
        print(
            f"{lid}: {level.foam.n_cells} cells, {len(level.foam.edges)} edges, "
            f"{level.foam.n_groups} groups, par {level.par}"
            + (f", {hidden} edges without a side on the raster" if hidden else "")
        )
    (out / "index.json").write_text(json.dumps({"levels": index}, indent=1))
    (out / "credits.json").write_text(json.dumps({"paintings": credits}, indent=1) + "\n")


if __name__ == "__main__":
    main()

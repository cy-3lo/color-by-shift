"""Make the foam grids and click-to-shift levels (mosaic_model.foam).

    python tools/make_foam.py [--only F03] [--force]

Reads levels/foam/pictures.toml: each level is a picture made of polygons (a pentagon, a
star ...), a number of cells, a seed and a number of scrambling moves. For each level writes
grids/foam_<id>.json (a foam whose cells follow the outlines of the shapes: sites,
neighbours, group of each edge) and levels/foam/<id>.json (one colour per region, scrambled
by random moves that each swap at least two pairs). Existing files are kept unless --force.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

import numpy as np

from mosaic_model.foam import make_foam, random_scramble

ROOT = Path(__file__).resolve().parents[1]


def polygon(spec: dict) -> list[list[float]]:
    """Corners of a shape of pictures.toml."""
    if "polygon" in spec:
        return [[float(x), float(y)] for x, y in spec["polygon"]]
    if "rect" in spec:
        x0, y0, x1, y1 = spec["rect"]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    cx, cy = spec["at"]
    rot = math.radians(spec.get("rot", -90))
    if "regular" in spec:
        k = int(spec["regular"])
        radii = [spec["r"]] * k
    else:
        k = 2 * int(spec["star"])
        radii = [spec["r"], spec["inner"]] * int(spec["star"])
    return [
        [cx + r * math.cos(rot + 2 * math.pi * i / k), cy + r * math.sin(rot + 2 * math.pi * i / k)]
        for i, r in enumerate(radii)
    ]


def write(path: Path, data: dict, force: bool) -> bool:
    if path.exists() and not force:
        print(f"kept {path.relative_to(ROOT)} (use --force to rebuild)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="one level id")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    spec = tomllib.loads((ROOT / "levels" / "foam" / "pictures.toml").read_text())
    for lv in spec["level"]:
        if a.only and lv["id"] != a.only:
            continue
        shapes = [polygon(s) for s in lv["shapes"]]
        foam = make_foam(lv["cells"], spec["aspect"], lv["seed"], lv.get("relax", 5), shapes=shapes)
        rel = f"grids/foam_{lv['id']}.json"
        data = foam.to_json() | {"seed": lv["seed"]}
        if not write(ROOT / rel, data, a.force):
            continue
        cols = lv["colours"]
        used = sorted(set(cols))
        target = np.array([used.index(c) for c in cols])[foam.site_region]
        rng = np.random.default_rng(lv["seed"] + 1000)
        scr = random_scramble(foam, target, len(used), lv["moves"], rng)
        level = {
            "id": lv["id"],
            "title": "",
            "foam": rel,
            "colours": [spec["palette"][c] for c in used],
            "picture": {"regions": [used.index(c) for c in cols]},
            "scramble": [{"group": g, "pair": [x, y]} for g, x, y in scr],
        }
        write(ROOT / "levels" / "foam" / f"{lv['id']}.json", level, True)
        deg = np.bincount(foam.edges.ravel(), minlength=foam.n_cells)
        per = np.bincount(foam.site_region, minlength=foam.n_regions).tolist()
        print(
            f"{lv['id']}: {foam.n_cells} cells {per} per region, {len(foam.edges)} edges,", end=" "
        )
        print(f"{deg.min()}-{deg.max()} neighbours, {foam.n_groups} groups;", end=" ")
        print("scramble", ", ".join(f"g{g} {x}|{y}" for g, x, y in scr))


if __name__ == "__main__":
    main()

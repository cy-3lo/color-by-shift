"""Make a foam grid and its click-to-shift levels (mosaic_model.foam).

    python tools/make_foam.py [--name a] [--cells 150] [--aspect 1.5] [--seed 1] [--relax 4]

Writes grids/foam_<name>.json (sites, neighbours, group of each edge) and the levels
levels/foam/F01.json ... (flag-like pictures, 2 to 5 colours, scrambled by 1 to 6 random
moves, each swapping at least two pairs). Existing files are not overwritten unless --force.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mosaic_model.foam import foam_picture, make_foam, random_scramble

ROOT = Path(__file__).resolve().parents[1]
COLOURS = ["#26466e", "#f0e2c4", "#d6604d", "#6fae8c", "#e9b44c"]
# id, picture, colours, moves
LEVELS = [
    ("F01", "bands_v", 2, 1),
    ("F02", "disc", 2, 2),
    ("F03", "bands_v", 3, 2),
    ("F04", "sun", 3, 3),
    ("F05", "quarters", 4, 3),
    ("F06", "rings", 4, 4),
    ("F07", "bands_d", 5, 5),
    ("F08", "rings", 5, 6),
]


def write(path: Path, data: dict, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists: use --force or another --name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="a")
    ap.add_argument("--cells", type=int, default=150)
    ap.add_argument("--aspect", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--relax", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    foam = make_foam(a.cells, a.aspect, a.seed, a.relax)
    deg = np.bincount(foam.edges.ravel(), minlength=foam.n_cells)
    sizes = [len(p[0]) for p in foam.pairs]
    print(f"{foam.n_cells} cells, {len(foam.edges)} edges, neighbours per cell", end=" ")
    print(f"{deg.min()}-{deg.max()} (mean {deg.mean():.2f}), {foam.n_groups} groups of {sizes}")
    rel = f"grids/foam_{a.name}.json"
    data = foam.to_json() | {"seed": a.seed, "relax": a.relax}
    write(ROOT / rel, data, a.force)
    rng = np.random.default_rng(a.seed + 1000)
    for id_, kind, n, moves in LEVELS:
        target = foam_picture(kind, foam, n)
        scr = random_scramble(foam, target, n, moves, rng)
        level = {
            "id": id_,
            "title": "",
            "foam": rel,
            "colours": COLOURS[:n],
            "picture": {"flag": kind},
            "scramble": [{"group": g, "pair": [x, y]} for g, x, y in scr],
        }
        write(ROOT / "levels" / "foam" / f"{id_}.json", level, a.force)
        print(id_, kind, n, "colours:", ", ".join(f"g{g} {x}|{y}" for g, x, y in scr))


if __name__ == "__main__":
    main()

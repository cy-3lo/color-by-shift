"""Demo levels with terrain (holes = walls, fixed cells) for the sandbox, written to
levels/sandbox/X*.json. Geometry parameters are in DEMOS below.

    python tools/terrain_levels.py

X01  Chris's mobile spiral attempt (S20261118-191619_mobile) with two disc obstacles.
X02  May-Leonard (T04) on a 128 x 128 torus cut into two rooms by two walls, each with a
     gate.
X03  Voter (T01) with zealot discs of both colours and a patch-size goal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mosaic_model import make_graph

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ROOT / "levels"

DEMOS = {
    "X01": {"discs": [[-20, 0, 5], [22, 10, 5]]},  # q, r, radius
    "X02": {"n": 128, "wall_rows": [0, 64], "gates": [[20, 4], [84, 4]]},  # (column, width)
    "X03": {
        "zealots": [
            [-30, 0, 2, 0],
            [30, -20, 2, 0],
            [0, 30, 2, 0],
            [30, 20, 2, 1],
            [-30, -25, 2, 1],
        ],
        "size_goal": [4000, 1e9],
    },
}


def hex_dist(dq: np.ndarray, dr: np.ndarray) -> np.ndarray:
    return (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2


def disc(g, q0: int, r0: int, rad: int) -> list[list[int]]:
    q, r = g.axial[:, 0], g.axial[:, 1]
    sel = hex_dist(q - q0, r - r0) <= rad
    return g.axial[sel].tolist()


def write(d: dict) -> Path:
    p = LEVELS / "sandbox" / f"{d['id']}.json"
    p.write_text(json.dumps(d, indent=1))
    print(f"wrote {p.relative_to(ROOT)}")
    return p


def x01() -> dict:
    d = json.loads((LEVELS / "sandbox" / "S20261118-191619_mobile.json").read_text())
    g = make_graph(d["grid"])
    holes = [c for q, r, rad in DEMOS["X01"]["discs"] for c in disc(g, q, r, rad)]
    d.update(
        id="X01",
        title="spiral attempt + mobility, two obstacles",
        terrain={"holes": holes},
        meta={"from": "tools/terrain_levels.py", "note": "do spirals anchor on holes?"},
    )
    return d


def x02() -> dict:
    p = DEMOS["X02"]
    d = json.loads((LEVELS / "T04_mayleonard.json").read_text())
    n = p["n"]
    d["grid"] = {"type": "hex", "n": n}
    g = make_graph(d["grid"])
    idx = np.arange(g.n_cells)
    row, col = idx // n, idx % n
    wall = np.isin(row, p["wall_rows"])
    for (c0, w), r0 in zip(p["gates"], p["wall_rows"], strict=True):
        wall &= ~((row == r0) & (col >= c0) & (col < c0 + w))
    d.update(
        id="X02",
        title="May-Leonard in two rooms with gates",
        terrain={"holes": g.axial[wall].tolist()},
        goals=[],
        meta={"from": "tools/terrain_levels.py", "note": "rooms coupled only through gates"},
    )
    return d


def x03() -> dict:
    p = DEMOS["X03"]
    d = json.loads((LEVELS / "T01_voter.json").read_text())
    g = make_graph(d["grid"])
    fixed = [[*c, colour] for q, r, rad, colour in p["zealots"] for c in disc(g, q, r, rad)]
    lo, hi = p["size_goal"]
    d.update(
        id="X03",
        title="voter with zealots: grow big patches of colour 0",
        terrain={"fixed": fixed},
        goals=[{"type": "size", "color": 0, "range": [lo, hi], "duration": 5.0}],
        meta={"from": "tools/terrain_levels.py", "note": "zealots of both colours"},
    )
    return d


def main() -> None:
    for make in (x01, x02, x03):
        write(make())


if __name__ == "__main__":
    main()

"""Levels on a large, living, isotropic medium: few walls (a few % of cells), sources as small
fixed patches, several reactions, regional (zone) goals. Written to levels/sandbox/X07-X09.

    python tools/living_levels.py build
    python tools/screen.py levels/sandbox/X07.json      # every static arrangement

X07 Bloom      contact process: red and blue are born into white and die back to white;
               sources hold clouds whose reach grows near the survival threshold; swaps move
               colours between the fast and slow death slots and the competition slot.
X08 Rooms      cyclic dominance with mobility; two walls cut the grid into a large west room
               and two small east rooms joined by gaps.
X09 Tug        two thriving species fed by sources at opposite sides; the competition blocks
               set which way the broad red-blue boundary drifts; goal band in a middle zone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "levels" / "sandbox"
W, H = 96, 64
COLOURS3 = ["#F4F1EA", "#D7263D", "#3A7BD5"]  # white, red, blue
NAMES3 = ["white", "red", "blue"]
LEGEND = {"#": "wall", ".": "random", "R": [1, "fixed"], "B": [2, "fixed"]}

P = {
    "X07": {
        "fractions": [0.8, 0.1, 0.1],
        "burn_in": 60,
        "birth": 0.5,  # per edge, each species; lattice threshold death ~1.95, mean field 3
        "death": [2.6, 1.6, 2.1],  # slots: red, blue, white (white -> white does nothing)
        "compete": 0.3,  # B|R -> B|B
        "red_source": (32, 8, 2),  # row, col, radius
        "blue_source": (52, 84, 2),
        "walls": [("col", 60, 0, 40), ("row", 44, 20, 44)],
        "zones": {"G": (22, 42, 26, 40), "K": (44, 61, 74, 94)},
        "goals": [{"type": "zone", "zone": "G", "color": 1, "range": [0.25, 1.0], "duration": 30}],
        "fail": [{"type": "zone", "zone": "K", "color": 2, "range": [0.0, 0.05]}],
    },
    "X08": {
        "fractions": [1 / 3, 1 / 3, 1 / 3],
        "burn_in": 60,
        "invade": 1.0,
        "exchange": [2.0, 0.3, 0.3],
        "walls": [("col", 48, 0, 63, (28, 35)), ("row", 32, 49, 95, (70, 75))],  # with a gap
        "zones": {"W": (0, 63, 0, 47), "N": (0, 31, 49, 95), "S": (33, 63, 49, 95)},
    },
    "X09": {
        "fractions": [0.4, 0.3, 0.3],
        "halves": [[0.3, 0.7, 0.0], [0.3, 0.0, 0.7]],  # start: red left half, blue right half
        "burn_in": 0,
        "birth": 0.5,
        "death": [0.5, 0.8],  # the two death slots (red starts in the slow one)
        "compete": 0.5,  # R|B -> R|R and B|R -> B|B, symmetric
        "sources_every": 8,  # a fixed cell every n rows on the left (red) and right (blue)
        "walls": [("col", 30, 20, 30), ("col", 66, 34, 44)],
        "zones": {"M": (0, 63, 56, 68), "E": (0, 63, 82, 94)},
        "goals": [{"type": "zone", "zone": "M", "color": 1, "range": [0.3, 0.6], "duration": 30}],
        "fail": [{"type": "zone", "zone": "E", "color": 1, "range": [0.25, 1.0]}],
    },
}


def grid() -> list[list[str]]:
    return [["."] * W for _ in range(H)]


def draw_walls(g, walls) -> None:
    for w in walls:
        kind, fixed, a, b = w[:4]
        gap = w[4] if len(w) > 4 else (-1, -2)
        for k in range(a, b + 1):
            if gap[0] <= k <= gap[1]:
                continue
            if kind == "col":
                g[k][fixed] = "#"
            else:
                g[fixed][k] = "#"


def draw_disc(g, r0: int, c0: int, rad: int, ch: str) -> None:
    for r in range(max(0, r0 - rad), min(H, r0 + rad + 1)):
        for c in range(max(0, c0 - rad), min(W, c0 + rad + 1)):
            if (r - r0) ** 2 + (c - c0) ** 2 <= rad * rad + 1 and g[r][c] == ".":
                g[r][c] = ch


def zone_rects(zones: dict) -> dict:
    """Zones as map rectangles (rows r0..r1, columns c0..c1)."""
    return {name: {"rect": list(rc)} for name, rc in zones.items()}


def finish(lid: str, title: str, g, p: dict, reactions, locked, extra_legend=None) -> dict:
    zs = zone_rects(p["zones"])
    rows = ["".join(r) for r in g]
    walls = sum(r.count("#") for r in rows)
    return {
        "id": lid,
        "title": title,
        "palette": 3,
        "colours": COLOURS3,
        "names": NAMES3,
        "map": {"rows": rows, "legend": {**LEGEND, **(extra_legend or {})}, "zones": zs},
        "reactions": reactions,
        "locked": locked,
        "start": {"seed": 1, "fractions": p["fractions"], "burn_in_sweeps": p["burn_in"]},
        "goals": p.get("goals", []),
        "fail": p.get("fail", []),
        "meta": {"from": "tools/living_levels.py", "walls_fraction": round(walls / (W * H), 4)},
    }


def lock_all_but(reactions, free: set) -> list:
    return [
        [i, side, k]
        for i, r in enumerate(reactions)
        for side in ("in", "out")
        for k in range(len(r[side]))
        if (i, side, k) not in free
    ]


def x07() -> dict:
    p = P["X07"]
    g = grid()
    draw_walls(g, p["walls"])
    draw_disc(g, *p["red_source"], "R")
    draw_disc(g, *p["blue_source"], "B")
    rx = [
        {"shape": "edge", "in": [1, 0], "out": [1, 1], "rate": p["birth"]},
        {"shape": "edge", "in": [2, 0], "out": [2, 2], "rate": p["birth"]},
        {"shape": "cell", "in": [1], "out": [0], "rate": p["death"][0]},
        {"shape": "cell", "in": [2], "out": [0], "rate": p["death"][1]},
        {"shape": "cell", "in": [0], "out": [0], "rate": p["death"][2]},
        {"shape": "edge", "in": [2, 1], "out": [2, 2], "rate": p["compete"]},
    ]
    free = {(2, "in", 0), (3, "in", 0), (4, "in", 0)}
    return finish(
        "X07", "Bloom: sources near the survival threshold", g, p, rx, lock_all_but(rx, free)
    )


def x08() -> dict:
    p = P["X08"]
    g = grid()
    draw_walls(g, p["walls"])
    k = p["invade"]
    e = p["exchange"]
    rx = [
        {"shape": "edge", "in": [1, 0], "out": [1, 1], "rate": k},
        {"shape": "edge", "in": [2, 1], "out": [2, 2], "rate": k},
        {"shape": "edge", "in": [0, 2], "out": [0, 0], "rate": k},
        {"shape": "edge", "in": [1, 0], "out": [0, 1], "rate": e[0]},
        {"shape": "edge", "in": [2, 1], "out": [1, 2], "rate": e[1]},
        {"shape": "edge", "in": [0, 2], "out": [2, 0], "rate": e[2]},
    ]
    free = {(3, "in", 0), (3, "out", 1), (4, "in", 0), (4, "out", 1), (5, "in", 0), (5, "out", 1)}
    return finish(
        "X08", "Rooms: cyclic dominance in rooms joined by gaps", g, p, rx, lock_all_but(rx, free)
    )


def x09() -> dict:
    p = P["X09"]
    g = grid()
    for r in range(H):
        for c in range(W):
            g[r][c] = "<" if c < W // 2 else ">"
    draw_walls(g, p["walls"])
    for r in range(2, H - 1, p["sources_every"]):
        g[r][0] = "R"
        g[r][W - 1] = "B"
    rx = [
        {"shape": "edge", "in": [1, 0], "out": [1, 1], "rate": p["birth"]},
        {"shape": "edge", "in": [2, 0], "out": [2, 2], "rate": p["birth"]},
        {"shape": "cell", "in": [1], "out": [0], "rate": p["death"][0]},
        {"shape": "cell", "in": [2], "out": [0], "rate": p["death"][1]},
        {"shape": "edge", "in": [1, 2], "out": [1, 1], "rate": p["compete"]},
        {"shape": "edge", "in": [2, 1], "out": [2, 2], "rate": p["compete"]},
    ]
    free = {(2, "in", 0), (3, "in", 0)}
    extra = {"<": {"random": p["halves"][0]}, ">": {"random": p["halves"][1]}}
    return finish("X09", "Tug of war: hold the boundary", g, p, rx, lock_all_but(rx, free), extra)


def main() -> None:
    if sys.argv[1:] != ["build"]:
        print(__doc__)
        return
    for make in (x07, x08, x09):
        d = make()
        old = SANDBOX / f"{d['id']}.json"
        old.write_text(json.dumps(d, indent=1))
        print(f"wrote {old.relative_to(ROOT)}  walls {d['meta']['walls_fraction']:.1%}")


if __name__ == "__main__":
    main()

"""Maze level X04: a red source in a white sea; bring red to green without touching black.

    python tools/maze.py build          # writes levels/sandbox/X04_maze.json, prints the map
    python tools/maze.py bots [--seeds 20] [--delay 1.5]

Mechanics. Red invades white at rate k_sea (rooms flood). Corridors are paved with yellow
or blue; red crosses a corridor only if the gate reaction R|X -> R|R carries that colour.
The player has one move: swap the yellow block of the gate with the blue block parked on a
switched-off slot (k = 0), which opens blue corridors and closes yellow ones. Green and
black are fixed cells; red touching green wins, red touching black fails.

Intended solution (two switches, timing matters): keep yellow until red enters room 1, then
blue until red enters room 2, then yellow again. Corridor lengths (cells, front speed about
one cell per time unit at k = 1) are in SEGMENTS; the bots check that
  - never switching fails (yellow corridor from room 1 to black K2),
  - switching at once fails (blue corridor from room 0 to black K1),
  - dithering fails most of the time,
  - the event-driven player (reaction delay --delay) wins in most seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import GoalTracker, load_level, make_sim  # noqa: E402
from mosaic_model.level import map_start  # noqa: E402
from mosaic_model.observables import observe  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

PATH = ROOT / "levels" / "sandbox" / "X04_maze.json"
W, H = 48, 34
# colours: 0 white sea, 1 red, 2 green, 3 black, 4 yellow, 5 blue
COLOURS = ["#F4F1EA", "#D7263D", "#2E9E5B", "#1B1B1E", "#F2C14E", "#3A7BD5"]
NAMES = ["white", "red", "green", "black", "yellow", "blue"]
LEGEND = {
    "#": "wall",
    ".": 0,
    "0": 0,  # room tags (all white sea) for the bots
    "1": 0,
    "2": 0,
    "R": [1, "fixed"],
    "G": [2, "fixed"],
    "K": [3, "fixed"],
    "y": 4,
    "b": 5,
}
K_SEA, K_GATE = 2.0, 1.0
# (char, kind, row/col, start, end): "row" = cells (r, c0..c1), "col" = cells (r0..r1, c)
SEGMENTS = [
    ("0", "rect", (2, 7), (2, 8)),  # room 0: rows 2-7, cols 2-8
    ("1", "rect", (1, 8), (34, 41)),  # room 1
    ("2", "rect", (19, 25), (32, 41)),  # room 2
    ("y", "row", 3, (9, 33)),  # A: room 0 -> room 1, yellow, 25 cells
    ("b", "col", 4, (8, 30)),  # K1 corridor: room 0 -> black, blue, 23 + 7 cells
    ("b", "row", 30, (5, 11)),
    ("K", "row", 30, (12, 13)),
    ("y", "row", 2, (42, 46)),  # K2 corridor: room 1 -> black, yellow, 5 + 27 cells
    ("y", "col", 46, (3, 29)),
    ("K", "col", 46, (30, 31)),
    ("b", "col", 37, (9, 18)),  # B: room 1 -> room 2, blue, 10 cells
    ("y", "row", 22, (20, 31)),  # G corridor: room 2 -> green, yellow, 12 cells
    ("G", "row", 22, (18, 19)),
    ("b", "col", 36, (26, 33)),  # K3 corridor: room 2 -> black, blue, 8 + 7 cells
    ("b", "row", 33, (37, 43)),
    ("K", "row", 33, (44, 45)),
    ("R", "col", 2, (4, 5)),  # red source on the left wall of room 0
]


def build_map() -> list[str]:
    g = [["#"] * W for _ in range(H)]
    for ch, kind, a, b in SEGMENTS:
        if kind == "rect":
            for r in range(a[0], a[1] + 1):
                for c in range(b[0], b[1] + 1):
                    g[r][c] = ch
        elif kind == "row":
            for c in range(b[0], b[1] + 1):
                g[a][c] = ch
        else:
            for r in range(b[0], b[1] + 1):
                g[r][a] = ch
    return ["".join(r) for r in g]


def level_dict() -> dict:
    gate = [0, "in", 1]  # yellow block of the gate (reaction 1)
    shelf = [2, "in", 0]  # blue block parked on the switched-off slot
    reactions = [
        {"shape": "edge", "in": [1, 0], "out": [1, 1], "rate": K_SEA},
        {"shape": "edge", "in": [1, 4], "out": [1, 1], "rate": K_GATE},
        {"shape": "cell", "in": [5], "out": [5], "rate": 0.0},
    ]
    free = {(1, "in", 1), (2, "in", 0)}
    locked = [
        [i, side, k]
        for i, r in enumerate(reactions)
        for side in ("in", "out")
        for k in range(len(r[side]))
        if (i, side, k) not in free
    ]
    return {
        "id": "X04",
        "title": "maze: bring red to green, avoid black",
        "palette": 6,
        "colours": COLOURS,
        "names": NAMES,
        "map": {"rows": build_map(), "legend": LEGEND},
        "reactions": reactions,
        "locked": locked,
        "start": {"seed": 1, "burn_in_sweeps": 0},
        "goals": [{"type": "contact", "colors": [1, 2]}],
        "fail": [{"type": "contact", "colors": [1, 3]}],
        "meta": {
            "from": "tools/maze.py",
            "note": f"swap blocks {gate} and {shelf}: yellow / blue corridors",
            "solution": "yellow until red enters room 1, blue until room 2, then yellow",
        },
    }


# ---- bots ----------------------------------------------------------------------------------
SWAP = ((1, "in", 1), (2, "in", 0))


def play(level, seed: int, policy: str, delay: float, t_max: float, dt: float = 0.25):
    """Run one game; returns (status, t_end, switch times)."""
    sim = make_sim(level, seed=seed)
    _, _, tags = map_start(sim.graph, level.map, level.palette)
    room1, room2 = tags["1"], tags["2"]
    tr = GoalTracker(level.goals, level.fail)
    switches: list[float] = []
    blue = False  # gate currently on blue
    pending: float | None = None  # time at which the planned switch happens
    stage = 0
    while sim.t < t_max:
        sim.advance(dt)
        col = sim.colours
        if policy == "planned":
            if stage == 0 and (col[room1] == 1).any():
                stage, pending = 1, sim.t + delay
            elif stage == 2 and (col[room2] == 1).any():
                stage, pending = 3, sim.t + delay
            if pending is not None and sim.t >= pending:
                sim.swap(*SWAP)
                blue = not blue
                switches.append(sim.t)
                pending = None
                stage += 1
        elif policy == "blue_now" and not switches:
            sim.swap(*SWAP)
            switches.append(sim.t)
        elif policy.startswith("dither"):
            period = float(policy[6:] or 1.0)
            if int(sim.t / period) % 2 != blue:
                sim.swap(*SWAP)
                blue = not blue
                switches.append(sim.t)
        st = tr.update(sim.t, sim.counts(), observe(sim, size=False))
        if st != "running":
            return st, sim.t, switches
    return "timeout", sim.t, switches


def bots(seeds: int, delay: float, t_max: float) -> None:
    level = load_level(PATH)
    policies = ["none", "blue_now", "dither1", "dither4", "planned"]
    d = new_run_dir("bots", "X04", {"seeds": seeds, "delay": delay, "t_max": t_max})
    rows, lines = [], [f"X04 maze bots, {seeds} seeds, reaction delay {delay}, t_max {t_max}"]
    lines.append("policy     won  failed  timeout   mean t_end  switches")
    for pol in policies:
        res = [play(level, 1000 + s, pol, delay, t_max) for s in range(seeds)]
        for s, (st, t, sw) in enumerate(res):
            rows.append([pol, s, st, round(t, 2), len(sw), " ".join(f"{x:.1f}" for x in sw)])
        n = {k: sum(r[0] == k for r in res) for k in ("won", "failed", "timeout")}
        tm = np.mean([r[1] for r in res])
        nsw = np.mean([len(r[2]) for r in res])
        lines.append(
            f"{pol:9s} {n['won']:4d} {n['failed']:7d} {n['timeout']:8d}   {tm:10.1f}  {nsw:8.1f}"
        )
        print(lines[-1], flush=True)
    write_csv(d / "bots.csv", ["policy", "seed", "status", "t_end", "n_switch", "times"], rows)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"saved {d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "bots"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--delay", type=float, default=1.5, help="planned bot reaction time")
    ap.add_argument("--t-max", type=float, default=300.0)
    a = ap.parse_args()
    if a.cmd == "build":
        d = level_dict()
        PATH.write_text(json.dumps(d, indent=1))
        print(
            "\n".join(("" if r % 2 == 0 else " ") + row for r, row in enumerate(d["map"]["rows"]))
        )
        print(f"wrote {PATH.relative_to(ROOT)}")
    else:
        bots(a.seeds, a.delay, a.t_max)


if __name__ == "__main__":
    main()

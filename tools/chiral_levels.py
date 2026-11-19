"""Chiral wall-following levels: the vertex reaction needs stone (fixed walls) at one corner,
so handedness picks a direction along every wall.

    python tools/chiral_levels.py build
    python tools/chiral_levels.py bots [--seeds 20] [--delay 1.5]

Vine  (R, W, S) -> (R, R, S), chiral, read clockwise: red grows into the white cell next to
      it along a stone wall, keeping the wall on its right. Swapping the two input blocks
      R and W gives (W, R, S) -> (R, R, S), the mirror rule (wall on its left). In a corridor
      one cell wide (stone on both sides) both orientations match: such gaps are two-way and
      join the loops of different walls (an island and the outer wall).
Walker (R, W, S) -> (W, R, S): the red cell itself moves along the wall (red conserved);
      in a one-cell gap it random-walks. Here a single swap (inputs or outputs) gives an
      identity reaction, so the walker stops; swapping both pairs reverses it.

Targets (green to touch, black to avoid) sit one cell off the wall, so the red line passes
beside them; a target in the wall lining would block it.

X05  vine: touch both greens (one on the top wall, one on the far side of an island reached
     through a one-cell gap), avoid the three blacks.
X06  walker on the same map: not built. On this map the walker cannot win (bots: every
     policy fails): in the gap it random-walks and leaves by either end. A walker level needs
     its own map; a gap end is a valve (one hand pushes the walker back in, the other carries
     it away along the wall).
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

SANDBOX = ROOT / "levels" / "sandbox"
W, H = 44, 26
# colours: 0 white floor, 1 red, 2 stone, 3 green, 4 black
COLOURS = ["#F4F1EA", "#D7263D", "#8A8D91", "#2E9E5B", "#1B1B1E"]
NAMES = ["white", "red", "stone", "green", "black"]
LEGEND = {".": 0, "g": 0, "S": [2, "fixed"], "R": [1, "fixed"], "r": 1, "G": [3, "fixed"]}
LEGEND["K"] = [4, "fixed"]
ISLAND = ((2, 8), (28, 35))  # rows, cols of the stone island; row 1 above it is the gap
CELLS = {  # (row, col): character
    "source": ((1, 22), "R"),
    "green_top": ((2, 12), "G"),
    "green_island": ((5, 37), "G"),
    "black_left": ((14, 2), "K"),
    "black_island": ((7, 26), "K"),
    "black_right": ((2, 40), "K"),
}


def build_map(source: str = "R") -> list[str]:
    g = [["."] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if r in (0, H - 1) or c in (0, W - 1):
                g[r][c] = "S"
    (r0, r1), (c0, c1) = ISLAND
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            g[r][c] = "S"
    for c in range(c0 - 1, c1 + 1):
        g[r0 - 1][c] = "g"  # the one-cell gap above the island and its approach (tag for bots)
    for name, ((r, c), ch) in CELLS.items():
        g[r][c] = source if name == "source" else ch
    return ["".join(row) for row in g]


def level(kind: str) -> dict:
    vine = kind == "vine"
    out = [1, 1, 2] if vine else [0, 1, 2]
    reactions = [{"shape": "vertex", "in": [1, 0, 2], "out": out, "rate": 1.0, "chiral": True}]
    locked = [[0, "in", 2], [0, "out", 2]] + ([[0, "out", 0], [0, "out", 1]] if vine else [])
    return {
        "id": "X05" if vine else "X06",
        "title": "vine: follow the wall, touch both greens"
        if vine
        else "walker: one red cell, touch both greens",
        "palette": 5,
        "colours": COLOURS,
        "names": NAMES,
        "map": {"rows": build_map("R" if vine else "r"), "legend": LEGEND},
        "reactions": reactions,
        "locked": locked,
        "start": {"seed": 1, "burn_in_sweeps": 0},
        "goals": [{"type": "contact", "colors": [1, 3], "all": True}],
        "fail": [{"type": "contact", "colors": [1, 4]}],
        "meta": {
            "from": "tools/chiral_levels.py",
            "note": "swap the two input blocks R and W to mirror the rule (wall on the other side)",
        },
    }


# ---- bots ----------------------------------------------------------------------------------
SWAPS = {  # the swaps that mirror the rule
    "X05": [((0, "in", 0), (0, "in", 1))],
    "X06": [((0, "in", 0), (0, "in", 1)), ((0, "out", 0), (0, "out", 1))],
}
POLICIES = {
    "none": [],
    "mirror_now": [("t", 0.0)],
    "gap_then_back": [("t", 0.0), ("tag", "g")],  # mirror at once, back when red enters gap
    "green_gap": [("touched", 1), ("tag", "g")],  # mirror after the first green, back at gap
    "dither1": "dither1",
    "dither4": "dither4",
}


def play(lv, seed: int, policy, delay: float, t_max: float, dt: float = 0.25):
    sim = make_sim(lv, seed=seed)
    swaps = SWAPS[lv.id]

    def mirror() -> None:
        for a, b in swaps:
            sim.swap(a, b)

    _, _, tags = map_start(sim.graph, lv.map, lv.palette)
    tr = GoalTracker(lv.goals, lv.fail)
    switches: list[float] = []
    step, pending = 0, None
    flipped = False
    while sim.t < t_max:
        if isinstance(policy, str):  # dither
            period = float(policy[6:])
            if int(sim.t / period) % 2 != flipped:
                mirror()
                flipped = not flipped
                switches.append(sim.t)
        elif step < len(policy):
            kind, arg = policy[step]
            if pending is None:
                if kind == "t" and sim.t >= arg:
                    pending = sim.t
                elif kind == "tag" and (sim.colours[tags[arg]] == 1).any():
                    pending = sim.t + delay
                elif kind == "touched" and len(tr._goals[0].touched) >= arg:
                    pending = sim.t + delay
            if pending is not None and sim.t >= pending:
                mirror()
                switches.append(sim.t)
                step, pending = step + 1, None
        sim.advance(dt)
        st = tr.update(sim.t, sim.counts(), observe(sim, size=False))
        if st != "running":
            return st, sim.t, switches
    return "timeout", sim.t, switches


def bots(seeds: int, delay: float, t_max: float) -> None:
    d = new_run_dir("bots", "chiral", {"seeds": seeds, "delay": delay, "t_max": t_max})
    rows, lines = [], []
    for lid in ("X05",):
        lv = load_level(SANDBOX / f"{lid}.json")
        lines.append(f"{lid}  ({seeds} seeds, delay {delay}, t_max {t_max})")
        lines.append("policy          won  failed  timeout   mean t_end  switches")
        for name, pol in POLICIES.items():
            res = [play(lv, 2000 + s, pol, delay, t_max) for s in range(seeds)]
            for s, (st, t, sw) in enumerate(res):
                rows.append([lid, name, s, st, round(t, 2), len(sw)])
            n = {k: sum(r[0] == k for r in res) for k in ("won", "failed", "timeout")}
            lines.append(
                f"{name:14s} {n['won']:4d} {n['failed']:7d} {n['timeout']:8d}   "
                f"{np.mean([r[1] for r in res]):10.1f}  {np.mean([len(r[2]) for r in res]):8.1f}"
            )
            print(lines[-1], flush=True)
    write_csv(d / "bots.csv", ["level", "policy", "seed", "status", "t_end", "n_switch"], rows)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"saved {d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "bots"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--t-max", type=float, default=200.0)
    a = ap.parse_args()
    if a.cmd == "build":
        for kind in ("vine",):
            d = level(kind)
            p = SANDBOX / f"{d['id']}.json"
            p.write_text(json.dumps(d, indent=1))
            print(f"wrote {p.relative_to(ROOT)}")
    else:
        bots(a.seeds, a.delay, a.t_max)


if __name__ == "__main__":
    main()

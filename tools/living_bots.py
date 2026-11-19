"""Bots for the living levels (X07, X09): one toggle (a swap of two blocks) played as
static0 / static1   never switch / switch once at t = 0,
dither<P>           toggle every P time units,
feedback            look at a zone fraction every `look` time units; after a reaction
                    delay, choose the arrangement that pushes it back toward the goal band
                    (the target colour's favoured arrangement below the band centre, the
                    other one above).

  python tools/living_bots.py levels/sandbox/X09.json [--seeds 10] [--delay 2]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import load_level, make_sim  # noqa: E402
from mosaic_model.goals import zone_fraction  # noqa: E402
from mosaic_model.observables import observe  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

TOGGLE = {"X09": ((2, "in", 0), (3, "in", 0)), "X07": ((2, "in", 0), (3, "in", 0))}


def play(args):
    path, policy, seed, delay, t_max, look = args
    lv = load_level(path)
    sim = make_sim(lv, seed=seed)
    tr = lv.tracker(sim.graph)
    goal = next(g for g in lv.goals if g["type"] == "zone")
    centre = 0.5 * sum(goal["range"])
    swap = TOGGLE[lv.id]
    state = 0  # 0 = level's own arrangement (target colour favoured), 1 = toggled
    switches = 0
    pending: tuple[float, int] | None = None
    if policy == "static1":
        sim.swap(*swap)
        state, switches = 1, 1
    st = "running"
    while sim.t < t_max:
        sim.advance(look)
        obs = {**observe(sim, size=False), "zones": tr.zones}
        st = tr.update(sim.t, sim.counts(), obs)
        if st != "running":
            break
        want = state
        if policy.startswith("dither"):
            want = int(sim.t / float(policy[6:])) % 2
        elif policy == "feedback":
            v = zone_fraction(obs, goal["zone"], goal["color"])
            target = 0 if v < centre else 1
            if pending is None and target != state:
                pending = (sim.t + delay, target)
            if pending is not None and sim.t >= pending[0]:
                want, pending = pending[1], None
        if want != state:
            sim.swap(*swap)
            state, switches = want, switches + 1
    return policy, seed, st, sim.t, switches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("level", type=Path)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--look", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=400.0)
    ap.add_argument("--policies", default="static0,static1,dither5,dither20,feedback")
    a = ap.parse_args()
    lv = load_level(a.level)
    pols = a.policies.split(",")
    d = new_run_dir("bots", lv.id, vars(a) | {"level": str(a.level)})
    jobs = [
        (str(a.level), p, 300 + s, a.delay, a.t_max, a.look) for p in pols for s in range(a.seeds)
    ]
    with mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1)) as pool:
        res = pool.map(play, jobs)
    lines = [f"{lv.id} bots: {a.seeds} seeds, delay {a.delay}, t_max {a.t_max}"]
    lines.append("policy       won  failed  running   mean t_end  switches")
    for p in pols:
        r = [x for x in res if x[0] == p]
        n = {k: sum(x[2] == k for x in r) for k in ("won", "failed", "running")}
        lines.append(
            f"{p:11s} {n['won']:4d} {n['failed']:7d} {n['running']:8d}   "
            f"{np.mean([x[3] for x in r]):10.1f}  {np.mean([x[4] for x in r]):8.1f}"
        )
    write_csv(d / "bots.csv", ["policy", "seed", "status", "t_end", "switches"], res)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

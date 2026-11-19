"""Screen a level: run every block arrangement reachable by swaps (distinct permutations of
the unlocked blocks), held fixed with no further switching, over several seeds, and report
the goal outcome of each.

    python tools/screen.py levels/sandbox/X07.json [--seeds 4] [--t-max 300] [--workers 8]

Reading the table: if some static arrangement wins in most seeds, the level needs no timing
(one decision); if none wins, a solution needs switches; if outcomes split across seeds, the
level is a lottery at that arrangement. CSV and summary in runs/screen/<level>_<time>/.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import load_level, make_sim  # noqa: E402
from mosaic_model.goals import zone_fraction  # noqa: E402
from mosaic_model.observables import observe  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402


def arrangements(level) -> tuple[list, list[tuple[int, ...]]]:
    rs = level.reaction_set()
    free = [b for b in rs.blocks() if b not in rs.locked]
    cols = [rs.get(b) for b in free]
    perms = sorted(set(itertools.permutations(cols)))
    perms.remove(tuple(cols))
    return free, [tuple(cols), *perms]  # the level's own arrangement first


def label(level, free, arr) -> str:
    rs = level.reaction_set()
    for b, c in zip(free, arr, strict=True):
        rs._set(b, c)
    return "; ".join(r.label().split("  k=")[0] for r in rs.reactions)


def run_one(args) -> tuple[int, int, str, float, list[float]]:
    path, ai, arr, seed, t_max, dt = args
    level = load_level(path)
    free, _ = arrangements(level)
    sim = make_sim(level, seed=seed)
    for b, c in zip(free, arr, strict=True):
        sim.reactions._set(b, c)
    sim.rebuild()
    tr = level.tracker(sim.graph)
    st = "running"
    while sim.t < t_max:
        sim.advance(dt)
        obs = observe(sim, size=tr.needs_size) if tr.needs_obs else None
        st = tr.update(sim.t, sim.counts(), obs)
        if st != "running":
            break
    obs = {**observe(sim, size=False), "zones": tr.zones}
    zf = [zone_fraction(obs, z, c) for z in sorted(tr.zones) for c in range(level.palette)]
    return ai, seed, st, sim.t, sim.fractions().tolist() + zf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("level", type=Path)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--t-max", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    level = load_level(a.level)
    free, arrs = arrangements(level)
    params = {"level": str(a.level), "seeds": a.seeds, "t_max": a.t_max, "free": str(free)}
    d = new_run_dir("screen", level.id, params)
    jobs = [
        (str(a.level), i, arr, 100 + s, a.t_max, a.dt)
        for i, arr in enumerate(arrs)
        for s in range(a.seeds)
    ]
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(run_one, jobs)
    rows, lines = [], []
    lines.append(f"{level.id}: {len(arrs)} arrangements x {a.seeds} seeds, t_max {a.t_max}")
    lines.append(f"unlocked blocks: {free}")
    zones = sorted(level.tracker().zones)
    zcols = [f"{z}:{c}" for z in zones for c in range(level.palette)]
    lines.append(
        "  #  won fail run  mean t   end fractions (all, " + ", ".join(zones) + ")   arrangement"
    )
    by: dict[int, list] = {}
    fr_by: dict[int, list] = {}
    for ai, seed, st, t, fr in res:
        by.setdefault(ai, []).append((st, t))
        fr_by.setdefault(ai, []).append(fr)
        rows.append([ai, seed, st, round(t, 2), *np.round(fr, 4)])
    order = sorted(by, key=lambda i: (-sum(s == "won" for s, _ in by[i]), i))
    for i in order:
        r = by[i]
        n = {k: sum(s == k for s, _ in r) for k in ("won", "failed", "running")}
        mark = "*" if i == 0 else " "
        mf = np.mean(fr_by[i], axis=0)
        p = level.palette
        groups = [mf[k : k + p] for k in range(0, len(mf), p)]
        frs = " | ".join(" ".join(f"{v:.2f}" for v in gr) for gr in groups)
        lines.append(
            f"{mark}{i:2d}  {n['won']:3d} {n['failed']:4d} {n['running']:3d}  "
            f"{np.mean([t for _, t in r]):6.1f}   {frs}   {label(level, free, arrs[i])}"
        )
    lines.append(f"(* = the level's own start arrangement; {time.perf_counter() - t0:.0f} s)")
    write_csv(
        d / "screen.csv",
        ["arrangement", "seed", "status", "t_end"]
        + [f"x{c}" for c in range(level.palette)]
        + zcols,
        rows,
    )
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {d}")


if __name__ == "__main__":
    main()

"""Zone test: can ONE reaction set keep three zones in three different regimes?

    python tools/zonetest.py [--w 60] [--rows 20] [--t-max 100] [--cuts]

World: open hex rectangle, three horizontal bands that touch (top to bottom), as in a
painting with sky, foam and sea:
  sky   starts as one colour (colour 2)
  foam  starts as a random speckle of colours 1 and 2 (--foam fractions)
  sea   starts as a random mix of the three colours (--sea fractions)
With --cuts the bands are separated by hermetic edges (sealed outlines).

Census: every set of three distinct edge reactions on 3 colours, all k = 1, identities
(A|B -> A|B) left out; colour names matter here, so there is no relabelling.

State of each zone at the end of the run:
  still   at least 98 % of its cells have their starting colour
  alive   cells keep changing (late change rate >= --alive per cell per unit time) and at
          least two colours hold 5 % each
  mono    one colour holds 98 % (and the zone is not still)
  frozen  change rate < --frozen, at least two colours at 5 %, not still (a jam)
  slow    anything else (still drifting)
Change rate: share of the zone's cells that differ between two snapshots one time unit apart,
averaged over the last quarter of the run.

Wanted ("solution"): sky still, foam still or frozen, sea alive.
Orbit of a set: the sets with the same input blocks and the same output blocks (what the
player can reach by swapping input <-> input and output <-> output).
Outputs in runs/zonetest/<time>/: sets.csv (one row per set), summary.txt.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from classify import reactions  # noqa: E402
from matchtest import flat, reaction_set, rebuild, text  # noqa: E402
from mosaic_model import Sim  # noqa: E402
from mosaic_model.graph import hex_rect  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

ZONES = ("sky", "foam", "sea")
_W: dict = {}  # world shared with the workers (fork)


def world(w: int, rows: int, foam, sea, cuts: bool, seed: int) -> dict:
    g = hex_rect(w, 3 * rows)
    band = (np.arange(g.n_cells) // w) // rows  # 0 sky, 1 foam, 2 sea (cell index r * w + c)
    rng = np.random.default_rng(seed)
    col = np.full(g.n_cells, 2)
    col[band == 1] = rng.choice(3, size=int((band == 1).sum()), p=foam)
    col[band == 2] = rng.choice(3, size=int((band == 2).sum()), p=sea)
    e = g.edges
    cut = e[band[e[:, 0]] != band[e[:, 1]]] if cuts else None
    return {"g": g, "band": band, "col": col, "cuts": cut}


def zone_state(start, end, rate, a) -> str:
    x = np.bincount(end, minlength=3) / len(end)
    if np.mean(start == end) >= 0.98:
        return "still"
    if x.max() >= 0.98:
        return "mono"
    two = (x >= 0.05).sum() >= 2
    if rate >= a["alive"] and two:
        return "alive"
    if rate < a["frozen"] and two:
        return "frozen"
    return "slow"


def run(args):
    i, rs, a = args
    w = _W
    sim = Sim(w["g"], reaction_set(rs, 3), w["col"], a["seed"], cuts=w["cuts"])
    t0 = 0.75 * a["t_max"]
    sim.advance(t0)
    rates = []
    prev = sim.colours.copy()
    while sim.t < a["t_max"] - 1e-9:
        sim.advance(1.0)
        c = sim.colours.copy()
        rates.append([float(np.mean((c != prev)[w["band"] == z])) for z in range(3)])
        prev = c
    rate = np.mean(rates, axis=0)
    res = []
    for z in range(3):
        m = w["band"] == z
        st = zone_state(w["col"][m], prev[m], rate[z], a)
        x = np.bincount(prev[m], minlength=3) / m.sum()
        res.append((st, x, float(rate[z])))
    return i, res


def main() -> None:
    global _W
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w", type=int, default=60)
    ap.add_argument("--rows", type=int, default=20, help="rows per zone")
    ap.add_argument("--t-max", type=float, default=100.0)
    ap.add_argument("--foam", default="0,0.35,0.65")
    ap.add_argument("--sea", default="0.5,0.2,0.3")
    ap.add_argument("--alive", type=float, default=0.02)
    ap.add_argument("--frozen", type=float, default=0.002)
    ap.add_argument("--cuts", action="store_true")
    ap.add_argument("--only", default="", help="'|'-separated sets to run instead of the census")
    ap.add_argument("--max-sets", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    foam = [float(v) for v in a.foam.split(",")]
    sea = [float(v) for v in a.sea.split(",")]
    _W = world(a.w, a.rows, foam, sea, a.cuts, a.seed)
    if a.only:
        from matchtest import parse_set

        sets = [parse_set(s) for s in a.only.split("|")]
    else:
        rx = [r for r in reactions("edge", 3) if r[1] != r[2]]
        sets = [tuple(sorted(c)) for c in itertools.combinations(rx, 3)]
        if a.max_sets:
            rng = np.random.default_rng(a.seed)
            sets = [sets[i] for i in sorted(rng.choice(len(sets), a.max_sets, replace=False))]
    d = new_run_dir("zonetest", "cuts" if a.cuts else "open", vars(a))
    par = {k: getattr(a, k) for k in ("t_max", "alive", "frozen", "seed")}
    jobs = [(i, s, par) for i, s in enumerate(sets)]
    print(f"{len(sets)} sets, world {a.w} x {3 * a.rows}, cuts {a.cuts}", flush=True)
    res: list = [None] * len(sets)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(run, jobs, chunksize=8)):
            res[i] = r
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(sets)} sets, {time.time() - t0:.0f} s", flush=True)
    triple = [tuple(z[0] for z in r) for r in res]
    good = [t[0] == "still" and t[1] in ("still", "frozen") and t[2] == "alive" for t in triple]
    rows = []
    for s, r, ok in zip(sets, res, good, strict=True):
        row = [text(s), int(ok)]
        for st, x, rate in r:
            row += [st, *[round(float(v), 4) for v in x], round(rate, 5)]
        rows.append(row)
    hdr = ["reactions", "solution"]
    for z in ZONES:
        hdr += [f"{z}_state", f"{z}_x0", f"{z}_x1", f"{z}_x2", f"{z}_rate"]
    write_csv(d / "sets.csv", hdr, rows)
    out = [
        f"zone test: {len(sets)} sets of 3 edge reactions, world {a.w} x {3 * a.rows}, "
        f"zones sky / foam {foam} / sea {sea}, cuts between zones: {a.cuts}, t_max {a.t_max:g} "
        f"({time.time() - t0:.0f} s)",
        f"solutions (sky still, foam still or frozen, sea alive): {sum(good)} "
        f"({np.mean(good):.2%})",
        "states (sky, foam, sea), most frequent:",
    ]
    for t, c in Counter(triple).most_common(15):
        out.append(f"  {c:6d}  {' / '.join(t)}")
    for zi, z in enumerate(ZONES):
        cz = Counter(t[zi] for t in triple)
        out.append(f"{z}: " + ", ".join(f"{k} {v}" for k, v in cz.most_common()))
    if not a.only:
        # orbits: what the player can reach by swapping blocks
        index = {s: i for i, s in enumerate(sets)}
        orb = defaultdict(list)
        for i, s in enumerate(sets):
            _, ins, outs = flat(s)
            orb[(tuple(sorted(ins)), tuple(sorted(outs)))].append(i)
        with_sol = {k: v for k, v in orb.items() if any(good[i] for i in v)}
        share = [np.mean([good[i] for i in v]) for v in with_sol.values()]
        kinds = [len({triple[i] for i in v}) for v in with_sol.values()]
        size = [len(v) for v in with_sol.values()]
        out.append(
            f"orbits: {len(orb)}; with at least one solution: {len(with_sol)}"
            + (
                f"; in those: sets per orbit median {np.median(size):.0f}, share of solutions "
                f"median {np.median(share):.1%} (min {min(share):.1%}, max {max(share):.1%}), "
                f"distinct (sky, foam, sea) outcomes per orbit median {np.median(kinds):.0f}"
                if with_sol
                else ""
            )
        )
        # one swap away from a solution: how many neighbours are solutions too
        near = []
        for i, s in enumerate(sets):
            if not good[i]:
                continue
            shapes, ins, outs = flat(s)
            nb = set()
            for side in (0, 1):
                blocks = list(ins if side == 0 else outs)
                for p, q in itertools.combinations(range(6), 2):
                    if blocks[p] == blocks[q]:
                        continue
                    b = blocks.copy()
                    b[p], b[q] = b[q], b[p]
                    s2 = (
                        rebuild(shapes, tuple(b), outs)
                        if side == 0
                        else rebuild(shapes, ins, tuple(b))
                    )
                    j = index.get(s2)
                    if j is not None and j != i:
                        nb.add(j)
            if nb:
                near.append(np.mean([good[j] for j in nb]))
        if near:
            out.append(
                f"one swap from a solution: {np.mean(near):.0%} of the neighbouring sets are "
                "solutions too (mean)"
            )
        # which reactions the solutions use
        use = Counter(r for s, ok in zip(sets, good, strict=True) if ok for r in s)
        out.append("reactions most used by solutions:")
        for r, c in use.most_common(12):
            out.append(f"  {c:5d}  {text((r,))}")
        ex = [text(s) for s, ok in zip(sets, good, strict=True) if ok][:12]
        out.append("examples: " + " | ".join(ex))
    (d / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"saved {d}")


if __name__ == "__main__":
    main()

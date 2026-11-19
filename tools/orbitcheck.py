"""Grid check over whole swap orbits: every arrangement reachable by swaps (input <-> input,
output <-> output, all rates k = 1) of the richest orbits of a mean-field census, run in mean
field and on the grid from the same start, side by side.

    python tools/orbitcheck.py runs/classify/<run>/census.csv [--min-behaviours 5]
                               [--seeds 2] [--t-max 400] [--x0 0.5,0.3,0.2]

Selection: orbits with at least --min-behaviours distinct mean-field behaviours, plus the
orbits of the sets in --include (default: rock-paper-scissors).

Outcome of an arrangement (end of the run, grid seeds must agree, else "split"):
  alive  reactions that change colour counts fire at >= --alive-rate per cell per unit
         time in the last quarter of the run (default 0.01)
  trickle they still fire, but more slowly (a few stragglers)
  mobile only exchanges (A|B -> B|A) fire: counts frozen, texture moving
  jam    nothing fires in the last quarter (identity reactions ignored) although at least
         two colours remain (frozen mosaic)
  mono   one colour left (absorbing)
  + the set of colours present, and the majority colour.
Mean field: colours above 1e-3 at t_max and the census label.
"differs" if the grid keeps another set of colours, another majority, or |x_grid - x_mf|_1 >
--tol (fractions averaged over the last quarter).
--redo <arrangements.csv> of an earlier run: keep its jam and mono rows, rerun the others.
Outputs in runs/orbitcheck/<census>_<time>/: arrangements.csv (one row per arrangement),
orbits.txt (one table per orbit, sorted by grid outcome), summary.txt (one line per orbit).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from classify import orbit_key  # noqa: E402
from gridcheck import GRID, parse  # noqa: E402
from mosaic_model import Sim, integrate, make_graph, rhs  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

INCLUDE = "01>00 ; 12>11 ; 02>22"  # rock-paper-scissors


def load_census(path: Path) -> list[tuple[str, str, str, str]]:
    """(family, behaviour, reactions, orbit) rows; the orbit field contains commas."""
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return [(r[0], r[1], r[2], ",".join(r[3:])) for r in rd]


def key_of(s: str, n: int) -> str:
    """Orbit key of a reaction string, in the census format."""
    rs = parse(s, n)
    t = []
    for r in rs.reactions:
        ins, outs = tuple(r.inp), tuple(r.out)
        if r.shape == "edge":
            ins, outs = min((ins, outs), (ins[::-1], outs[::-1]))
        t.append((r.shape, ins, outs))
    return str(orbit_key(tuple(sorted(t)), n))


def run(args):
    """Mean field (mode "mf") or one grid seed: fractions over time and late firing."""
    i, s, n, mode, seed, x0, t_max, dt = args
    g = make_graph(GRID)
    rs = parse(s, n)
    col = np.random.default_rng(seed).choice(n, size=g.n_cells, p=x0)
    times = np.arange(0.0, t_max + 1e-9, dt)
    if mode == "mf":
        xs = integrate(rhs(rs), np.bincount(col, minlength=n) / g.n_cells, times, dt=0.01)
        return i, mode, xs, -1
    sim = Sim(g, rs, col, seed)
    xs = [sim.fractions()]
    q = int(0.75 * (len(times) - 1))
    f_late = np.zeros(len(rs.reactions), dtype=np.int64)
    for k in range(1, len(times)):
        if k == q + 1:
            f_late = np.array(sim.fires)
        sim.advance(dt)
        xs.append(sim.fractions())
    late = np.array(sim.fires) - f_late
    change = [sorted(r.inp) != sorted(r.out) for r in rs.reactions]  # counts change
    move = [r.inp != r.out and not c for r, c in zip(rs.reactions, change, strict=True)]
    rate = late[change].sum() / (g.n_cells * (times[-1] - times[q]))  # per cell per time
    fires = (int(late[change].sum()), int(late[move].sum()), float(rate))
    return i, mode, np.array(xs), fires


KINDS = ("alive", "trickle", "mobile", "jam", "mono", "split")
PRESENT = 1e-3  # a colour counts as present above this fraction (grid and mean field)


def outcome(xs: np.ndarray, fires, alive_rate: float) -> tuple[str, tuple[int, ...], int]:
    """fires = (late firings of count-changing reactions, of exchanges, rate per cell)."""
    present = tuple(int(c) for c in np.flatnonzero(xs[-1] > PRESENT))
    if (xs[-1] > 0).sum() == 1:
        kind = "mono"
    elif fires[2] >= alive_rate:
        kind = "alive"
    elif fires[0]:
        kind = "trickle"
    else:
        kind = "mobile" if fires[1] else "jam"
    return kind, present, int(np.argmax(xs[-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("census", type=Path)
    ap.add_argument("--min-behaviours", type=int, default=5)
    ap.add_argument("--include", default=INCLUDE, help="sets whose orbits are added, '|'-sep.")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--t-max", type=float, default=400.0)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--x0", default="0.5,0.3,0.2")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--redo", type=Path, default=None)
    ap.add_argument(
        "--alive-rate", type=float, default=0.01, help="count-changing firings per cell per time"
    )
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    x0 = np.array([float(v) for v in a.x0.split(",")])
    n = len(x0)
    rows = load_census(a.census)
    beh: dict[tuple[str, str], set] = defaultdict(set)
    for fam, lab, _, orb in rows:
        beh[(fam, orb)].add(lab)
    want = {k for k, v in beh.items() if len(v) >= a.min_behaviours}
    extra = {key_of(s.strip(), n) for s in a.include.split("|") if s.strip()}
    want |= {k for k in beh if k[1] in extra}
    sets = [(fam, lab, s, orb) for fam, lab, s, orb in rows if (fam, orb) in want]
    kept: list[list] = []
    if a.redo is not None:  # keep jam and mono rows of an earlier run
        old = {}
        lines = a.redo.read_text().splitlines()
        has_rate = lines[0].endswith(",rate")
        for line in lines[1:]:
            f = line.split(",")
            if has_rate:
                f = f[:-1]
            tail: list = f[-15:] + [0.0]  # reactions .. verdict (n = 3 colours), rate
            for j in (2, 3, 4, 8, 9, 10, 13):  # fractions and distance
                tail[j] = float(tail[j])
            for j in (6, 12):  # majority colours
                tail[j] = int(tail[j])
            old[(f[0], f[-15])] = [f[0], ",".join(f[1:-15]), *tail]
        kept = [
            old[(fam, s)]
            for fam, _, s, _ in sets
            if (fam, s) in old and old[(fam, s)][4 + n + 2] in ("jam", "mono")
        ]
        skip = {(r[0], r[2]) for r in kept}
        sets = [x for x in sets if (x[0], x[2]) not in skip]
    params = {**vars(a), "census": str(a.census), "redo": str(a.redo)}
    d = new_run_dir("orbitcheck", a.census.parent.name, params)
    jobs = []
    for i, (_, _, s, _) in enumerate(sets):
        jobs.append((i, s, n, "mf", 0, x0, a.t_max, a.dt))
        jobs += [(i, s, n, "grid", 2000 + k, x0, a.t_max, a.dt) for k in range(a.seeds)]
    print(
        f"{len(want)} orbits, {len(sets)} arrangements to run ({len(kept)} kept), {len(jobs)} runs",
        flush=True,
    )
    t0 = time.perf_counter()
    res: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, mode, xs, fires in pool.imap_unordered(run, jobs, chunksize=4):
            res[i][mode].append((xs, fires))
    q = int(0.75 * (len(np.arange(0.0, a.t_max + 1e-9, a.dt)) - 1))
    out_rows = []
    for i, (fam, lab, s, orb) in enumerate(sets):
        xm_t = res[i]["mf"][0][0]
        xm = xm_t[q:].mean(axis=0)
        surv_m = tuple(int(c) for c in np.flatnonzero(xm_t[-1] > PRESENT))
        maj_m = int(np.argmax(xm_t[-1]))
        g = res[i]["grid"]
        xg = np.mean([xs[q:].mean(axis=0) for xs, _ in g], axis=0)
        outs = {outcome(xs, f, a.alive_rate) for xs, f in g}
        rate = float(np.mean([f[2] for _, f in g]))
        kind, surv_g, maj_g = next(iter(outs)) if len(outs) == 1 else ("split", (), -1)
        dist = float(np.abs(xg - xm).sum())
        differs = kind == "split" or surv_g != surv_m or maj_g != maj_m or dist > a.tol
        out_rows.append(
            [fam, orb, s, lab, *np.round(xm, 4), "".join(map(str, surv_m)), maj_m, kind]
            + [*np.round(xg, 4), "".join(map(str, surv_g)), maj_g, round(dist, 4)]
            + ["differs" if differs else "same", round(rate, 5)]
        )
    m = 4 + n
    for r in kept:  # same present threshold and verdict rule for rows of the earlier run
        xg_k = np.array(r[m + 3 : m + 3 + n], dtype=float)
        r[m + 3 + n] = "".join(str(c) for c in np.flatnonzero(xg_k > PRESENT))
        same = r[m + 3 + n] == r[m] and r[m + 4 + n] == r[m + 1] and r[m + 5 + n] <= a.tol
        r[m + 6 + n] = "same" if same else "differs"
    out_rows += kept
    hdr = ["family", "orbit", "reactions", "mf_behaviour"] + [f"mf{c}" for c in range(n)]
    hdr += ["mf_present", "mf_major", "grid_kind"] + [f"grid{c}" for c in range(n)]
    hdr += ["grid_present", "grid_major", "d", "verdict", "rate"]
    write_csv(d / "arrangements.csv", hdr, out_rows)

    by: dict[tuple[str, str], list] = defaultdict(list)
    for r in out_rows:
        by[(r[0], r[1])].append(r)
    summ = [
        f"orbit check of {a.census}: grid {GRID}, x0 {a.x0}, t_max {a.t_max}, {a.seeds} seeds "
        f"({time.perf_counter() - t0:.0f} s)",
        f"Per orbit: arrangements; mean-field behaviours; grid outcomes ({' / '.join(KINDS)}); "
        "distinct alive outcomes (colours present, majority); differ. Alive: count-changing "
        f"firings >= {a.alive_rate} per cell per time; present: fraction > {PRESENT}.",
    ]
    tables = []

    def f3(v):
        return " ".join(f"{float(u):.3f}" for u in v)

    for (fam, orb), rs in sorted(by.items(), key=lambda kv: -len(beh[kv[0]])):
        kinds = defaultdict(int)
        for r in rs:
            kinds[r[m + 2]] += 1
        alive = {(r[m + 3 + n], r[m + 4 + n]) for r in rs if r[m + 2] == "alive"}
        nd = sum(r[-2] == "differs" for r in rs)
        summ.append(
            f"{fam:6} {orb}\n    {len(rs)} arrangements; mf {len(beh[(fam, orb)])} "
            f"behaviours {sorted(beh[(fam, orb)])}; grid "
            + ", ".join(f"{k} {kinds[k]}" for k in KINDS)
            + f"; {len(alive)} distinct alive outcomes; {nd} differ"
        )
        tables.append(f"\n{fam} {orb}")
        tables.append(
            f"  {'grid':7} {'rate':>6} {'present':7} {'x_grid':17} {'mf':9} {'present':7} "
            f"{'x_mf':17} {'d':>5} verdict  reactions"
        )
        for r in sorted(rs, key=lambda r: (KINDS.index(r[m + 2]), r[m + 3 + n], -r[-1], r[2])):
            tables.append(
                f"  {r[m + 2]:7} {r[-1]:6.3f} {r[m + 3 + n]:7} {f3(r[m + 3 : m + 3 + n]):17} "
                f"{r[3]:9} {r[m]:7} {f3(r[4 : 4 + n]):17} {r[-3]:5.3f} {r[-2]:8} {r[2]}"
            )
    tot = {k: sum(r[m + 2] == k for r in out_rows) for k in KINDS}
    summ.append(
        f"total: {len(out_rows)} arrangements; "
        + ", ".join(f"{k} {tot[k]}" for k in KINDS)
        + f"; {sum(r[-2] == 'differs' for r in out_rows)} differ from mean field"
    )
    (d / "summary.txt").write_text("\n".join(summ) + "\n")
    (d / "orbits.txt").write_text("\n".join(tables) + "\n")
    print("\n".join(summ))
    print(f"saved {d}")


if __name__ == "__main__":
    main()

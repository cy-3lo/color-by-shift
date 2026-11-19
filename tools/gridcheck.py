"""Grid check of the mean-field census: run representatives of each mean-field behaviour on
the grid, in the shuffled control and in the mean-field ODE, from the same start, and report
where the grid departs from mean field.

    python tools/gridcheck.py runs/classify/<run>/census.csv [--per-class 3] [--t-max 400]
                              [--seeds 2] [--families cell3,edge2,edge3,mix12,mix21]

For each set: random start with fractions --x0 (product measure), all rates k = 1, the game
grid (hexagonal torus, side 74). Measured over the last quarter of the run:
  x       time-averaged fractions (grid seeds averaged; shuffled; mean field)
  surv    number of colours still present at the end (grid: in every seed)
  rho     interface density / (1 - sum x^2): 1 = well mixed, < 1 = patches (coarsening)
  osc     largest standard deviation of a fraction (sustained or noisy oscillation)
  d_grid  L1 distance |x_grid - x_mf|; d_shuf the same for the shuffled control
Verdict: "same" if d_grid < --tol and the grid keeps the same colours as mean field, else
"differs". The shuffled control checks the mean-field prediction itself (finite-size noise).
CSV (summary + time series) and summary.txt in runs/gridcheck/<census>_<time>/.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import Reaction, ReactionSet, Sim, integrate, make_graph, rhs  # noqa: E402
from mosaic_model.observables import interface_density  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

GRID = {"type": "hexagon", "side": 74}
EXTRA = {  # always included: classic benchmarks
    "voter": "01>00 ; 01>11",
    "rps": "01>00 ; 12>11 ; 02>22",
}


def parse(s: str, n: int) -> ReactionSet:
    rx = []
    for part in s.split(";"):
        a, b = part.strip().split(">")
        shape = "cell" if len(a) == 1 else "edge"
        rx.append(Reaction(shape, [int(c) for c in a], [int(c) for c in b], 1.0))
    return ReactionSet(rx, n)


def pick(census: Path, families: list[str], per: int, seed: int) -> list[tuple[str, str, str]]:
    by: dict[tuple[str, str], list[str]] = {}
    with open(census) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r[0] in families:
                by.setdefault((r[0], r[1]), []).append(r[2])
    rng = np.random.default_rng(seed)
    out = []
    for (fam, lab), sets in sorted(by.items()):
        for i in rng.choice(len(sets), size=min(per, len(sets)), replace=False):
            out.append((fam, lab, sets[i]))
    return out


def run(args):
    """One (set, mode, seed): time series of fractions and interface density."""
    key, s, n, mode, seed, x0, t_max, dt = args
    g = make_graph(GRID)
    rs = parse(s, n)
    rng = np.random.default_rng(seed)
    col = rng.choice(n, size=g.n_cells, p=x0)
    times = np.arange(0.0, t_max + 1e-9, dt)
    if mode == "mf":
        xs = integrate(rhs(rs), np.bincount(col, minlength=n) / g.n_cells, times, dt=0.01)
        return key, mode, seed, times, xs, np.full(len(times), np.nan)
    sim = Sim(g, rs, col, seed, shuffled=mode == "shuf")
    xs, rho = [sim.fractions()], [interface_density(sim.colours, sim.edge_arr)]
    for _ in times[1:]:
        sim.advance(dt)
        xs.append(sim.fractions())
        rho.append(interface_density(sim.colours, sim.edge_arr))
    return key, mode, seed, times, np.array(xs), np.array(rho)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("census", type=Path)
    ap.add_argument("--families", default="cell3,edge2,edge3,mix12,mix21")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--t-max", type=float, default=400.0)
    ap.add_argument("--dt", type=float, default=2.0, help="sampling interval")
    ap.add_argument("--x0", default="0.5,0.3,0.2")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    x0 = np.array([float(v) for v in a.x0.split(",")])
    n = len(x0)
    sets = [("extra", k, v) for k, v in EXTRA.items()]
    sets += pick(a.census, a.families.split(","), a.per_class, 0)
    d = new_run_dir("gridcheck", a.census.parent.name, {**vars(a), "census": str(a.census)})
    jobs = []
    for i, (_, _, s) in enumerate(sets):
        jobs.append((i, s, n, "mf", 0, x0, a.t_max, a.dt))
        jobs.append((i, s, n, "shuf", 1000, x0, a.t_max, a.dt))
        jobs += [(i, s, n, "grid", 2000 + k, x0, a.t_max, a.dt) for k in range(a.seeds)]
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    series = []
    runs: dict[int, dict[str, list]] = {}
    for key, mode, seed, times, xs, rho in res:
        runs.setdefault(key, {}).setdefault(mode, []).append((xs, rho))
        for t, x, r in zip(times, xs, rho, strict=True):
            series.append([key, mode, seed, round(t, 3), *np.round(x, 5), round(r, 5)])
    q = int(0.75 * len(np.arange(0.0, a.t_max + 1e-9, a.dt)))

    def stats(lst, thr):
        xm = np.mean([xs[q:].mean(axis=0) for xs, _ in lst], axis=0)
        surv = min(int((xs[-1] > thr).sum()) for xs, _ in lst)
        osc = max(float(xs[q:].std(axis=0).max()) for xs, _ in lst)
        rr = [r[q:].mean() / max(1e-9, 1 - (xs[q:] ** 2).sum(axis=1).mean()) for xs, r in lst]
        return xm, surv, osc, float(np.mean(rr))

    rows = []
    hdr = (
        f"{'#':>3} {'family':6} {'mean field':10} {'verdict':8} {'x_mf':17} {'x_grid':17} "
        f"{'x_shuf':17} srv(mf/g/s) {'d_grid':>6} {'d_shuf':>6} {'rho':>5} {'osc_g':>5} "
        f"{'osc_mf':>6}  reactions"
    )
    lines = [
        f"grid check of {a.census}: grid {GRID}, start x0 {a.x0}, t_max {a.t_max}, "
        f"{a.seeds} grid seeds; averages over the last quarter",
        hdr,
    ]
    for i, (fam, lab, s) in enumerate(sets):
        xm, sm, om, _ = stats(runs[i]["mf"], 1e-3)  # mean field: below 1e-3 counts as gone
        xg, sg, og, rg = stats(runs[i]["grid"], 0.0)  # grid: at least one cell
        xs_, ss, _, _ = stats(runs[i]["shuf"], 0.0)
        dg, ds = float(np.abs(xg - xm).sum()), float(np.abs(xs_ - xm).sum())
        verdict = "same" if dg < a.tol and sg == sm else "differs"
        rows.append([i, fam, lab, s, verdict, *np.round(xm, 4), *np.round(xg, 4)])
        rows[-1] += [*np.round(xs_, 4), sm, sg, ss, round(dg, 4), round(ds, 4), round(rg, 3)]
        rows[-1] += [round(og, 4), round(om, 4)]

        def f3(v):
            return " ".join(f"{u:.3f}" for u in v)

        lines.append(
            f"{i:3d} {fam:6} {lab:10} {verdict:8} {f3(xm):17} {f3(xg):17} {f3(xs_):17} "
            f"  {sm}/{sg}/{ss}     {dg:6.3f} {ds:6.3f} {rg:5.2f} {og:5.3f} {om:6.3f}  {s}"
        )
    nd = sum(r[4] == "differs" for r in rows)
    lines.append(f"{nd} of {len(rows)} sets differ ({time.perf_counter() - t0:.0f} s)")
    cols = [f"{p}{c}" for p in ("mf", "grid", "shuf") for c in range(n)]
    write_csv(
        d / "summary.csv",
        ["set", "family", "mf_behaviour", "reactions", "verdict", *cols]
        + ["surv_mf", "surv_grid", "surv_shuf", "d_grid", "d_shuf", "rho", "osc_grid", "osc_mf"],
        rows,
    )
    write_csv(
        d / "series.csv",
        ["set", "mode", "seed", "t", *[f"x{c}" for c in range(n)], "interface"],
        series,
    )
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {d}")


if __name__ == "__main__":
    main()

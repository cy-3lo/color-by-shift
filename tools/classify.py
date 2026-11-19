"""Census of mean-field behaviours of small reaction sets with equal rates, and of what the
swap rule (input <-> input, output <-> output) can reach.

    python tools/classify.py [--colours 3] [--families cell3,edge2,edge3,mix12] [--ics 8]
                             [--workers N]

Families: cellR = R cell reactions, edgeR = R edge reactions, mixCE = C cell + E edge.
Every reaction has rate k = 1. Mean field (meanfield.py): dx/dt = sum_r w_r x_in(r) nu_r with
w = 1 for a cell reaction and 6 (= 3 edges per cell x 2 orientations) for an edge reaction.

Reaction sets are enumerated up to colour relabelling and edge orientation
(A|B -> C|D is the same reaction as B|A -> D|C). Each set is integrated (RK4) from `ics`
interior initial conditions and classified from the long-time behaviour:
  frozen        nothing changes in mean field (identities and exchanges only)
  fp:<s>        every start goes to the same fixed point with s colours present (1, 2, 3)
  multi:<...>   several attracting fixed points (bistability etc.)
  family        converged, but the end point depends continuously on the start
                (a line or plane of fixed points: neutral, e.g. the voter model)
  cycle         sustained oscillation of the same amplitude from different starts
  neutral       closed orbits whose amplitude depends on the start (a conserved quantity)
  hetero        oscillation creeping toward the boundary (heteroclinic cycle)
  damped        oscillation still decaying at t_max (a slow focus)
  slow          not settled by t_max (marginal cases)
Swap orbits: sets with the same shapes and the same multisets of input and output colours
(up to relabelling); the player can move between any two members of an orbit.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from runlog import new_run_dir, write_csv  # noqa: E402

W = {"cell": 1.0, "edge": 6.0}


# ---- enumeration ---------------------------------------------------------------------------
def reactions(shape: str, n: int) -> list[tuple]:
    """All reactions of a shape up to edge orientation: (shape, inputs, outputs)."""
    out = set()
    m = 1 if shape == "cell" else 2
    for ins in itertools.product(range(n), repeat=m):
        for outs in itertools.product(range(n), repeat=m):
            if shape == "edge":
                a = (ins, outs)
                b = (ins[::-1], outs[::-1])
                ins, outs = min(a, b)
            out.add((shape, ins, outs))
    return sorted(out)


def relabel(rs: tuple, perm: tuple[int, ...]) -> tuple:
    res = []
    for shape, ins, outs in rs:
        i2 = tuple(perm[c] for c in ins)
        o2 = tuple(perm[c] for c in outs)
        if shape == "edge":
            i2, o2 = min((i2, o2), (i2[::-1], o2[::-1]))
        res.append((shape, i2, o2))
    return tuple(sorted(res))


def canonical(rs: tuple, n: int) -> tuple:
    return min(relabel(rs, p) for p in itertools.permutations(range(n)))


def family_sets(n_cell: int, n_edge: int, n: int) -> list[tuple]:
    """Canonical reaction sets (multisets of reactions) with these numbers of each shape."""
    cells, edges = reactions("cell", n), reactions("edge", n)
    seen = set()
    for cs in itertools.combinations_with_replacement(cells, n_cell):
        for es in itertools.combinations_with_replacement(edges, n_edge):
            seen.add(canonical(tuple(sorted(cs + es)), n))
    return sorted(seen)


def orbit_key(rs: tuple, n: int) -> tuple:
    """Shapes and the multisets of input and output colours, up to relabelling."""
    best = None
    for p in itertools.permutations(range(n)):
        r2 = relabel(rs, p)
        key = (
            tuple(sorted(s for s, _, _ in r2)),
            tuple(sorted(c for _, ins, _ in r2 for c in ins)),
            tuple(sorted(c for _, _, outs in r2 for c in outs)),
        )
        best = key if best is None or key < best else best
    return best


# ---- batched mean field --------------------------------------------------------------------
def build(sets: list[tuple], n: int):
    """Arrays for a batch of sets of identical size: inputs (S, R, 2), weights (S, R),
    net changes (S, R, n); a cell reaction uses input index n (a constant 1)."""
    s_, r_ = len(sets), len(sets[0])
    ins = np.full((s_, r_, 2), n, dtype=np.int64)
    w = np.zeros((s_, r_))
    nu = np.zeros((s_, r_, n))
    for i, rs in enumerate(sets):
        for j, (shape, a, b) in enumerate(rs):
            ins[i, j, : len(a)] = a
            w[i, j] = W[shape]
            for c in b:
                nu[i, j, c] += 1
            for c in a:
                nu[i, j, c] -= 1
    return ins, w, nu


def rhs(x: np.ndarray, ins, w, nu) -> np.ndarray:
    """x: (S, M, n) -> dx/dt."""
    s_, m_, n = x.shape
    xe = np.concatenate([x, np.ones((s_, m_, 1))], axis=2)  # index n = 1
    idx0 = np.broadcast_to(ins[:, None, :, 0], (s_, m_, ins.shape[1]))
    idx1 = np.broadcast_to(ins[:, None, :, 1], (s_, m_, ins.shape[1]))
    p = np.take_along_axis(xe, idx0, axis=2) * np.take_along_axis(xe, idx1, axis=2)  # (S,M,R)
    return np.einsum("smr,sr,srn->smn", p, w, nu)


def integrate(x0, ins, w, nu, t_max: float, dt: float, n_rec: int):
    x = x0.copy()
    steps = int(t_max / dt)
    rec_every = max(1, steps // n_rec)
    recs = []
    for k in range(steps):
        k1 = rhs(x, ins, w, nu)
        k2 = rhs(x + 0.5 * dt * k1, ins, w, nu)
        k3 = rhs(x + 0.5 * dt * k2, ins, w, nu)
        k4 = rhs(x + dt * k3, ins, w, nu)
        x = np.clip(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0.0, 1.0)
        x /= x.sum(axis=2, keepdims=True)
        if (k + 1) % rec_every == 0:
            recs.append(x.copy())
    return np.array(recs)  # (T, S, M, n)


def classify(traj: np.ndarray, d0: np.ndarray) -> list[str]:
    """traj (T, S, M, n); d0 (S,) max |dx/dt| at the start points."""
    t_, s_, m_, n = traj.shape
    late = traj[int(0.75 * t_) :]  # last quarter
    mid = traj[int(0.5 * t_) : int(0.75 * t_)]
    out = []
    for s in range(s_):
        if d0[s] < 1e-12:
            out.append("frozen")
            continue
        amp_late = (late[:, s].max(axis=0) - late[:, s].min(axis=0)).max(axis=1)  # (M,)
        amp_mid = (mid[:, s].max(axis=0) - mid[:, s].min(axis=0)).max(axis=1)
        end = late[-1, s]  # (M, n)
        if (amp_late < 1e-3).all():
            # converged: compare end points
            pts = np.round(end, 2)
            uniq = {tuple(p) for p in pts}
            spread = np.ptp(end, axis=0).max()
            if len(uniq) == 1 or spread < 0.02:
                out.append(f"fp:{int((end[0] > 1e-3).sum())}")
            elif len(uniq) <= max(2, m_ // 2):
                sup = sorted({int((p > 1e-3).sum()) for p in end})
                out.append("multi:" + "".join(map(str, sup)))
            else:
                out.append("family")
            continue
        if (amp_late < 1e-3).any():
            out.append("slow")
            continue
        minx_late = late[:, s].min(axis=(0, 2))
        minx_mid = mid[:, s].min(axis=(0, 2))
        steady = (amp_late >= 0.97 * amp_mid) & (amp_late <= 1.03 * amp_mid)
        if (minx_late < 0.5 * minx_mid).all() and (amp_late >= 0.95 * amp_mid).all():
            out.append("hetero")
        elif (amp_late < 0.97 * amp_mid).any():
            out.append("damped")  # still decaying at t_max: a slow focus, not a cycle
        elif steady.all() and np.ptp(amp_late) < 0.02 * amp_late.mean():
            out.append("cycle")
        elif steady.all():
            out.append("neutral")
        else:
            out.append("slow")
    return out


def _label_chunk(args) -> list[str]:
    chunk, n, x0, t_max, dt = args
    ins, w, nu = build(chunk, n)
    x = np.broadcast_to(x0, (len(chunk), len(x0), n)).copy()
    d0 = np.abs(rhs(x, ins, w, nu)).max(axis=(1, 2))
    return classify(integrate(x, ins, w, nu, t_max, dt, 200), d0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--colours", type=int, default=3)
    ap.add_argument("--families", default="cell2,cell3,edge1,edge2,edge3,mix11,mix12,mix21")
    ap.add_argument("--ics", type=int, default=8)
    ap.add_argument("--t-max", type=float, default=400.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    n = a.colours
    d = new_run_dir("classify", f"n{n}", vars(a))
    lines = [f"Mean-field census, {n} colours, all rates k = 1, {a.ics} starts, t_max {a.t_max}"]
    rows = []
    for fam in a.families.split(","):
        if fam.startswith("cell"):
            nc, ne = int(fam[4:]), 0
        elif fam.startswith("edge"):
            nc, ne = 0, int(fam[4:])
        else:
            nc, ne = int(fam[3]), int(fam[4])
        t0 = time.perf_counter()
        sets = family_sets(nc, ne, n)
        rng = np.random.default_rng([n, zlib.crc32(fam.encode())])  # starts depend on family only
        x0 = rng.dirichlet(np.ones(n), size=a.ics)
        jobs = [(sets[b : b + a.batch], n, x0, a.t_max, a.dt) for b in range(0, len(sets), a.batch)]
        with mp.get_context("fork").Pool(a.workers) as pool:
            labels = [lab for part in pool.map(_label_chunk, jobs) for lab in part]
        census = Counter(labels)
        orbits: dict[tuple, set] = defaultdict(set)
        members: dict[tuple, int] = Counter()
        for rs, lab in zip(sets, labels, strict=True):
            k = orbit_key(rs, n)
            orbits[k].add(lab)
            members[k] += 1
            rows.append(
                [
                    fam,
                    lab,
                    " ; ".join(f"{''.join(map(str, i))}>{''.join(map(str, o))}" for _, i, o in rs),
                    str(k),
                ]
            )
        richness = Counter(len(v) for v in orbits.values())
        lines.append(
            f"\n{fam}: {len(sets)} reaction sets, {len(orbits)} swap orbits "
            f"({time.perf_counter() - t0:.0f} s)"
        )
        lines.append("  behaviours: " + ", ".join(f"{k} {v}" for k, v in census.most_common()))
        lines.append(
            "  distinct behaviours per orbit: "
            + ", ".join(f"{k}: {v} orbits" for k, v in sorted(richness.items()))
        )
        best = sorted(orbits, key=lambda k: -len(orbits[k]))[:3]
        for k in best:
            lines.append(f"  richest orbit {k}: {members[k]} sets, {sorted(orbits[k])}")
        print("\n".join(lines[-5:]), flush=True)
    write_csv(d / "census.csv", ["family", "behaviour", "reactions", "orbit"], rows)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"saved {d}")


if __name__ == "__main__":
    main()

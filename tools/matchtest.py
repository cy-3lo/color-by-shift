"""Bot test of the "match the texture" game.

    python tools/matchtest.py runs/orbitcheck/<run>/arrangements.csv [--orbits 3] [--t-max 200]

The game under test: a target mosaic is made by a hidden arrangement of blocks; the player
swaps blocks (input <-> input, output <-> output, all k = 1) and wins when the mosaic of
their arrangement has the same signature as the target.

Player space of an orbit: every distinct reaction set with the orbit's shapes and its input
and output colour multisets (colour names matter here: a relabelled set is another set).
The --orbits orbits with the most alive arrangements in the orbit check are tested.

Each set is run on the game grid from a random start with equal fractions, for two seeds:
seed A plays the target, seed B plays the player's attempt. Signature (averages over the last
quarter of the run):
  x_c      colour fractions                                   tolerance --tol-x
  e_ab     share of neighbouring pairs of colours a != b      tolerance --tol-e
  S_c      patch size of colour c, as log10(S_c + 1)          tolerance --tol-s
  act      count-changing firings per cell per unit time,
           as log10(act + 0.01)                               tolerance --tol-a
Two signatures match if every component agrees within its tolerance. d = mean over the
components of |difference| / tolerance (the bot's "how far" reading).
Targets: sets that are alive (act >= 0.01 and at least two colours, both seeds) and
reproducible (seed B matches seed A).

Measures per orbit (summary.txt):
  classes     groups of mutually reachable matching targets (connected components of "match")
              with the full signature / with the fractions only
  matches     sets whose attempt matches the target: full signature / fractions only
  texture     share of targets where some set matches the fractions but not the full signature
  tries       sets run before a match (the start counts):
     blind    random order
     mean field   sets ordered by |x_meanfield - x_target| (a player who solves the well-mixed
              equations and ignores the grid)
     greedy   from a random set, try single swaps in random order, move to the first one with a
              smaller d; when no swap improves, jump to a random untried set
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from gridcheck import GRID  # noqa: E402
from mosaic_model import Reaction, ReactionSet, Sim, integrate, make_graph, rhs  # noqa: E402
from mosaic_model.observables import patch_sizes  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

Set = tuple  # sorted tuple of (shape, inputs, outputs)


def parse_set(s: str) -> Set:
    out = []
    for part in s.split(";"):
        a, b = part.strip().split(">")
        out.append(
            norm("cell" if len(a) == 1 else "edge", [int(c) for c in a], [int(c) for c in b])
        )
    return tuple(sorted(out))


def norm(shape: str, ins, outs) -> tuple:
    ins, outs = tuple(ins), tuple(outs)
    if shape == "edge":
        ins, outs = min((ins, outs), (ins[::-1], outs[::-1]))
    return (shape, ins, outs)


def text(rs: Set) -> str:
    return " ; ".join(f"{''.join(map(str, i))}>{''.join(map(str, o))}" for _, i, o in rs)


def rebuild(shapes: list[str], ins: tuple, outs: tuple) -> Set:
    """Set from the flat block lists (slot order = shapes)."""
    out, k = [], 0
    for sh in shapes:
        m = 1 if sh == "cell" else 2
        out.append(norm(sh, ins[k : k + m], outs[k : k + m]))
        k += m
    return tuple(sorted(out))


def flat(rs: Set) -> tuple[list[str], tuple, tuple]:
    return (
        [r[0] for r in rs],
        tuple(c for r in rs for c in r[1]),
        tuple(c for r in rs for c in r[2]),
    )


def player_orbit(rs: Set) -> list[Set]:
    shapes, ins, outs = flat(rs)
    pi = sorted(set(itertools.permutations(ins)))
    po = sorted(set(itertools.permutations(outs)))
    return sorted({rebuild(shapes, a, b) for a in pi for b in po})


def swap_neighbours(rs: Set, index: dict) -> list[int]:
    shapes, ins, outs = flat(rs)
    me, nb = index[rs], set()
    for side in (0, 1):
        blocks = list(ins if side == 0 else outs)
        for i, j in itertools.combinations(range(len(blocks)), 2):
            if blocks[i] == blocks[j]:
                continue
            b = blocks.copy()
            b[i], b[j] = b[j], b[i]
            s2 = rebuild(shapes, tuple(b), outs) if side == 0 else rebuild(shapes, ins, tuple(b))
            nb.add(index[s2])
    nb.discard(me)
    return sorted(nb)


def reaction_set(rs: Set, n: int) -> ReactionSet:
    return ReactionSet([Reaction(sh, list(i), list(o), 1.0) for sh, i, o in rs], n)


PAIRS = [(0, 1), (0, 2), (1, 2)]


def run(args):
    """Signature of one set for one seed (mode "grid") or its mean-field fractions ("mf")."""
    i, rs, n, mode, seed, t_max, dt = args
    g = make_graph(GRID)
    r = reaction_set(rs, n)
    col = np.random.default_rng(seed).choice(n, size=g.n_cells)
    if mode == "mf":
        x0 = np.bincount(col, minlength=n) / g.n_cells
        xs = integrate(rhs(r), x0, np.array([0.0, t_max]), dt=0.01)
        return i, mode, seed, xs[-1]
    sim = Sim(g, r, col, seed)
    t0 = 0.75 * t_max
    sim.advance(t0)
    f0 = np.array(sim.fires)
    cells = np.arange(g.n_cells)
    e = g.edges
    xs, es, ss = [], [], []
    for t in np.arange(t0, t_max + 1e-9, dt):
        sim.advance(t - sim.t)
        c = sim.colours
        xs.append(sim.fractions())
        a, b = np.minimum(c[e[:, 0]], c[e[:, 1]]), np.maximum(c[e[:, 0]], c[e[:, 1]])
        es.append([float(np.mean((a == p) & (b == q))) for p, q in PAIRS])
        ss.append(np.log10(patch_sizes(c, e, cells, n)[1] + 1.0))
    late = np.array(sim.fires) - f0
    change = np.array([sorted(x.inp) != sorted(x.out) for x in r.reactions])
    act = late[change].sum() / (g.n_cells * (sim.t - t0))
    x = np.mean(xs, axis=0)
    alive = bool(act >= 0.01 and (x > 1e-3).sum() >= 2)
    sig = np.concatenate([x, np.mean(es, axis=0), np.mean(ss, axis=0), [np.log10(act + 0.01)]])
    return i, mode, seed, (sig, alive)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arrangements", type=Path)
    ap.add_argument("--orbits", type=int, default=3)
    ap.add_argument("--t-max", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--tol-x", type=float, default=0.03)
    ap.add_argument("--tol-e", type=float, default=0.03)
    ap.add_argument("--tol-s", type=float, default=0.3)
    ap.add_argument("--tol-a", type=float, default=0.3)
    ap.add_argument("--starts", type=int, default=5, help="bot runs per target")
    ap.add_argument("--max-sets", type=int, default=0, help="smoke test: first sets only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    n = 3
    tol = np.array([a.tol_x] * 3 + [a.tol_e] * 3 + [a.tol_s] * 3 + [a.tol_a])
    names = ["x0", "x1", "x2", "e01", "e02", "e12", "S0", "S1", "S2", "act"]
    fr = slice(0, 3)
    # orbits with the most alive arrangements in the orbit check (fields parsed from the end)
    alive_n: dict[str, int] = defaultdict(int)
    rep: dict[str, str] = {}
    with open(a.arrangements) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            orb = ",".join(r[1:-16])
            rep.setdefault(orb, r[-16])
            alive_n[orb] += r[-9] == "alive"
    chosen = sorted(alive_n, key=lambda k: -alive_n[k])[: a.orbits]
    d = new_run_dir(
        "matchtest", a.arrangements.parent.name, {**vars(a), "arrangements": str(a.arrangements)}
    )
    out = [
        f"match test from {a.arrangements}: grid {GRID}, equal start fractions, t_max {a.t_max:g}, "
        f"2 seeds; tolerances x {a.tol_x}, pairs {a.tol_e}, log10 S {a.tol_s}, log10 act {a.tol_a}"
    ]
    rows = []
    rng = np.random.default_rng(a.seed)
    for orb in chosen:
        sets = player_orbit(parse_set(rep[orb]))
        if a.max_sets:
            sets = sets[: a.max_sets]
        index = {s: i for i, s in enumerate(sets)}
        ns = len(sets)
        jobs = [(i, s, n, "grid", sd, a.t_max, a.dt) for i, s in enumerate(sets) for sd in (11, 12)]
        jobs += [(i, s, n, "mf", 11, a.t_max, a.dt) for i, s in enumerate(sets)]
        print(f"orbit {orb}: {ns} sets, {len(jobs)} runs", flush=True)
        sig = np.zeros((2, ns, len(tol)))
        alive = np.zeros((2, ns), bool)
        mf = np.zeros((ns, n))
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, (i, mode, sd, res) in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
                if mode == "mf":
                    mf[i] = res
                else:
                    sig[sd - 11, i], alive[sd - 11, i] = res
                if (k + 1) % 200 == 0:
                    print(f"  {k + 1}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
        A, B = sig  # target seed, attempt seed
        dev = np.abs(B[:, None, :] - A[None, :, :]) / tol  # [attempt set, target set, component]
        match = (dev <= 1).all(axis=2)
        match_x = (dev[:, :, fr] <= 1).all(axis=2)
        dist = dev.mean(axis=2)
        selfm = match[np.arange(ns), np.arange(ns)]
        targets = np.flatnonzero(alive.all(axis=0) & selfm)
        seed_diff = np.median(np.abs(B - A)[alive.all(axis=0)], axis=0)
        for i, s in enumerate(sets):
            rows.append(
                [orb.replace(",", ";"), text(s), int(alive[:, i].all()), int(selfm[i])]
                + [float(v) for v in A[i]]
                + [float(v) for v in B[i]]
                + [float(v) for v in mf[i]]
            )

        def classes(m: np.ndarray, targets=targets) -> int:
            lab = {int(t): int(t) for t in targets}
            for p in targets:
                for q in targets:
                    if m[p, q] and m[q, p]:
                        lp, lq = lab[int(p)], lab[int(q)]
                        if lp != lq:
                            lab = {k: (lp if v == lq else v) for k, v in lab.items()}
            return len(set(lab.values()))

        nbs = [swap_neighbours(s, index) for s in sets] if not a.max_sets else None
        m_full = match[:, targets].sum(axis=0)
        m_x = match_x[:, targets].sum(axis=0)
        texture = float(np.mean(m_x > m_full)) if len(targets) else 0.0
        blind, mfp, greedy = [], [], []
        for tg in targets:
            ok = match[:, tg]
            for _ in range(a.starts):
                order = rng.permutation(ns)
                blind.append(int(np.argmax(ok[order])) + 1)
            order = np.argsort(np.abs(mf - A[tg, fr]).max(axis=1), kind="stable")
            mfp.append(int(np.argmax(ok[order])) + 1)
            if nbs is None:
                continue
            for _ in range(a.starts):
                cur = int(rng.integers(ns))
                tried = {cur}
                while not ok[cur]:
                    nxt = None
                    for j in rng.permutation(nbs[cur]):
                        j = int(j)
                        if j in tried:
                            continue
                        tried.add(j)
                        if ok[j] or dist[j, tg] < dist[cur, tg]:
                            nxt = j
                            break
                    if nxt is None:  # local minimum: jump
                        rest = [j for j in range(ns) if j not in tried]
                        nxt = int(rng.choice(rest))
                        tried.add(nxt)
                    cur = nxt
                greedy.append(len(tried))

        def q(v) -> str:
            if not len(v):
                return "-"
            v = np.array(v)
            return (
                f"median {np.median(v):.0f}, 90 % {np.quantile(v, 0.9):.0f}, "
                f"<=3 tries {np.mean(v <= 3):.0%}"
            )

        out += [
            f"orbit {orb}",
            f"  {ns} sets; alive in both seeds {int(alive.all(axis=0).sum())}; reproducible "
            f"targets {len(targets)}; median seed difference "
            + ", ".join(f"{nm} {v:.3f}" for nm, v in zip(names, seed_diff, strict=True)),
            f"  classes among targets: full signature {classes(match)}, fractions only "
            f"{classes(match_x)}",
            f"  sets matching a target: full median "
            f"{np.median(m_full) if len(targets) else 0:.0f}, "
            f"fractions only median {np.median(m_x) if len(targets) else 0:.0f}; texture needed "
            f"(some set matches fractions but not the full signature) for {texture:.0%} of targets",
            f"  tries, blind:      {q(blind)}",
            f"  tries, mean field: {q(mfp)}",
            f"  tries, greedy:     {q(greedy)}",
        ]
        print("\n".join(out[-7:]), flush=True)
    hdr = ["orbit", "reactions", "alive", "reproducible"]
    hdr += [f"A_{k}" for k in names] + [f"B_{k}" for k in names] + ["mf0", "mf1", "mf2"]
    write_csv(d / "signatures.csv", hdr, rows)
    (d / "summary.txt").write_text("\n".join(out) + "\n")
    print(f"saved {d}")


if __name__ == "__main__":
    main()

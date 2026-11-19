"""Outcome table for the colouring game: what every small rule set does to every kind of start.

    python tools/ruletable.py                      # run the table, then the level finder
    python tools/ruletable.py --table runs/ruletable/<run>/table.csv   # level finder only

Game under study (Chris, 2026-11-19): a picture made of zones sealed by fixed edges; every
zone starts with a given content and has a target look (a flat colour or a pattern of two or
three colours); the player writes ONE rule set (free writing, a few slots) for the whole
picture, presses play, reads which zones failed and changes the rules.

Because zones are sealed, a zone's outcome depends only on (rule set, its start). The table
runs every set of 1 to --max-reactions distinct edge reactions on 3 colours (k = 1, identities
left out) on one sealed patch (open hex rectangle --w x --w) for four canonical starts:
  F  flat: all colour 0
  S  seeded: colour 0 with --seeds of colour 1 scattered
  M  mix: colours 0 and 1, half each
  T  three-way mix, a third each
The 13 start kinds of the game (flat A; A seeded with B; mix A|B; three-way) follow by
relabelling colours, since the list of rule sets is closed under relabelling.

Look of a patch at the end of the run (two seeds; "unreliable" if they disagree or drift):
  flat c            one colour holds 99 %
  frozen / alive    pattern: change rate (share of cells that change per unit time, last
                    quarter) below --frozen / above --alive; described by the colours
                    present (1 % each), the majority colour, its share (even < 0.6 <= mostly
                    < 0.85 <= dotted) and the grain (fine if the unlike-neighbour density is
                    at least 0.75 of a random mix with the same shares, else coarse)
  slow              in between (still drifting): unreliable
Outputs in runs/ruletable/<time>/: table.csv, summary.txt (vocabulary of looks, puzzle counts
and examples).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from classify import reactions, relabel  # noqa: E402
from matchtest import parse_set, reaction_set, text  # noqa: E402
from mosaic_model import Sim  # noqa: E402
from mosaic_model.graph import hex_rect  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

STARTS = ("F", "S", "M", "T")
SEEDS = (1, 2)
_P: dict = {}


def start_colours(kind: str, n_cells: int, seeds: float, rng) -> np.ndarray:
    if kind == "F":
        return np.zeros(n_cells, np.int64)
    p = {"S": [1 - seeds, seeds, 0], "M": [0.5, 0.5, 0], "T": [1 / 3, 1 / 3, 1 / 3]}[kind]
    return rng.choice(3, size=n_cells, p=p)


def can_fire(rs, col: np.ndarray, e: np.ndarray) -> bool:
    a, b = np.minimum(col[e[:, 0]], col[e[:, 1]]), np.maximum(col[e[:, 0]], col[e[:, 1]])
    present = set(np.unique(a * 3 + b).tolist())
    return any(min(i) * 3 + max(i) in present for _, i, _ in rs)


def run_set(args):
    """All starts and seeds for one rule set: rows (start, seed, x0, x1, x2, rate, rho, same)."""
    i, rs = args
    p = _P
    g, e = p["g"], p["g"].edges
    out = []
    for kind in STARTS:
        for seed in SEEDS:
            col0 = start_colours(kind, g.n_cells, p["seeds"], np.random.default_rng([seed, 7]))
            sim = Sim(g, reaction_set(rs, 3), col0, seed)
            t0 = 0.75 * p["t_max"]
            rates, prev, dead = [], None, False
            while sim.t < p["t_max"] - 1e-9:
                c = sim.colours
                if int(round(sim.t)) % 10 == 0 and not can_fire(rs, c, e):
                    dead = True  # absorbing: no reaction has its input pair anywhere
                    break
                if sim.t >= t0:
                    if prev is not None:
                        rates.append(float(np.mean(c != prev)))
                    prev = c.copy()
                sim.advance(1.0)
            c = sim.colours
            if prev is not None and not dead:
                rates.append(float(np.mean(c != prev)))
            x = np.bincount(c, minlength=3) / g.n_cells
            rho = float(np.mean(c[e[:, 0]] != c[e[:, 1]]))
            rate = 0.0 if dead or not rates else float(np.mean(rates))
            out.append((kind, seed, *x.tolist(), rate, rho, float(np.mean(c == col0))))
    return i, out


def look(x, rate: float, rho: float, a) -> tuple:
    """Look of a patch (see the module docstring)."""
    x = np.asarray(x, dtype=float)
    m = int(np.argmax(x))
    if x[m] >= 0.99:
        return ("flat", m)
    cls = "alive" if rate >= a.alive else "frozen" if rate < a.frozen else "slow"
    cols = tuple(int(c) for c in np.flatnonzero(x >= 0.01))
    share = "even" if x[m] < 0.6 else "mostly" if x[m] < 0.85 else "dotted"
    rnd = 1.0 - float((x**2).sum())
    grain = "fine" if rho >= 0.75 * rnd else "coarse"
    return (cls, cols, m, share, grain)


def look_text(k) -> str:
    if k is None:
        return "unreliable"
    if k[0] == "flat":
        return f"flat {k[1]}"
    return f"{k[0]} {''.join(map(str, k[1]))} ({k[3]} {k[2]}, {k[4]})"


def relabel_look(k, p):
    if k is None:
        return None
    if k[0] == "flat":
        return ("flat", p[k[1]])
    return (k[0], tuple(sorted(p[c] for c in k[1])), p[k[2]], k[3], k[4])


def start_kinds() -> list[tuple[str, str, tuple]]:
    """(name, canonical start, permutation p: canonical colour -> game colour)."""
    out = []
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        out.append((f"flat {a}", "F", (a, b, c)))
    for a, b in itertools.permutations(range(3), 2):
        c = 3 - a - b
        out.append((f"{a} seeded with {b}", "S", (a, b, c)))
    for a, b in itertools.combinations(range(3), 2):
        out.append((f"mix {a}|{b}", "M", (a, b, 3 - a - b)))
    out.append(("mix 0|1|2", "T", (0, 1, 2)))
    return out


def analyse(table: Path, a, d: Path) -> None:
    raw: dict = defaultdict(dict)  # set text -> (start, seed) -> (x, rate, rho, same)
    with open(table) as f:
        for r in csv.DictReader(f):
            x = [float(r["x0"]), float(r["x1"]), float(r["x2"])]
            raw[r["reactions"]][(r["start"], int(r["seed"]))] = (
                x,
                float(r["rate"]),
                float(r["rho"]),
                float(r["same"]),
            )
    sets = sorted(raw, key=lambda s: (s.count(";"), s))
    canon: dict = {}  # (set text, canonical start) -> look or None
    for s in sets:
        for kind in STARTS:
            ks = [look(*raw[s][(kind, sd)][:3], a) for sd in SEEDS]
            canon[(s, kind)] = ks[0] if ks[0] == ks[1] and ks[0][0] != "slow" else None
    kinds = start_kinds()
    parsed = {s: parse_set(s) for s in sets}
    by_tuple = {v: k for k, v in parsed.items()}
    vec: dict[str, tuple] = {}
    for s in sets:
        v = []
        for _, kind, p in kinds:
            pinv = tuple(int(i) for i in np.argsort(p))
            s2 = by_tuple[relabel(parsed[s], pinv)]
            v.append(relabel_look(canon[(s2, kind)], p))
        vec[s] = tuple(v)
    nre = {s: s.count(";") + 1 for s in sets}
    out = [
        f"rule table {table}: {len(sets)} rule sets (1 to {max(nre.values())} edge reactions, "
        f"3 colours), {len(kinds)} start kinds",
        f"reliable outcomes (two seeds agree, not drifting): "
        f"{np.mean([k is not None for v in vec.values() for k in v]):.0%}",
        "",
        "Vocabulary: distinct looks reachable from each start kind (number of rule sets)",
    ]
    for j, (name, _, _) in enumerate(kinds):
        cnt = Counter(v[j] for v in vec.values() if v[j] is not None)
        out.append(f"  {name}: {len(cnt)} looks")
        for k, c in cnt.most_common(8):
            out.append(f"      {c:5d}  {look_text(k)}")
    out.append("")
    distinct = len(set(vec.values()))
    out.append(f"distinct outcome vectors over the {len(kinds)} start kinds: {distinct}")
    # puzzles: z zones (distinct start kinds) with the looks produced by an intended solution
    rng = np.random.default_rng(a.seed)
    idx = {s: i for i, s in enumerate(sets)}
    cols = [
        np.array([hash(vec[s][j]) if vec[s][j] is not None else -1 for s in sets])
        for j in range(len(kinds))
    ]
    start_look = [relabel_look(look(*_start_x(kind, a), a), p) for _, kind, p in kinds]
    out.append("")
    out.append(
        "Puzzles: zones = distinct start kinds, targets = what a hidden 3-reaction set does to "
        "them (each target reliable and different from doing nothing); solutions = rule sets "
        "giving every target"
    )
    three = [s for s in sets if nre[s] == 3]
    for z in (2, 3, 4):
        nsol, minre, decoys, examples = [], [], [], []
        tries = 0
        while len(nsol) < a.puzzles and tries < 50 * a.puzzles:
            tries += 1
            s = three[int(rng.integers(len(three)))]
            js = sorted(rng.choice(len(kinds), z, replace=False).tolist())
            if any(vec[s][j] is None or vec[s][j] == start_look[j] for j in js):
                continue
            if len({vec[s][j] for j in js}) < min(z, 2):
                continue
            ok = np.ones(len(sets), bool)
            hits = np.zeros(len(sets), int)
            for j in js:
                m = cols[j] == hash(vec[s][j])
                ok &= m
                hits += m
            sol = [sets[i] for i in np.flatnonzero(ok)]
            nsol.append(len(sol))
            mr = min(nre[t] for t in sol)
            minre.append(mr)
            decoys.append(int((hits == z - 1).sum()))
            if mr == 3 and len(sol) <= 3 and len(examples) < 4:
                zs = "; ".join(f"{kinds[j][0]} -> {look_text(vec[s][j])}" for j in js)
                examples.append(f"      [{len(sol)} solution(s), e.g. {sol[0]}]  {zs}")
        nsol, minre, decoys = np.array(nsol), np.array(minre), np.array(decoys)
        out.append(
            f"  {z} zones ({len(nsol)} puzzles): solutions median {np.median(nsol):.0f} "
            f"(<= 5 in {np.mean(nsol <= 5):.0%}, unique in {np.mean(nsol == 1):.0%}); fewest "
            f"reactions needed 1 / 2 / 3: {np.mean(minre == 1):.0%} / {np.mean(minre == 2):.0%} / "
            f"{np.mean(minre == 3):.0%}; near misses (all zones but one) median "
            f"{np.median(decoys):.0f}"
        )
        out += examples
    _ = idx
    (d / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def _start_x(kind: str, a) -> tuple:
    """Shares, change rate and unlike-neighbour density of an untouched canonical start."""
    p = {"F": [1, 0, 0], "S": [1 - a.seeds, a.seeds, 0], "M": [0.5, 0.5, 0], "T": [1 / 3] * 3}[kind]
    x = np.array(p, dtype=float)
    return x, 0.0, 1.0 - float((x**2).sum())


def main() -> None:
    global _P
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=None, help="analyse an existing table.csv")
    ap.add_argument("--max-reactions", type=int, default=3)
    ap.add_argument("--w", type=int, default=30)
    ap.add_argument("--t-max", type=float, default=150.0)
    ap.add_argument("--seeds", type=float, default=0.03, help="share of seed cells in start S")
    ap.add_argument("--alive", type=float, default=0.02)
    ap.add_argument("--frozen", type=float, default=0.002)
    ap.add_argument("--puzzles", type=int, default=2000, help="random puzzles per zone count")
    ap.add_argument("--max-sets", type=int, default=0, help="smoke test: random subset")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    par = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    d = new_run_dir("ruletable", "find" if a.table else "table", par)
    table = a.table
    if table is None:
        rx = [r for r in reactions("edge", 3) if r[1] != r[2]]
        sets = [
            tuple(sorted(c))
            for m in range(1, a.max_reactions + 1)
            for c in itertools.combinations(rx, m)
        ]
        if a.max_sets:
            rng = np.random.default_rng(a.seed)
            sets = [sets[i] for i in sorted(rng.choice(len(sets), a.max_sets, replace=False))]
        _P = {"g": hex_rect(a.w, a.w), "t_max": a.t_max, "seeds": a.seeds}
        print(f"{len(sets)} rule sets x {len(STARTS)} starts x {len(SEEDS)} seeds", flush=True)
        rows, t0 = [], time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            jobs = list(enumerate(sets))
            for k, (i, res) in enumerate(pool.imap_unordered(run_set, jobs, chunksize=8)):
                for r in res:
                    rows.append([text(sets[i]), *r])
                if (k + 1) % 500 == 0:
                    print(f"  {k + 1}/{len(sets)} sets, {time.time() - t0:.0f} s", flush=True)
        table = d / "table.csv"
        hdr = ["reactions", "start", "seed", "x0", "x1", "x2", "rate", "rho", "same"]
        write_csv(table, hdr, rows)
        print(f"table saved to {table} ({time.time() - t0:.0f} s)", flush=True)
    if not a.max_sets:
        analyse(table, a, d)
    print(f"saved {d}")


if __name__ == "__main__":
    main()

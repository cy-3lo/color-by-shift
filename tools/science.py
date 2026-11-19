"""Headless science checks (CLAUDE.md, Python twin, checks 1-4). CSV + PNG snapshots + summary
in runs/science/<check>_<timestamp>/.

    python tools/science.py voter|rps|chiral|levels|terrain|all [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import (  # noqa: E402
    Reaction,
    ReactionSet,
    Sim,
    hex_torus,
    initial_colours,
    integrate,
    load_level,
    make_sim,
    rhs,
)
from mosaic_model.raster import hex_pixel_map, render_rgb  # noqa: E402

sys.path.insert(0, str(ROOT / "toy"))

from palette import palette_rgb, ui_colours  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

CFG = tomllib.loads((ROOT / "toy" / "viewer.toml").read_text())


# ---- observables ---------------------------------------------------------------------------
def unlike_edges(sim: Sim) -> float:
    c = sim.colours
    e = sim.graph.edges
    return float(np.mean(c[e[:, 0]] != c[e[:, 1]]))


def vertex_stats(sim: Sim) -> tuple[float, float]:
    """(fraction of trichromatic vertices, chirality (n_cw - n_ccw) / (n_cw + n_ccw)),
    cw meaning colours 0, 1, 2 read clockwise."""
    c = sim.colours.astype(np.int64)
    v = sim.graph.vertices
    a, b, d = c[v[:, 0]], c[v[:, 1]], c[v[:, 2]]
    tri = (a != b) & (b != d) & (a != d) & (a < 3) & (b < 3) & (d < 3)
    cw = tri & (((a + 1) % 3) == b)
    n_tri = int(tri.sum())
    n_cw = int(cw.sum())
    chi = (2 * n_cw - n_tri) / n_tri if n_tri else 0.0
    return n_tri / len(v), chi


def trace(
    sim: Sim,
    t_max: float,
    dt_rec: float,
    stop_when_one_left: bool = True,
    extra: Callable[[Sim], list[float]] | None = None,
) -> list[list[float]]:
    """Run to t_max, recording [t, fractions..., extra...] every dt_rec."""
    rows = []
    next_rec = sim.t
    while True:
        if sim.t >= next_rec - 1e-12:
            rows.append([sim.t, *sim.fractions().tolist(), *(extra(sim) if extra else [])])
            next_rec += dt_rec
        if sim.t >= t_max:
            break
        if stop_when_one_left and (sim.counts() > 0).sum() <= 1:
            rows.append([sim.t, *sim.fractions().tolist(), *(extra(sim) if extra else [])])
            break
        sim.sweep()
    return rows


def first_extinction(rows: list[list[float]], palette: int) -> float:
    for r in rows:
        if min(r[1 : 1 + palette]) == 0.0:
            return r[0]
    return float("nan")


def snapshot(sim: Sim, path: Path, size: float = 5.0) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pal = palette_rgb(CFG, sim.palette)
    pm = hex_pixel_map(sim.graph, size, border=CFG["grid"]["border"])
    rgb = render_rgb(pm, sim.colours, pal, ui_colours(CFG, None, len(pal))["bg"])
    pygame.image.save(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), str(path))


def _summary(d: Path, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    (d / "summary.txt").write_text(text)
    print(text)


# ---- check 5: voter model with fixed cells and a wall (exact harmonic benchmark) ----------
def harmonic(g, edges: np.ndarray, fixed: np.ndarray, value: np.ndarray, world) -> np.ndarray:
    """Discrete harmonic function on the world cells: u = value on fixed cells, and on free
    cells u_i = mean of u over its neighbours along `edges` (holes are not connected, so
    walls and open sides are no-flux boundaries). Dense solve."""
    free = np.flatnonzero(world & ~fixed)
    pos = -np.ones(g.n_cells, dtype=np.int64)
    pos[free] = np.arange(len(free))
    a = np.zeros((len(free), len(free)))
    b = np.zeros(len(free))
    for i, j in edges.tolist():
        for u, v in ((i, j), (j, i)):
            if pos[u] < 0:
                continue
            a[pos[u], pos[u]] += 1.0
            if pos[v] >= 0:
                a[pos[u], pos[v]] -= 1.0
            else:
                b[pos[u]] += value[v]
    out = np.where(fixed, value, np.nan).astype(float)
    out[free] = np.linalg.solve(a, b)
    return out


def grey_png(g, u: np.ndarray, hole: np.ndarray, path: Path, size: float = 10.0) -> None:
    """u in [0, 1] as grey levels (white = 1), holes in mid blue."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pm = hex_pixel_map(g, size)
    lev = np.clip(np.nan_to_num(u), 0, 1)
    lut = np.stack([lev * 255] * 3, axis=1).astype(np.uint8)
    lut[hole] = (70, 90, 150)
    lut = np.vstack([lut, [30, 30, 36]])
    rgb = lut[np.where(pm < 0, len(u), pm)]
    pygame.image.save(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), str(path))


def check_terrain(quick: bool) -> None:
    """Voter model (A|B -> A|A, B|A -> B|B) on an open n x n grid, left column fixed at 0,
    right column fixed at 1, a wall in the middle column with a gap at the bottom. The voter
    model is linear in the mean: d<u_i>/dt = sum_j (<u_j> - <u_i>), u_i = 1[s_i = 0], so the
    stationary mean is the discrete harmonic function with u = 1 / 0 on the fixed columns
    and no flux through the wall and the open sides. Exact; the only error is statistical."""
    p = {
        "n": 24,
        "wall_rows": 0.6,
        "seeds": 2 if quick else 4,
        "t_burn": 500.0,
        "t_run": 4000.0 if quick else 40000.0,
        "blocks": 10,
    }
    d = new_run_dir("science", "terrain", p)
    n = p["n"]
    g = hex_torus(n, periodic=False)
    r, c = np.arange(g.n_cells) // n, np.arange(g.n_cells) % n
    fixed = (c == 0) | (c == n - 1)
    hole = (c == n // 2) & (r < int(p["wall_rows"] * n))
    rs = ReactionSet(
        [Reaction("edge", [0, 1], [0, 0], 1.0), Reaction("edge", [1, 0], [1, 1], 1.0)], 2
    )
    start = np.where(c < n // 2, 0, 1)
    blocks = []
    sim = None
    for seed in range(p["seeds"]):
        sim = Sim(g, rs, start, seed=seed + 500, holes=hole, fixed=fixed)
        sim.sweep(int(p["t_burn"] * sim.K))
        per_block = int(p["t_run"] / p["blocks"])
        for _ in range(p["blocks"]):
            acc = np.zeros(g.n_cells)
            for _ in range(per_block):  # one sample per time unit
                sim.sweep(int(sim.K))
                acc += sim.colours == 0
            blocks.append(acc / per_block)
    world = ~hole
    u_th = harmonic(g, sim.edge_arr, fixed, (c == 0).astype(float), world)
    bl = np.array(blocks)
    u = bl.mean(axis=0)
    se = bl.std(axis=0, ddof=1) / np.sqrt(len(bl))
    free = world & ~fixed
    z = (u - u_th)[free] / np.maximum(se[free], 1e-9)
    err = np.abs(u - u_th)[free]
    left, right = free & (c < n // 2), free & (c > n // 2)
    rows = [[i, int(r[i]), int(c[i]), u_th[i], u[i], se[i]] for i in np.flatnonzero(free)]
    write_csv(d / "terrain_voter.csv", ["cell", "row", "col", "u_harmonic", "u_grid", "se"], rows)
    grey_png(g, u_th, hole, d / "harmonic.png")
    grey_png(g, u, hole, d / "grid_mean.png")
    _summary(
        d,
        [
            "Voter model with fixed columns (left = 0, right = 1) and a wall with a gap:",
            f"grid {n} x {n} open, wall {int(hole.sum())} cells, {int(free.sum())} free cells,",
            f"{p['seeds']} seeds x {p['blocks']} blocks of {p['t_run'] / p['blocks']:g} time"
            " units (one sample per unit time).",
            "u = time-averaged probability of colour 0 against the discrete harmonic function.",
            f"  max |u - u_h| = {err.max():.4f}, rms = {np.sqrt((err**2).mean()):.4f}",
            f"  mean standard error (block means) = {se[free].mean():.4f}",
            f"  z = (u - u_h) / se: mean {z.mean():+.2f}, rms {np.sqrt((z**2).mean()):.2f}"
            f" (expect about 0 and 1), |z| > 3 in {(np.abs(z) > 3).mean():.1%} of cells",
            f"  mean u left of the wall {u[left].mean():.4f} (harmonic {u_th[left].mean():.4f}),"
            f" right {u[right].mean():.4f} (harmonic {u_th[right].mean():.4f})",
            "PNG: harmonic.png and grid_mean.png (white = colour 0 always, wall in blue).",
        ],
    )


# ---- check 1: voter --------------------------------------------------------------------
def check_voter(quick: bool) -> None:
    p = {"n": 64 if quick else 128, "seeds": 2 if quick else 3, "t_max": 100.0 if quick else 500.0}
    d = new_run_dir("science", "voter", p)
    rs = ReactionSet(
        [Reaction("edge", [0, 1], [0, 0], 1.0), Reaction("edge", [1, 0], [1, 1], 1.0)], 2
    )
    g = hex_torus(p["n"])
    rows = []
    for seed in range(p["seeds"]):
        sim = Sim(g, rs, initial_colours(g.n_cells, [0.5, 0.5], seed), seed=seed + 100)
        next_t = 0.5
        while sim.t < p["t_max"]:
            sim.sweep()
            if sim.t >= next_t:
                rows.append([seed, sim.sweeps, sim.t, unlike_edges(sim), sim.fractions()[0]])
                next_t *= 1.25
        snapshot(sim, d / f"voter_seed{seed}_t{int(sim.t)}.png", 3.0 if p["n"] > 64 else 5.0)
    write_csv(d / "voter.csv", ["seed", "sweep", "t", "rho_unlike", "x0"], rows)
    a = np.array(rows, dtype=float)
    late = a[a[:, 2] >= 5.0]
    t, rho = late[:, 2], late[:, 3]
    # 1/rho vs ln t (voter, d = 2) against log rho vs log t (power law)
    c1 = np.polyfit(np.log(t), 1 / rho, 1)
    r1 = np.corrcoef(np.log(t), 1 / rho)[0, 1] ** 2
    c2 = np.polyfit(np.log(t), np.log(rho), 1)
    r2 = np.corrcoef(np.log(t), np.log(rho))[0, 1] ** 2
    lines = [
        f"Voter model, hex torus n={p['n']}, {p['seeds']} seeds, t <= {p['t_max']}",
        f"K = {rs.max_propensity()} (one sweep = {1 / rs.max_propensity()} time units)",
        "rho(t), fraction of unlike edges, mean over seeds:",
    ]
    for tt in [1, 5, 20, 50, 100, 200, 500]:
        sel = np.abs(np.log(a[:, 2] / tt)) < 0.12
        if sel.any():
            lines.append(f"  t ~ {tt:4d}: rho = {a[sel, 3].mean():.4f}")
    lines += [
        f"fit 1/rho = {c1[0]:.3f} ln t + {c1[1]:.3f}   R^2 = {r1:.4f}",
        f"fit rho ~ t^{c2[0]:.3f}                     R^2 = {r2:.4f}",
        "expected (recalled): slow decay rho ~ 1/ln t in 2D, much slower than the t^(-1/2)",
        "of curvature-driven coarsening.",
    ]
    _summary(d, lines)


# ---- check 2: rock-paper-scissors with exchange ---------------------------------------------
def rps_set(eps: float) -> ReactionSet:
    rx = [
        Reaction("edge", [0, 1], [0, 0], 1.0),
        Reaction("edge", [1, 2], [1, 1], 1.0),
        Reaction("edge", [2, 0], [2, 2], 1.0),
    ]
    if eps > 0:
        rx += [
            Reaction("edge", [0, 1], [1, 0], eps),
            Reaction("edge", [1, 2], [2, 1], eps),
            Reaction("edge", [2, 0], [0, 2], eps),
        ]
    return ReactionSet(rx, 3)


def check_rps(quick: bool) -> None:
    p = {
        "grids": [[32, 60.0]] if quick else [[32, 1000.0], [64, 300.0]],  # [n, t_max]
        "seeds": 2 if quick else 4,
        "eps": [0, 1, 10] if quick else [0, 0.1, 0.3, 1, 3, 10, 30],
    }
    d = new_run_dir("science", "rps", p)
    rows, lines = (
        [],
        [
            "Cyclic invasion 0>1>2>0 (rate 1) + exchange of every unlike pair at rate eps,",
            f"hex torus, uniform start, {p['seeds']} seeds; 'mixed' = well-mixed control, eps = 0.",
            "n    t_max  eps      mode    extinct/seeds  mean t_ext  rho_unlike(end)  wall s",
        ],
    )
    for n, t_max in p["grids"]:
        g = hex_torus(n)
        cases = [(e, False) for e in p["eps"]] + [(0, True)]
        for eps, mixed in cases:
            ext, rho_end, t0 = [], [], time.perf_counter()
            for seed in range(p["seeds"]):
                cols = initial_colours(g.n_cells, [1 / 3] * 3, seed)
                sim = Sim(g, rps_set(eps), cols, seed + 7, mixed)
                r = trace(sim, t_max, 1.0, extra=lambda s: [unlike_edges(s)])
                rows += [[n, eps, int(mixed), seed, *row] for row in r]
                ext.append(first_extinction(r, 3))
                rho_end.append(r[-1][-1])
                if seed == 0:
                    tag = "mixed" if mixed else f"eps{eps:g}"
                    snapshot(sim, d / f"rps_n{n}_{tag}_t{int(sim.t)}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = np.nanmean(ext) if n_ext else float("nan")
            lines.append(
                f"{n:<4d} {t_max:<6g} {eps:<8g} {'mixed' if mixed else 'grid':7s} "
                f"{n_ext}/{p['seeds']:<13d}{mt:10.1f}  {np.mean(rho_end):15.3f}  "
                f"{time.perf_counter() - t0:7.1f}"
            )
    write_csv(
        d / "rps.csv", ["n", "eps", "mixed", "seed", "t", "x0", "x1", "x2", "rho_unlike"], rows
    )
    lines += [
        "Expected (Reichenbach, Mobilia & Frey 2007, qualitatively): coexistence at low mobility,",
        "loss of species at high mobility. Their model has empty sites (May-Leonard); ours is",
        "the zero-sum cyclic invasion, whose mean field has neutral cycles (x0 x1 x2 conserved).",
    ]
    _summary(d, lines)


# ---- check 2b: May-Leonard (colour 3 is the gap left by selection) ---------------------
def may_leonard_set(eps: float, exchange: str = "all", sigma: float = 1.0, mu: float = 1.0):
    """Species 0, 1, 2 and gap colour 3: selection i|i+1 -> i|E, reproduction i|E -> i|i,
    exchange at rate eps between all unlike pairs ("all") or only with empty ("holes")."""
    e = 3
    rx = []
    for i in range(3):
        rx.append(Reaction("edge", [i, (i + 1) % 3], [i, e], sigma))
    for i in range(3):
        rx.append(Reaction("edge", [i, e], [i, i], mu))
    if eps > 0:
        pairs = [(0, 1), (1, 2), (2, 0)] if exchange == "all" else []
        pairs += [(i, e) for i in range(3)]
        rx += [Reaction("edge", [a, b], [b, a], eps) for a, b in pairs]
    return ReactionSet(rx, 4)


def check_mayleonard(quick: bool) -> None:
    p = {
        "n": 64,
        "seeds": 1 if quick else 2,
        "t_max": 60.0 if quick else 200.0,
        "eps": [0, 1] if quick else [0, 0.3, 1, 3, 10],
        "exchange": ["all", "holes"],
    }
    d = new_run_dir("science", "mayleonard", p)
    g = hex_torus(p["n"])
    rows, lines = (
        [],
        [
            "May-Leonard: selection i|i+1 -> i|E, reproduction i|E -> i|i (rates 1), exchange eps",
            f"n={p['n']}, start 1/4 each, {p['seeds']} seeds, t <= {p['t_max']}.",
            "exchange  eps    extinct  mean t_ext  x_empty(end)  rho_unlike(end)",
        ],
    )
    for ex in p["exchange"]:
        for eps in p["eps"]:
            if ex == "holes" and eps == 0:
                continue
            ext, xe, rho = [], [], []
            for seed in range(p["seeds"]):
                sim = Sim(
                    g,
                    may_leonard_set(eps, ex),
                    initial_colours(g.n_cells, [0.25] * 4, seed),
                    seed + 3,
                )
                r = trace(sim, p["t_max"], 1.0, extra=lambda s: [unlike_edges(s)])
                rows += [[ex, eps, seed, *row] for row in r]
                # extinction of a species (colours 0..2), not of the empty colour
                ext.append(next((row[0] for row in r if min(row[1:4]) == 0.0), float("nan")))
                xe.append(r[-1][4])
                rho.append(r[-1][5])
                if seed == 0:
                    snapshot(sim, d / f"ml_{ex}_eps{eps:g}_t{int(sim.t)}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = np.nanmean(ext) if n_ext else float("nan")
            lines.append(
                f"{ex:9s} {eps:<6g} {n_ext}/{p['seeds']}  {mt:10.1f}  "
                f"{np.mean(xe):12.3f}  {np.mean(rho):15.3f}"
            )
    write_csv(
        d / "mayleonard.csv",
        ["exchange", "eps", "seed", "t", "x0", "x1", "x2", "xE", "rho_unlike"],
        rows,
    )
    _summary(d, lines)


# ---- check 3: chiral vertex rotation --------------------------------------------------------
def check_chiral(quick: bool) -> None:
    p = {"n": 64, "seeds": 2 if quick else 3, "t_max": 60.0 if quick else 200.0, "k_rot": 0.3}
    d = new_run_dir("science", "chiral", p)
    g = hex_torus(p["n"])
    rot = [0, 1, 2], [2, 0, 1]
    cases = {
        "rot_chiral": [Reaction("vertex", *rot, 1.0, chiral=True)],
        "rot_achiral": [Reaction("vertex", *rot, 1.0, chiral=False)],
        "rps": rps_set(0).reactions,
        "rps+rot_chiral": rps_set(0).reactions + [Reaction("vertex", *rot, p["k_rot"], True)],
        "rps+rot_achiral": rps_set(0).reactions + [Reaction("vertex", *rot, p["k_rot"], False)],
    }
    rows = []
    lines = [
        f"Vertex rotation (0,1,2) -> (2,0,1), alone (k=1) and added to cyclic invasion "
        f"(k={p['k_rot']}), n={p['n']}, {p['seeds']} seeds, t <= {p['t_max']}.",
        "case               extinct  mean t_ext  rho_unlike  trichromatic  chirality  (end, mean)",
    ]
    for name, rx in cases.items():
        ext, ends = [], []
        for seed in range(p["seeds"]):
            rs = ReactionSet(
                [Reaction(r.shape, list(r.inp), list(r.out), r.rate, r.chiral) for r in rx], 3
            )
            sim = Sim(g, rs, initial_colours(g.n_cells, [1 / 3] * 3, seed), seed + 11)

            def extra(s: Sim) -> list[float]:
                return [unlike_edges(s), *vertex_stats(s)]

            r = trace(sim, p["t_max"], 1.0, extra=extra)
            rows += [[name, seed, *row] for row in r]
            ext.append(first_extinction(r, 3))
            ends.append(r[-1][4:])
            if seed == 0:
                snapshot(sim, d / f"chiral_{name}_t{int(sim.t)}.png")
        e = np.array(ends)
        n_ext = int(np.sum(~np.isnan(ext)))
        mt = np.nanmean(ext) if n_ext else float("nan")
        lines.append(
            f"{name:18s} {n_ext}/{p['seeds']}  {mt:10.1f}  {e[:, 0].mean():10.3f}  "
            f"{e[:, 1].mean():12.3f}  {e[:, 2].mean():9.3f}"
        )
    lines += [
        "Note: chirality is identically 0 on a torus: #cw - #ccw trichromatic triangles is the",
        "degree of the colouring map (Sperner-type argument), zero on a closed surface.",
        "Rotation alone: each firing is a bijection, so the uniform random state is stationary",
        "(chiral or not); no pattern forms.",
    ]
    write_csv(
        d / "chiral.csv",
        ["case", "seed", "t", "x0", "x1", "x2", "rho_unlike", "trichromatic", "chirality"],
        rows,
    )
    _summary(d, lines)


# ---- check 4: normal against shuffled for each level ----------------------------------------
def check_levels(quick: bool) -> None:
    paths = sorted((ROOT / "levels").glob("*.json"))
    p = {
        "seeds": 2 if quick else 4,
        "t_max": 60.0 if quick else 200.0,
        "levels": [x.name for x in paths],
    }
    d = new_run_dir("science", "levels", p)
    rows = []
    lines = [
        "Initial arrangement of each level, no switches; normal vs shuffled (well-mixed per",
        f"attempt), {p['seeds']} seeds, burn-in as in the level, then t <= {p['t_max']}.",
        "level mode     extinct  mean t_ext  final fractions (mean)     mean-field at t_max",
    ]
    for path in paths:
        lev = load_level(path)
        pal = lev.palette
        for mixed in (False, True):
            ext, fin, mf_end = [], [], []
            for seed in range(p["seeds"]):
                sim = make_sim(lev, seed=1000 + seed, shuffled=mixed)
                x0 = sim.fractions()
                r = trace(sim, p["t_max"], 1.0)
                rows += [[lev.id, int(mixed), seed, *row, *([0.0] * (4 - pal))] for row in r]
                ext.append(first_extinction(r, pal))
                fin.append(r[-1][1 : 1 + pal])
                mf = integrate(rhs(lev.reaction_set()), x0, np.array([0.0, p["t_max"]]), 1e-2)
                mf_end.append(mf[-1])
                if seed == 0:
                    snapshot(sim, d / f"{lev.id}_{'mixed' if mixed else 'grid'}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = np.nanmean(ext) if n_ext else float("nan")
            lines.append(
                f"{lev.id:5s} {'mixed' if mixed else 'grid':8s} {n_ext}/{p['seeds']}  {mt:10.1f}  "
                f"{np.round(np.mean(fin, axis=0), 3)!s:26s} {np.round(np.mean(mf_end, axis=0), 3)}"
            )
    write_csv(d / "levels.csv", ["level", "mixed", "seed", "t", "x0", "x1", "x2", "x3"], rows)
    _summary(d, lines)


# ---- check 5: world shape -----------------------------------------------------------------
WORLD_SPECS = {
    "torus64 (rect, glued)": {"type": "hex", "n": 64},
    "rect64 open": {"type": "hex", "n": 64, "periodic": False},
    "hextorus37 (glued)": {"type": "hexagon", "side": 37},
    "hexagon37 open": {"type": "hexagon", "side": 37, "periodic": False},
}


def chiral_count_set(mirror: bool = False) -> ReactionSet:
    """A count-changing chiral vertex reaction (0,1,2) -> (0,0,2), or its mirror image,
    with 0|2 -> 2|2 (0.3) and 2 -> 1 (0.2)."""
    inp, out = ([0, 2, 1], [0, 2, 0]) if mirror else ([0, 1, 2], [0, 0, 2])
    return ReactionSet(
        [
            Reaction("vertex", inp, out, 1.0, chiral=True),
            Reaction("edge", [0, 2], [2, 2], 0.3),
            Reaction("cell", [2], [1], 0.2),
        ],
        3,
    )


def check_shapes(quick: bool) -> None:
    from mosaic_model import make_graph

    p = {
        "seeds": 2 if quick else 4,
        "worlds": WORLD_SPECS,
        "voter_t": [5, 20, 50],
        "rps_eps": [0, 3] if quick else [0, 3, 10],
        "rps_t": 100.0 if quick else 300.0,
        "ml_t": 60.0 if quick else 200.0,
        "chiral_t": 30.0 if quick else 60.0,
    }
    d = new_run_dir("science", "shapes", p)
    rows = []
    lines = [f"World shape, {p['seeds']} seeds per case. Cells: "]
    graphs = {name: make_graph(spec) for name, spec in WORLD_SPECS.items()}
    lines[0] += ", ".join(f"{k} {g.n_cells}" for k, g in graphs.items())
    lines += [
        "Each column is one world; entries are means over seeds.",
        "",
    ]
    table: dict[str, dict[str, str]] = {}

    def put(row: str, world: str, val: str) -> None:
        table.setdefault(row, {})[world] = val

    for wname, g in graphs.items():
        boundary = sum(1 for x in g.neighbours() if len(x) < 6) / g.n_cells
        put("boundary cells", wname, f"{100 * boundary:.1f}%")
        # voter
        rs = ReactionSet(
            [Reaction("edge", [0, 1], [0, 0], 1.0), Reaction("edge", [1, 0], [1, 1], 1.0)], 2
        )
        rho = {t: [] for t in p["voter_t"]}
        for seed in range(p["seeds"]):
            sim = Sim(g, rs.copy(), initial_colours(g.n_cells, [0.5, 0.5], seed), seed + 100)
            for t in p["voter_t"]:
                while sim.t < t:
                    sim.sweep()
                rho[t].append(unlike_edges(sim))
        for t in p["voter_t"]:
            put(f"voter rho(t={t})", wname, f"{np.mean(rho[t]):.3f}")
        # zero-sum RPS with exchange
        for eps in p["rps_eps"]:
            ext = []
            for seed in range(p["seeds"]):
                sim = Sim(g, rps_set(eps), initial_colours(g.n_cells, [1 / 3] * 3, seed), seed + 7)
                r = trace(sim, p["rps_t"], 1.0)
                ext.append(first_extinction(r, 3))
                rows += [[wname, "rps", eps, seed, *row[:4]] for row in r]
                if seed == 0 and eps == 0:
                    snapshot(sim, d / f"rps_eps0_{g.label()}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = f" t~{np.nanmean(ext):.0f}" if n_ext else ""
            put(
                f"RPS eps={eps:g}: lost a species (t<={p['rps_t']:g})",
                wname,
                f"{n_ext}/{p['seeds']}{mt}",
            )
        # May-Leonard
        for ex in ("all", "holes"):
            ext, xe = [], []
            for seed in range(p["seeds"]):
                sim = Sim(
                    g,
                    may_leonard_set(1.0, ex),
                    initial_colours(g.n_cells, [0.25] * 4, seed),
                    seed + 3,
                )
                r = trace(sim, p["ml_t"], 1.0)
                ext.append(next((row[0] for row in r if min(row[1:4]) == 0.0), float("nan")))
                xe.append(r[-1][4])
                rows += [[wname, f"ml_{ex}", 1.0, seed, *row[:4]] for row in r]
                if seed == 0:
                    snapshot(sim, d / f"ml_{ex}_{g.label()}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            put(f"May-Leonard {ex} eps=1: lost a species", wname, f"{n_ext}/{p['seeds']}")
            put(f"May-Leonard {ex} eps=1: x3 at end", wname, f"{np.mean(xe):.3f}")
        # chiral count-changing reaction and its mirror
        for mirror in (False, True):
            xs, chis = [], []
            for seed in range(p["seeds"]):
                sim = Sim(
                    g,
                    chiral_count_set(mirror),
                    initial_colours(g.n_cells, [1 / 3] * 3, seed),
                    seed + 9,
                )
                while sim.t < p["chiral_t"]:
                    sim.sweep()
                xs.append(sim.fractions())
                chis.append(vertex_stats(sim)[1])
            tag = "mirror" if mirror else "clockwise"
            put(
                f"chiral {tag}: x0, x1, x2",
                wname,
                ", ".join(f"{v:.3f}" for v in np.mean(xs, axis=0)),
            )
            put(f"chiral {tag}: net chirality", wname, f"{np.mean(chis):+.3f}")
    names = list(graphs)
    w0 = max(len(r) for r in table) + 2
    lines.append(" " * w0 + "".join(f"{n:>24s}" for n in names))
    for rname, vals in table.items():
        lines.append(f"{rname:{w0}s}" + "".join(f"{vals.get(n, ''):>24s}" for n in names))
    write_csv(d / "shapes.csv", ["world", "case", "eps", "seed", "t", "x0", "x1", "x2"], rows)
    _summary(d, lines)


# ---- check 6: relative handedness ------------------------------------------------------------
def check_handedness(quick: bool) -> None:
    """T05 mechanism: a chiral vertex reaction creates (0,1,2) triangles from background
    colour 3, another consumes clockwise ones. Same vs opposite handedness, against mean field
    (identical for both)."""
    p = {"n": 64, "seeds": 2 if quick else 4, "t_max": 20.0 if quick else 40.0}
    d = new_run_dir("science", "handedness", p)
    g = hex_torus(p["n"])
    base = [
        Reaction("edge", [0, 3], [0, 0], 0.15),
        Reaction("cell", [0], [3], 0.1),
        Reaction("cell", [1], [3], 0.3),
        Reaction("cell", [2], [3], 0.3),
    ]
    consume = Reaction("vertex", [0, 1, 2], [3, 3, 3], 1.0)
    cases = {
        "creates clockwise (same hand)": Reaction("vertex", [0, 3, 3], [0, 1, 2], 0.5),
        "creates anticlockwise (opposite)": Reaction("vertex", [0, 3, 3], [0, 2, 1], 0.5),
    }
    x0 = np.array([0.1, 0.1, 0.1, 0.7])
    lines = [
        f"Relative handedness, n={p['n']}, {p['seeds']} seeds, t = {p['t_max']:g}; mean (s.e.)",
        "case                               x0             x1             x3       cw%    ccw%",
    ]
    rows = []
    for name, create in cases.items():
        res = []
        for seed in range(p["seeds"]):
            rs = ReactionSet(
                [
                    *(Reaction(r.shape, list(r.inp), list(r.out), r.rate) for r in base),
                    create,
                    consume,
                ],
                4,
            )
            sim = Sim(g, rs, initial_colours(g.n_cells, x0, seed), seed + 41)
            while sim.t < p["t_max"]:
                sim.sweep()
            c = sim.colours.astype(np.int64)
            v = g.vertices
            a, b, e = c[v[:, 0]], c[v[:, 1]], c[v[:, 2]]
            tri = (a < 3) & (b < 3) & (e < 3) & (a != b) & (b != e) & (a != e)
            cw = tri & ((a + 1) % 3 == b)
            res.append([*sim.fractions(), cw.mean(), (tri & ~cw).mean()])
            rows.append([name, seed, *res[-1]])
            if seed == 0:
                snapshot(sim, d / f"handedness_{'same' if 'same' in name else 'opposite'}.png")
        m = np.mean(res, axis=0)
        se = np.std(res, axis=0) / np.sqrt(len(res))
        lines.append(
            f"{name:34s} {m[0]:.3f} ({se[0]:.3f})  {m[1]:.3f} ({se[1]:.3f})  "
            f"{m[3]:.3f} ({se[3]:.3f})  {100 * m[4]:5.2f}  {100 * m[5]:5.2f}"
        )
    rs = ReactionSet([*base, cases["creates clockwise (same hand)"], consume], 4)
    mf = integrate(rhs(rs), x0, np.array([0.0, p["t_max"]]), 1e-2)[-1]
    lines.append(f"mean field (identical for both)    x = {np.round(mf, 3)}")
    write_csv(d / "handedness.csv", ["case", "seed", "x0", "x1", "x2", "x3", "cw", "ccw"], rows)
    _summary(d, lines)


CHECKS = {
    "voter": check_voter,
    "rps": check_rps,
    "mayleonard": check_mayleonard,
    "chiral": check_chiral,
    "levels": check_levels,
    "shapes": check_shapes,
    "handedness": check_handedness,
    "terrain": check_terrain,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", choices=[*CHECKS, "all"])
    ap.add_argument("--quick", action="store_true", help="small, fast version")
    a = ap.parse_args()
    for name, fn in CHECKS.items():
        if a.check in (name, "all"):
            t0 = time.perf_counter()
            fn(a.quick)
            print(f"[{name}] {time.perf_counter() - t0:.1f} s\n")


if __name__ == "__main__":
    main()

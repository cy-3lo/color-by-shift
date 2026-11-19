"""Painting prototypes on an open hex grid: regrowth and dissolve (with the movie in reverse).

    python tools/painting.py assets/paintings/great_wave.jpg --mode regrow [--fixed 0.05]
    python tools/painting.py assets/paintings/great_wave.jpg --mode dissolve
                             [--rules voter,majority,both,cycle] [--outlines 0,0.12]
                             [--t-end 300] [--smooth 0.5]

The painting is quantised to --colours colours (k-means in RGB, dark to light) on an open hex
grid of width --w.

regrow    keep a fraction --fixed of the cells as fixed tiles, scramble the rest with the
          painting's fractions, run the rules; accuracy = fraction of free cells with the
          painting's colour; reference "voronoi" = colour of the nearest tile (graph distance).
dissolve  start from the painting itself, run the rules to --t-end, record --rec-frames
          frames at geometrically spaced times, and write the movie in reverse as GIFs
          (the painting emerges at the end):
          reverse_linear.gif  frames equally spaced in sim time;
          reverse_change.gif  frames equally spaced in "amount of change": arclength of the
                              curve (accuracy, interface density), each normalised by its
                              range over the run.
          Plus forward.gif (linear), sheet.png (6 stills of each reverse), measures.csv.

Outlines (hermetic edges, "cuts"): for each value q in --outlines, the fraction q of edges
with the largest colour difference in a blurred copy of the painting are cut, so nothing
reacts across them. q = 0: no outlines. Outlines leak where the drawing's lines have gaps.

Rule sets (every rate k = 1; n colours, all ordered pairs A != B):
  voter     edge A|B -> A|A: a cell copies a neighbour (coarsening, no surface tension).
  majority  vertex (A, A, B) -> (A, A, A), achiral: the minority cell of a vertex converts
            (curvature-driven, sharpens interfaces).
  both      voter + majority.
  cycle     cyclic invasion A|A+1 -> A|A (mod n) plus exchange A|B -> B|A (mobility).
  explicit  a set written as "00>01;01>00;01>10" (colours 0 = darkest; one digit = cell).
Images, GIFs and CSV in runs/painting/<name>_<time>/.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import Reaction, ReactionSet, Sim  # noqa: E402
from mosaic_model.graph import SQRT3, hex_rect  # noqa: E402
from mosaic_model.level import terrain_to_json  # noqa: E402
from mosaic_model.observables import interface_density, pace_frames  # noqa: E402
from mosaic_model.raster import cut_mask, hex_pixel_map, render_rgb  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

BG = (24, 24, 28)
LINE = (20, 20, 24)  # outline colour


def sample(img: Image.Image, w: int, blur: float = 0.0) -> tuple:
    """Open hex grid of width w with the painting's aspect; RGB of the image at cell centres
    (after a Gaussian blur of `blur` cell widths, if > 0)."""
    W, H = img.size
    h = max(2, round(w * SQRT3 * H / W / 1.5))  # columns sqrt(3) apart, rows 1.5 apart
    g = hex_rect(w, h)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur * W / w))
    x, y = g.centers[:, 0], g.centers[:, 1]
    u = (x - x.min()) / (x.max() - x.min())
    v = (y - y.min()) / (y.max() - y.min())
    arr = np.asarray(img.convert("RGB"), dtype=float)
    px = np.clip((u * (W - 1)).round().astype(int), 0, W - 1)
    py = np.clip((v * (H - 1)).round().astype(int), 0, H - 1)
    return g, arr[py, px]


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        c2 = np.array([x[lab == j].mean(axis=0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(c2, c):
            break
        c = c2
    order = np.argsort(c.sum(axis=1))  # dark to light
    c = c[order]
    lab = np.argsort(order)[lab]
    return c.round().astype(np.uint8), lab


def rules(name: str, n: int) -> list[Reaction]:
    out = []
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    if name in ("voter", "both"):
        out += [Reaction("edge", [a, b], [a, a], 1.0) for a, b in pairs]
    if name in ("majority", "both"):
        out += [Reaction("vertex", [a, a, b], [a, a, a], 1.0, chiral=False) for a, b in pairs]
    if name == "cycle":
        out += [Reaction("edge", [a, (a + 1) % n], [a, a], 1.0) for a in range(n)]
        out += [Reaction("edge", [a, b], [b, a], 1.0) for a, b in pairs if a < b]
    if ">" in name:  # explicit set, e.g. "00>01;01>00;0>1" (one digit = cell reaction)
        for part in name.split(";"):
            i, o = part.strip().split(">")
            shape = "cell" if len(i) == 1 else "edge"
            out.append(Reaction(shape, [int(c) for c in i], [int(c) for c in o], 1.0))
    if not out:
        raise ValueError(f"unknown rule set {name}")
    return out


def outline_cuts(g, rgb_blur: np.ndarray, q: float) -> np.ndarray:
    """The fraction q of edges with the largest colour difference (blurred RGB)."""
    if q <= 0:
        return np.zeros((0, 2), np.int64)
    e = g.edges
    d = np.linalg.norm(rgb_blur[e[:, 0]] - rgb_blur[e[:, 1]], axis=1)
    thr = np.quantile(d, 1 - q)
    return e[d > thr]


def regions(g, cuts: np.ndarray) -> np.ndarray:
    """Sizes of the connected regions left by the cuts (largest first)."""
    n = g.n_cells
    key = set(map(tuple, np.sort(cuts, axis=1).tolist()))
    nb = [[] for _ in range(n)]
    for a, b in g.edges.tolist():
        if (min(a, b), max(a, b)) not in key:
            nb[a].append(b)
            nb[b].append(a)
    lab = -np.ones(n, int)
    sizes = []
    for s in range(n):
        if lab[s] >= 0:
            continue
        lab[s] = len(sizes)
        q, k = deque([s]), 0
        while q:
            a = q.popleft()
            k += 1
            for b in nb[a]:
                if lab[b] < 0:
                    lab[b] = lab[s]
                    q.append(b)
        sizes.append(k)
    return np.sort(np.array(sizes))[::-1]


def voronoi(g, target: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Colour of the nearest fixed tile (multi-source BFS on the cell graph)."""
    nb = g.neighbours()
    col = np.full(g.n_cells, -1)
    q = deque(np.flatnonzero(fixed).tolist())
    col[fixed] = target[fixed]
    while q:
        a = q.popleft()
        for b in nb[a]:
            if col[b] < 0:
                col[b] = col[a]
                q.append(b)
    return col


def image(pm, col, pal, mask=None, line=None) -> np.ndarray:
    c = np.asarray(col).astype(np.int64).copy()
    if mask is not None:
        c[~mask] = len(pal)  # background
    rgb = render_rgb(pm, np.where(c >= len(pal), -1, c), pal, BG)
    if line is not None:
        rgb[line] = LINE
    return rgb


def save(pm, col, pal, path: Path, mask=None, line=None) -> None:
    Image.fromarray(image(pm, col, pal, mask, line)).save(path)


# ---- dissolve and reverse --------------------------------------------------------------------
def dissolve(job: dict) -> dict:
    """Run one rule set from the painting; return recorded frames and measures."""
    g, target, n = job["g"], job["target"], job["n"]
    sim = Sim(g, ReactionSet(rules(job["rules"], n), n), target, job["seed"], cuts=job["cuts"])
    t0 = time.time()
    frames, ts = [sim.colours.copy()], [0.0]
    # geometric recording times: the painting dissolves in a few time units, then all slows
    for t in np.geomspace(job["rec_t0"], job["t_end"], job["rec_frames"]):
        sim.advance(t - sim.t)
        frames.append(sim.colours.copy())
        ts.append(sim.t)
    f = np.array(frames)
    acc = (f == target[None]).mean(axis=1)
    rho = np.array([interface_density(fr, g.edges) for fr in f])
    print(f"  {job['tag']}: {time.time() - t0:.0f} s, end accuracy {acc[-1]:.3f}", flush=True)
    return {**job, "frames": f, "t": np.array(ts), "acc": acc, "rho": rho}


def pick_frames(r: dict, how: str, m: int) -> np.ndarray:
    """m frame indices from start to end, equally spaced in time or in amount of change."""
    return pace_frames(r["t"], [r["acc"], r["rho"]], how, m)


def write_gif(path: Path, imgs: list[np.ndarray], ms: int, hold_first: int, hold_last: int):
    pil = [Image.fromarray(a).convert("P", palette=Image.Palette.ADAPTIVE, colors=16) for a in imgs]
    durs = [ms] * len(pil)
    durs[0] += hold_first
    durs[-1] += hold_last
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=durs, loop=0, optimize=True)


def export_level(path: Path, a, g, target, pal, reacts, cuts) -> None:
    """Level JSON for toy/viewer.py: the quantised painting as a drawn start (one character
    per cell, colour index), the painting's palette, the given reactions, no goals."""
    w = a.w
    rows = [
        "".join(str(int(c)) for c in target[r * w : (r + 1) * w]) for r in range(len(target) // w)
    ]
    level = {
        "id": a.image.stem,
        "title": f"{a.image.stem} ({len(pal)} colours)",
        "palette": len(pal),
        "colours": ["#{:02X}{:02X}{:02X}".format(*(int(v) for v in p)) for p in pal],
        "map": {"rows": rows, "legend": {str(c): c for c in range(len(pal))}},
        "reactions": [
            {"shape": r.shape, "in": list(r.inp), "out": list(r.out), "rate": r.rate}
            | ({"chiral": False} if r.shape == "vertex" and not r.chiral else {})
            for r in reacts
        ],
        "locked": [],
        "start": {"seed": a.seed, "burn_in_sweeps": 0},
        "goals": [],
        "fail": [],
        "meta": {"from": "tools/painting.py", "image": str(a.image), "smooth": a.smooth},
    }
    ter = terrain_to_json(g, np.zeros(g.n_cells, bool), np.zeros(g.n_cells, bool), target, cuts)
    if ter:
        level["terrain"] = ter
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(level, separators=(",", ":")).replace('","', '",\n"') + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--mode", choices=("regrow", "dissolve"), default="dissolve")
    ap.add_argument("--colours", type=int, default=4)
    ap.add_argument("--rules", default="voter,majority,both,cycle")
    ap.add_argument("--outlines", default="0,0.12", help="fractions of edges cut")
    ap.add_argument("--blur", type=float, default=0.7, help="blur for outlines, in cell widths")
    ap.add_argument("--w", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fixed", type=float, default=0.05, help="regrow: fraction of tiles")
    ap.add_argument("--t", default="0,5,20,80", help="regrow: snapshot times")
    ap.add_argument("--t-end", type=float, default=300.0, help="dissolve: run length")
    ap.add_argument("--rec-t0", type=float, default=0.02, help="dissolve: first frame time")
    ap.add_argument("--rec-frames", type=int, default=500, help="frames, geometric in time")
    ap.add_argument("--smooth", type=float, default=0.5, help="blur before sampling, in cells")
    ap.add_argument("--gif-frames", type=int, default=120)
    ap.add_argument("--gif-ms", type=int, default=60)
    ap.add_argument("--size", type=float, default=3.0, help="hex radius in px for PNG")
    ap.add_argument("--gif-size", type=float, default=3.0, help="hex radius in px for GIF")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--export-level",
        type=Path,
        default=None,
        help="write a viewer level (painting as drawn start, first rule set, first outlines)",
    )
    a = ap.parse_args()
    d = new_run_dir(
        "painting",
        f"{a.image.stem}_{a.mode}",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
    )
    img = Image.open(a.image)
    g, rgb = sample(img, a.w, a.smooth)
    _, rgb_blur = sample(img, a.w, a.blur)
    pal, target = kmeans(rgb, a.colours, a.seed)
    n = a.colours
    frac = np.bincount(target, minlength=n) / g.n_cells
    pm, dx, dy = hex_pixel_map(g, a.size, with_local=True)
    qs = [float(v) for v in a.outlines.split(",")]
    cuts = {q: outline_cuts(g, rgb_blur, q) for q in qs}
    lines = {q: cut_mask(g, pm, dx, dy, cuts[q]) if len(cuts[q]) else None for q in qs}
    save(pm, target, pal, d / "target.png")
    out = [
        f"{a.image.name}: hex {a.w} x {g.n_cells // a.w} = {g.n_cells} cells, {n} colours "
        f"(fractions {np.round(frac, 3).tolist()}); palette "
        f"{[tuple(int(v) for v in p) for p in pal]}; chance {float((frac**2).sum()):.3f}"
    ]
    for q in qs:
        if len(cuts[q]):
            save(pm, target, pal, d / f"outlines_q{q:g}.png", line=lines[q])
            rs = regions(g, cuts[q])
            out.append(
                f"outlines q = {q:g}: {len(cuts[q])} cuts, {len(rs)} regions, largest "
                f"{rs[:5].tolist()} cells, {int((rs >= 10).sum())} regions of 10+ cells"
            )
    print("\n".join(out), flush=True)
    names = a.rules.split(",")
    rng = np.random.default_rng(a.seed)
    if a.export_level:
        export_level(a.export_level, a, g, target, pal, rules(names[0], n), cuts[qs[0]])
        print(f"level written to {a.export_level}")
        return

    if a.mode == "regrow":
        fixed = rng.random(g.n_cells) < a.fixed
        free = ~fixed
        save(pm, target, pal, d / "tiles.png", mask=fixed)
        vor = voronoi(g, target, fixed)
        save(pm, vor, pal, d / "voronoi.png")
        times = [float(t) for t in a.t.split(",")]
        out.append(
            f"fixed tiles {fixed.sum()} ({fixed.mean():.1%}); voronoi "
            f"{float((vor[free] == target[free]).mean()):.3f}; accuracy at t = {times}"
        )
        rows = []
        for q in qs:
            for name in names:
                start = target.copy()
                start[free] = rng.choice(n, size=int(free.sum()), p=frac)
                rs_ = ReactionSet(rules(name, n), n)
                sim = Sim(g, rs_, start, a.seed, fixed=fixed, cuts=cuts[q])
                accs = []
                for t in times:
                    sim.advance(t - sim.t)
                    acc = float((sim.colours[free] == target[free]).mean())
                    accs.append(acc)
                    rows.append([name, q, t, round(acc, 4)])
                    save(pm, sim.colours, pal, d / f"{name}_q{q:g}_t{t:g}.png", line=lines[q])
                out.append(f"  {name:9s} q {q:<5g}" + "  ".join(f"{v:.3f}" for v in accs))
                print(out[-1], flush=True)
        write_csv(d / "accuracy.csv", ["rules", "outlines", "t", "accuracy"], rows)
    else:
        jobs = [
            {
                "tag": f"{name.replace('>', '-').replace(';', '_').replace(' ', '')}_q{q:g}",
                "rules": name,
                "g": g,
                "target": target,
                "n": n,
                "cuts": cuts[q],
                "q": q,
                "seed": a.seed,
                "t_end": a.t_end,
                "rec_t0": a.rec_t0,
                "rec_frames": a.rec_frames,
            }
            for q in qs
            for name in names
        ]
        with mp.get_context("fork").Pool(min(a.workers, len(jobs))) as pool:
            res = pool.map(dissolve, jobs)
        gpm, gdx, gdy = hex_pixel_map(g, a.gif_size, with_local=True)
        rows = []
        out.append("accuracy (fraction of cells with the painting's colour) and interface density")
        out.append(
            f"at t = 0, 10, 50, {a.t_end:g}; change-paced GIF: share of frames in the "
            "first 10 % of sim time (of the forward movie)"
        )
        for r in res:
            sub = d / r["tag"]
            sub.mkdir()
            gline = cut_mask(g, gpm, gdx, gdy, r["cuts"]) if len(r["cuts"]) else None
            for k in range(len(r["t"])):
                rows.append([r["tag"], float(r["t"][k]), float(r["acc"][k]), float(r["rho"][k])])
            for how in ("linear", "change"):
                idx = pick_frames(r, how, a.gif_frames)
                imgs = [image(gpm, r["frames"][i], pal, line=gline) for i in idx[::-1]]
                write_gif(sub / f"reverse_{how}.gif", imgs, a.gif_ms, 600, 2000)
                if how == "change":
                    early = float((r["t"][idx] <= 0.1 * a.t_end).mean())
            idx = pick_frames(r, "linear", a.gif_frames)
            fwd = [image(gpm, r["frames"][i], pal, line=gline) for i in idx]
            write_gif(sub / "forward.gif", fwd, a.gif_ms, 2000, 600)
            # contact sheet: 6 stills of each reverse movie (rows: linear, change)
            stills = []
            for how in ("linear", "change"):
                idx = pick_frames(r, how, 6)[::-1]
                stills.append(np.hstack([image(gpm, r["frames"][i], pal, line=gline) for i in idx]))
            Image.fromarray(np.vstack(stills)).save(sub / "sheet.png")
            at = [int(np.argmin(np.abs(r["t"] - t))) for t in (0, 10, 50, a.t_end)]
            out.append(
                f"  {r['tag']:14s} acc "
                + " ".join(f"{r['acc'][i]:.2f}" for i in at)
                + "  rho "
                + " ".join(f"{r['rho'][i]:.3f}" for i in at)
                + f"  early {early:.2f}"
            )
        write_csv(d / "measures.csv", ["run", "t", "accuracy", "interface"], rows)
    (d / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[1:]))
    print(f"saved {d}")


if __name__ == "__main__":
    main()

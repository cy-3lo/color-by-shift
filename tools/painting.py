"""Painting regrowth prototype: quantise a painting onto an open hex grid, keep a few percent
of its cells as fixed tiles, scramble the rest, and let a reaction set regrow the picture.

    python tools/painting.py assets/paintings/great_wave.jpg [--colours 5] [--fixed 0.05]
                             [--rules voter,majority,both] [--t 0,5,20,80] [--w 150]

Rule sets (every rate k = 1; n colours, all ordered pairs A != B):
  voter     edge A|B -> A|A: a cell copies a neighbour. With fixed tiles the mean colour
            field is the discrete harmonic interpolation of the tiles (blurred, noisy).
  majority  vertex (A, A, B) -> (A, A, A), achiral: a cell in the minority of a vertex
            joins the majority; sharpens interfaces, cannot create a colour absent locally.
  both      voter + majority.
Measured: accuracy = fraction of free cells with the painting's colour, over time; reference
"voronoi" = each free cell takes the colour of its nearest tile (graph distance), the
best a local copy rule can be expected to do. Images and CSV in runs/painting/<name>_<time>/.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import Reaction, ReactionSet, Sim  # noqa: E402
from mosaic_model.graph import SQRT3, hex_rect  # noqa: E402
from mosaic_model.raster import hex_pixel_map, render_rgb  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

BG = (24, 24, 28)


def sample(img: Image.Image, w: int) -> tuple:
    """Open hex grid of width w with the painting's aspect; RGB of the image at cell centres."""
    W, H = img.size
    h = max(2, round(w * SQRT3 * H / W / 1.5))  # columns sqrt(3) apart, rows 1.5 apart
    g = hex_rect(w, h)
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
    return out


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


def save(pm, col, pal, path: Path, mask=None) -> None:
    c = np.asarray(col).copy()
    if mask is not None:
        c[~mask] = len(pal)  # background
    rgb = render_rgb(pm, np.where(c >= len(pal), -1, c), pal, BG)
    Image.fromarray(rgb).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--colours", type=int, default=5)
    ap.add_argument("--fixed", type=float, default=0.05)
    ap.add_argument("--rules", default="voter,majority,both")
    ap.add_argument("--t", default="0,5,20,80")
    ap.add_argument("--w", type=int, default=150)
    ap.add_argument("--size", type=float, default=3.0, help="hex radius in px for images")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = new_run_dir("painting", a.image.stem, {**vars(a), "image": str(a.image)})
    g, rgb = sample(Image.open(a.image), a.w)
    pal, target = kmeans(rgb, a.colours, a.seed)
    n = a.colours
    rng = np.random.default_rng(a.seed)
    fixed = rng.random(g.n_cells) < a.fixed
    free = ~fixed
    frac = np.bincount(target, minlength=n) / g.n_cells
    pm = hex_pixel_map(g, a.size)
    save(pm, target, pal, d / "target.png")
    save(pm, target, pal, d / "tiles.png", mask=fixed)
    vor = voronoi(g, target, fixed)
    save(pm, vor, pal, d / "voronoi.png")
    acc_vor = float((vor[free] == target[free]).mean())
    times = [float(t) for t in a.t.split(",")]
    lines = [
        f"{a.image.name}: hex {a.w} x {g.n_cells // a.w} = {g.n_cells} cells, {n} colours "
        f"(fractions {np.round(frac, 3).tolist()}), fixed tiles {fixed.sum()} "
        f"({fixed.mean():.1%}); palette {[tuple(int(v) for v in p) for p in pal]}",
        f"reference: chance (painting fractions) {float((frac**2).sum()):.3f}; "
        f"voronoi from tiles {acc_vor:.3f}",
        "accuracy on free cells at t = " + ", ".join(f"{t:g}" for t in times),
    ]
    rows = []
    for name in a.rules.split(","):
        start = target.copy()
        start[free] = rng.choice(n, size=int(free.sum()), p=frac)  # scrambled, same fractions
        sim = Sim(g, ReactionSet(rules(name, n), n), start, a.seed, fixed=fixed)
        accs = []
        for t in times:
            sim.advance(t - sim.t)
            acc = float((sim.colours[free] == target[free]).mean())
            accs.append(acc)
            rows.append([name, t, round(acc, 4)])
            save(pm, sim.colours, pal, d / f"{name}_t{t:g}.png")
        lines.append(f"  {name:9s} " + "  ".join(f"{v:.3f}" for v in accs))
        print(lines[-1], flush=True)
    write_csv(d / "accuracy.csv", ["rules", "t", "accuracy"], rows)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))
    print(f"saved {d}")


if __name__ == "__main__":
    main()

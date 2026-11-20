"""Pictures and rule presets for the painting toy (toy/painting_toy.py); no pygame here.

A picture is an open hex grid with a start colour per cell and an RGB palette ordered from
dark to light (colour 0 = darkest), so that rule presets can refer to colours by rank.

Paintings are quantised to n colours by k-means in RGB on a lightly blurred copy (blur of
`smooth` cell widths, which removes paper and canvas speckle). Generated pictures:
  random    independent uniform colours
  regular   the 3-colouring of the hex lattice, (q - r) mod n: with 3 colours no two
            neighbours share a colour; with other n, diagonal stripes one cell wide
  stripes   horizontal bands
  rings     concentric rings around the centre
  destijl   an original composition of rectangles and dark lines (random subdivision,
            seeded), in the spirit of De Stijl; not a copy of any painting

Rule presets (every rate k = 1; n colours, 0 = darkest, n - 1 = lightest):
  voter       a cell copies a neighbour (edge A|B -> A|A, all ordered pairs)
  smoothing   voter + majority
  majority    the minority cell of a vertex joins the two others ((A, A, B) -> (A, A, A))
  shuffle     neighbours swap colours (A|B -> B|A): every colour count is conserved
  rps         rock-paper-scissors around all n colours (A|A+1 -> A|A)
  spirals     May-Leonard: the lightest colour is empty space; the others reproduce into
              it (A|E -> A|A), eat the next one cyclically (A|B -> A|E), and every pair
              of neighbours swaps (mobility); spirals at k = 1 on 17 000 cells
  living sea  the darkest colour comes alive and the rest holds (00>01, 01>00, 01>10)
  ghost       everything thins into scattered dots (00>01, 01>10, 11>12)
  swirl       chiral rotation of every vertex with three different colours (A, B, C) ->
              (C, A, B), clockwise only
  none        no preset (only the player's own reactions)
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from mosaic_model import Reaction
from mosaic_model.graph import SQRT3, Graph, hex_rect

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Picture:
    key: str
    title: str
    graph: Graph
    colours: np.ndarray  # start colour of each cell
    palette: np.ndarray  # (n, 3) uint8, dark to light


# ---- grids and paintings ---------------------------------------------------------------------
def grid_for(aspect: float, cells: int) -> Graph:
    """Open hex rectangle of about `cells` cells with width / height = aspect (pixels)."""
    w = max(4, round((cells * 1.5 * aspect / SQRT3) ** 0.5))
    h = max(2, round(w * SQRT3 / aspect / 1.5))
    return hex_rect(w, h)


def sample(img: Image.Image, g: Graph, blur: float = 0.0) -> np.ndarray:
    """RGB of the image at the cell centres of g (after a blur of `blur` cell widths)."""
    W, H = img.size
    w = int(g.axial[:, 0].max() - g.axial[:, 0].min()) + 1
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur * W / w))
    x, y = g.centers[:, 0], g.centers[:, 1]
    u = (x - x.min()) / (x.max() - x.min())
    v = (y - y.min()) / (y.max() - y.min())
    arr = np.asarray(img.convert("RGB"), dtype=float)
    px = np.clip((u * (W - 1)).round().astype(int), 0, W - 1)
    py = np.clip((v * (H - 1)).round().astype(int), 0, H - 1)
    return arr[py, px]


def kmeans(
    x: np.ndarray, k: int, seed: int, iters: int = 50, restarts: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """(palette uint8 dark to light, label of each sample). k-means++ starts, `restarts`
    runs, the one with the smallest within-cluster sum of squares is kept."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):  # k-means++: next centre with probability ~ squared distance
            d2 = ((x[:, None, :] - np.array(c)[None]) ** 2).sum(axis=2).min(axis=1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())] if d2.sum() > 0 else x[0])
        c = np.array(c, dtype=float)
        for _ in range(iters):
            lab = ((x[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
            c2 = np.array(
                [x[lab == j].mean(axis=0) if (lab == j).any() else c[j] for j in range(k)]
            )
            if np.allclose(c2, c):
                break
            c = c2
        lab = ((x[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        sse = float(((x - c[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c, lab)
    _, c, lab = best
    order = np.argsort(c.sum(axis=1))
    c = c[order]
    lab = np.argsort(order)[lab]
    return c.round().astype(np.uint8), lab


def lum(rgb) -> float:
    r, g, b = (float(v) / 255 for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---- generated pictures ----------------------------------------------------------------------
def generated(kind: str, g: Graph, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = g.axial[:, 0], g.axial[:, 1]
    x, y = g.centers[:, 0], g.centers[:, 1]
    if kind == "random":
        return rng.integers(0, n, g.n_cells)
    if kind == "regular":
        return (q - r) % n
    if kind == "stripes":
        h = int(r.max() - r.min()) + 1
        return ((r - r.min()) * n // h) % n
    if kind == "rings":
        d = np.hypot(x - x.mean(), y - y.mean())
        return (d / (d.max() / (2 * n))).astype(int) % n
    if kind == "destijl":
        return destijl(g, n, rng)
    raise ValueError(f"unknown generated picture {kind!r}")


def destijl(g: Graph, n: int, rng) -> np.ndarray:
    """Rectangles split at random, separated by lines of the darkest colour; most fills are
    the lightest colour, a few take the middle colours."""
    x, y = g.centers[:, 0], g.centers[:, 1]
    boxes = [(x.min() - 1, x.max() + 1, y.min() - 1, y.max() + 1)]
    for _ in range(9):
        k = int(rng.integers(len(boxes)))
        x0, x1, y0, y1 = boxes.pop(k)
        if (x1 - x0) > (y1 - y0) * rng.uniform(0.6, 1.6):
            s = rng.uniform(x0 + 0.3 * (x1 - x0), x1 - 0.3 * (x1 - x0))
            boxes += [(x0, s, y0, y1), (s, x1, y0, y1)]
        else:
            s = rng.uniform(y0 + 0.3 * (y1 - y0), y1 - 0.3 * (y1 - y0))
            boxes += [(x0, x1, y0, s), (x0, x1, s, y1)]
    col = np.full(g.n_cells, n - 1)
    mids = list(range(1, n - 1)) or [n - 1]
    for k, (x0, x1, y0, y1) in enumerate(boxes):
        inside = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        col[inside] = mids[k % len(mids)] if rng.random() < 0.45 else n - 1
    lw = 1.6  # line half-width, cell units
    for x0, x1, y0, y1 in boxes:
        inside = (x > x0 - lw) & (x < x1 + lw) & (y > y0 - lw) & (y < y1 + lw)
        edge = (np.abs(x - x0) < lw) | (np.abs(x - x1) < lw) | (np.abs(y - y0) < lw)
        edge |= np.abs(y - y1) < lw
        col[inside & edge] = 0
    return col


def generated_palette(name: str, n: int) -> np.ndarray:
    """n colours of a viewer palette (toy/viewer.toml), sorted dark to light."""
    from palette import palette_rgb

    cfg = tomllib.loads((ROOT / "toy" / "viewer.toml").read_text())
    pal = palette_rgb(cfg, n, name)
    return pal[np.argsort([lum(c) for c in pal])]


def load_picture(entry: dict, n: int, cells: int, smooth: float, seed: int, gen_palette: str):
    """Picture from a [[pictures]] entry of toy/toy.toml."""
    if "file" in entry:
        img = Image.open(ROOT / entry["file"])
        g = grid_for(img.size[0] / img.size[1], cells)
        pal, lab = kmeans(sample(img, g, smooth), n, seed)
        title = f"{entry.get('artist', '')}, {entry.get('title', entry['key'])}"
        if entry.get("year"):
            title += f" ({entry['year']})"
        return Picture(entry["key"], title, g, lab.astype(np.int64), pal)
    g = grid_for(float(entry.get("aspect", 1.3)), cells)
    col = generated(entry["generated"], g, n, seed)
    pal = generated_palette(entry.get("palette", gen_palette), n)
    return Picture(entry["key"], entry.get("title", entry["key"]), g, col.astype(np.int64), pal)


# ---- rules -----------------------------------------------------------------------------------
PRESETS = [
    "voter",
    "smoothing",
    "majority",
    "shuffle",
    "rps",
    "spirals",
    "living sea",
    "ghost",
    "swirl",
    "none",
]
DESCRIPTIONS = {
    "voter": "a cell copies a neighbour",
    "smoothing": "copy a neighbour, and the minority of a corner joins the majority",
    "majority": "the odd cell of a corner joins the other two",
    "shuffle": "neighbours swap colours: every colour count is conserved",
    "rps": "rock-paper-scissors around all colours: each eats the next",
    "spirals": "lightest = empty space; the others reproduce, eat the next, and move",
    "living sea": "the darkest colour comes alive, the rest holds",
    "ghost": "everything thins into scattered dots",
    "swirl": "corners with three colours rotate clockwise",
    "none": "only your own reactions",
}


def E(i, o) -> Reaction:
    return Reaction("edge", list(i), list(o), 1.0)


def preset(name: str, n: int) -> list[Reaction]:
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    if name == "voter":
        return [E((a, b), (a, a)) for a, b in pairs]
    if name == "majority":
        return [Reaction("vertex", [a, a, b], [a, a, a], 1.0, chiral=False) for a, b in pairs]
    if name == "smoothing":
        return preset("voter", n) + preset("majority", n)
    if name == "shuffle":
        return [E((a, b), (b, a)) for a, b in pairs if a < b]
    if name == "rps":
        return [E((a, (a + 1) % n), (a, a)) for a in range(n)]
    if name == "spirals":
        e, sp = n - 1, list(range(n - 1))
        out = [E((a, e), (a, a)) for a in sp]
        if len(sp) >= 2:
            out += [E((a, sp[(k + 1) % len(sp)]), (a, e)) for k, a in enumerate(sp)]
        return out + preset("shuffle", n)
    if name == "living sea":
        return parse_rules("00>01 ; 01>00 ; 01>10", n)
    if name == "ghost":
        return parse_rules("00>01 ; 01>10 ; 11>12", n)
    if name == "swirl":
        out = []
        for a, b, c in itertools.combinations(range(n), 3):
            for t in ((a, b, c), (a, c, b)):
                out.append(Reaction("vertex", list(t), [t[2], t[0], t[1]], 1.0))
        return out
    if name == "none":
        return []
    raise ValueError(f"unknown preset {name!r}")


def parse_rules(s: str, n: int) -> list[Reaction]:
    """'01>00 ; 0>1 ; 012>120': one digit = cell, two = edge, three = chiral vertex. Colours
    beyond n - 1 are wrapped (mod n)."""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        i, o = part.split(">")
        shape = {1: "cell", 2: "edge", 3: "vertex"}[len(i)]
        if len(o) != len(i):
            raise ValueError(f"reaction {part!r}: input and output lengths differ")
        out.append(Reaction(shape, [int(c) % n for c in i], [int(c) % n for c in o], 1.0))
    return out


def rule_text(rx: list[Reaction]) -> str:
    return " ; ".join(f"{''.join(map(str, r.inp))}>{''.join(map(str, r.out))}" for r in rx)

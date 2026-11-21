"""Mondrian-like compositions: a box cut into coloured rectangles by straight cuts.

Generated from a seed, not copied from any painting. Used as pictures for the learning
levels of the click-to-shift game (tools/make_ladder.py).
"""

from __future__ import annotations

import itertools

import numpy as np

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1

CUTS = (1 / 3, 0.4, 0.5, 0.6, 2 / 3)


def cut_box(n: int, aspect: float, rng, min_side: float = 0.16, tries: int = 2000) -> list[Rect]:
    """n rectangles tiling [0, aspect] x [0, 1]: a rectangle drawn with probability
    proportional to its squared area is cut in two, mostly across its longer side, at one of
    a few simple fractions; no side shorter than min_side."""
    rects: list[Rect] = [(0.0, 0.0, float(aspect), 1.0)]
    for _ in range(tries):
        if len(rects) == n:
            return rects
        area = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects])
        k = int(rng.choice(len(rects), p=area**2 / (area**2).sum()))
        x0, y0, x1, y1 = rects[k]
        along_x = (x1 - x0 >= y1 - y0) == (rng.random() < 0.8)
        f = CUTS[int(rng.integers(len(CUTS)))]
        if along_x:
            xm = round(x0 + f * (x1 - x0), 4)
            two = [(x0, y0, xm, y1), (xm, y0, x1, y1)]
        else:
            ym = round(y0 + f * (y1 - y0), 4)
            two = [(x0, y0, x1, ym), (x0, ym, x1, y1)]
        if min(min(r[2] - r[0], r[3] - r[1]) for r in two) >= min_side:
            rects[k : k + 1] = two
    raise ValueError(f"cannot cut the box into {n} rectangles with sides >= {min_side}")


def touching(rects: list[Rect], eps: float = 1e-6) -> list[tuple[int, int]]:
    """Pairs of rectangles that share a piece of side (not only a corner)."""
    out = []
    for (i, a), (j, b) in itertools.combinations(enumerate(rects), 2):
        over_x = min(a[2], b[2]) - max(a[0], b[0])
        over_y = min(a[3], b[3]) - max(a[1], b[1])
        if (abs(over_x) < eps and over_y > eps) or (abs(over_y) < eps and over_x > eps):
            out.append((i, j))
    return out


def colour_rects(rects: list[Rect], n_colours: int, rng, tries: int = 300) -> list[int]:
    """One colour per rectangle, every colour used; colour 0 (the white of the palette) is
    drawn three times as often as each other one. Among `tries` draws, keeps the one with the
    largest share of touching rectangles of different colours, then the most even areas."""
    if n_colours > len(rects):
        raise ValueError("more colours than rectangles")
    pairs = touching(rects)
    area = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects])
    p = np.array([3.0] + [1.0] * (n_colours - 1))
    best, best_score = None, None
    for _ in range(tries):
        col = rng.choice(n_colours, size=len(rects), p=p / p.sum())
        if len(set(col.tolist())) < n_colours:
            continue
        differ = sum(col[i] != col[j] for i, j in pairs) / max(len(pairs), 1)
        smallest = min(area[col == c].sum() for c in range(n_colours)) / area.sum()
        score = (round(differ, 6), smallest)
        if best_score is None or score > best_score:
            best, best_score = col.tolist(), score
    if best is None:
        raise ValueError("no colouring uses every colour")
    return [int(c) for c in best]


def composition(n_rects: int, n_colours: int, aspect: float, seed: int, min_side: float = 0.16):
    """Rectangles and their colours. As shapes of a foam: the first rectangle is the
    background (region 0), the others are polygons (see `polygons`)."""
    rng = np.random.default_rng(seed)
    rects = cut_box(n_rects, aspect, rng, min_side)
    return rects, colour_rects(rects, n_colours, rng)


def polygons(rects: list[Rect]) -> list[list[list[float]]]:
    """Corners of every rectangle but the first, for mosaic_model.foam.make_foam(shapes=...)."""
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in rects[1:]]

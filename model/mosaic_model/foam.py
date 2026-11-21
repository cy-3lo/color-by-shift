"""Reversible mosaic on a foam: irregular cells, swaps by groups of disjoint pairs.

A foam is a set of sites in the box [0, aspect] x [0, 1]; a cell is the region nearer to its
site than to any other (Voronoi). Two cells that share a side are neighbours (an edge).
Every edge belongs to exactly one group, and the edges of a group share no cell (a proper
edge colouring of the neighbour graph), so all the pairs of a group can swap at once.

A move names a group and two colours: every pair of the group that holds these two colours
swaps them. The decision depends only on the unordered colours of the pair, which the swap
does not change, so a move is its own inverse and any sequence is undone by the same moves
in reverse order. In play a move is given by one pair of neighbouring cells: its group and
its two colours.

A foam can follow drawn shapes: polygons cut the box into regions (0 = the background, k =
inside polygon k, later polygons on top) and a cell never crosses from one region to
another, because a pixel only looks for the nearest site of its own region. The outline of a
shape is then made of cell sides, and a picture is one colour per region.

No geometry library: cells are found on a pixel raster (nearest site), which also gives the
neighbours (cells whose pixels touch along at least `min_side` pixels).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .shift import flag


def region_map(shapes: list, width: int, height: int, aspect: float) -> np.ndarray:
    """(height, width) array: region of each pixel centre (even-odd rule per polygon)."""
    xs = ((np.arange(width) + 0.5) / width * aspect)[None, :]
    ys = ((np.arange(height) + 0.5) / height)[:, None]
    out = np.zeros((height, width), dtype=np.int32)
    for k, poly in enumerate(shapes):
        inside = np.zeros((height, width), bool)
        pts = np.asarray(poly, dtype=float)
        for (x0, y0), (x1, y1) in zip(pts, np.roll(pts, -1, axis=0), strict=True):
            if y0 == y1:
                continue
            cross = ((y0 <= ys) != (y1 <= ys)) & (xs < x0 + (ys - y0) * (x1 - x0) / (y1 - y0))
            inside ^= cross
        out[inside] = k + 1
    return out


def nearest_site(
    sites: np.ndarray,
    width: int,
    height: int,
    aspect: float,
    site_region: np.ndarray | None = None,
    reg: np.ndarray | None = None,
) -> np.ndarray:
    """(height, width) array: index of the site nearest to each pixel centre; with a region
    map `reg` of that size, nearest among the sites of the pixel's own region."""
    xs = ((np.arange(width) + 0.5) / width * aspect).astype(np.float32)
    ys = ((np.arange(height) + 0.5) / height).astype(np.float32)
    px, py = np.meshgrid(xs, ys)
    out = np.empty((height, width), dtype=np.int32)
    if reg is not None:
        groups = [(reg == r, np.nonzero(site_region == r)[0]) for r in np.unique(reg)]
    else:
        groups = [(np.ones((height, width), bool), np.arange(len(sites)))]
    for mask, idx in groups:
        if len(idx) == 0:
            raise ValueError("a region of the picture holds no site")
        x, y = px[mask], py[mask]
        sx, sy = sites[idx, 0].astype(np.float32), sites[idx, 1].astype(np.float32)
        near = np.empty(len(x), dtype=np.int32)
        step = max(1, 4_000_000 // len(idx))
        for k in range(0, len(x), step):
            d = (x[k : k + step, None] - sx[None, :]) ** 2
            d += (y[k : k + step, None] - sy[None, :]) ** 2
            near[k : k + step] = idx[d.argmin(axis=1)]
        out[mask] = near
    return out


def resample(reg: np.ndarray, width: int, height: int) -> np.ndarray:
    """A region raster at another size (nearest pixel)."""
    iy = np.minimum((np.arange(height) + 0.5) * reg.shape[0] / height, reg.shape[0] - 1)
    ix = np.minimum((np.arange(width) + 0.5) * reg.shape[1] / width, reg.shape[1] - 1)
    return reg[iy.astype(int)[:, None], ix.astype(int)[None, :]].astype(np.int32)


def neighbours(labels: np.ndarray, min_side: int) -> tuple[np.ndarray, np.ndarray]:
    """(edges (E, 2) with u < v, side length in pixels) of the cells of a label raster."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = a != b
        lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        pairs.append(np.stack([lo, hi], axis=1))
    uniq, count = np.unique(np.concatenate(pairs), axis=0, return_counts=True)
    keep = count >= min_side
    return uniq[keep].astype(np.int64), count[keep]


def colour_edges(edges: np.ndarray, n_cells: int, rng, tries: int = 300) -> np.ndarray:
    """Group of each edge such that the edges of a group share no cell, with as few groups as
    a greedy colouring finds over `tries` orders (busiest cells first, ties at random)."""
    deg = np.bincount(edges.ravel(), minlength=n_cells)
    weight = deg[edges[:, 0]] + deg[edges[:, 1]]
    best = None
    for _ in range(tries):
        order = np.lexsort((rng.random(len(edges)), -weight))
        used: list[set[int]] = [set() for _ in range(n_cells)]
        grp = np.empty(len(edges), dtype=np.int64)
        for i in order:
            u, v = edges[i]
            g = next(k for k in itertools.count() if k not in used[u] and k not in used[v])
            grp[i] = g
            used[u].add(g)
            used[v].add(g)
        if best is None or grp.max() < best.max():
            best = grp
    return best


@dataclass
class Foam:
    aspect: float
    sites: np.ndarray  # (n, 2) in [0, aspect] x [0, 1]
    edges: np.ndarray  # (E, 2) neighbouring cells, u < v
    group: np.ndarray  # (E,) group of each edge
    shapes: list = field(default_factory=list)  # polygons [[x, y], ...] the cells follow
    site_region: np.ndarray | None = None  # region of each site (0 = background)
    regions: np.ndarray | None = None  # or a raster of regions the cells follow (a painting)
    pairs: list[tuple[np.ndarray, np.ndarray]] = field(init=False)  # per group: (u, v)
    lookup: dict[tuple[int, int], int] = field(init=False)  # (u, v) -> group

    def __post_init__(self) -> None:
        self.sites = np.asarray(self.sites, dtype=float)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.group = np.asarray(self.group, dtype=np.int64)
        if self.site_region is None:
            self.site_region = np.zeros(len(self.sites), dtype=np.int64)
        self.site_region = np.asarray(self.site_region, dtype=np.int64)
        self.pairs = []
        for g in range(self.n_groups):
            e = self.edges[self.group == g]
            if len(np.unique(e)) != e.size:
                raise ValueError(f"group {g}: two pairs share a cell")
            self.pairs.append((e[:, 0], e[:, 1]))
        self.lookup = {
            (int(u), int(v)): int(g) for (u, v), g in zip(self.edges, self.group, strict=True)
        }

    @property
    def n_cells(self) -> int:
        return len(self.sites)

    @property
    def n_groups(self) -> int:
        return int(self.group.max()) + 1 if len(self.group) else 0

    @property
    def n_regions(self) -> int:
        if self.regions is not None:
            return int(self.regions.max()) + 1
        return len(self.shapes) + 1

    def region_raster(self, width: int, height: int) -> np.ndarray | None:
        """(height, width) array: the region of each pixel (None: a foam without regions)."""
        if self.regions is not None:
            return resample(self.regions, width, height)
        if self.shapes:
            return region_map(self.shapes, width, height, self.aspect)
        return None

    def raster(self, width: int, height: int) -> np.ndarray:
        """(height, width) array: the cell of each pixel."""
        reg = self.region_raster(width, height)
        return nearest_site(self.sites, width, height, self.aspect, self.site_region, reg)

    def group_of(self, u: int, v: int) -> int | None:
        """Group of the edge between two cells, None if they are not neighbours."""
        return self.lookup.get((min(u, v), max(u, v)))

    def swapping(self, col: np.ndarray, g: int, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """The pairs (u, v) of group g that hold colours a and b (none if a == b)."""
        u, v = self.pairs[g]
        cu, cv = col[u], col[v]
        m = (((cu == a) & (cv == b)) | ((cu == b) & (cv == a))) & (a != b)
        return u[m], v[m]

    def move(self, col: np.ndarray, g: int, a: int, b: int) -> np.ndarray:
        """Swap colours a and b across every pair of group g that holds them."""
        u, v = self.swapping(col, g, a, b)
        out = col.copy()
        out[u], out[v] = col[v], col[u]
        return out

    def to_json(self) -> dict:
        return {
            "aspect": self.aspect,
            "sites": [[round(float(x), 5), round(float(y), 5)] for x, y in self.sites],
            "edges": self.edges.tolist(),
            "group": self.group.tolist(),
            "shapes": [
                [[round(float(x), 5), round(float(y), 5)] for x, y in p] for p in self.shapes
            ],
            "site_region": self.site_region.tolist(),
        }


def make_foam(
    n: int,
    aspect: float,
    seed: int,
    relax: int = 4,
    width: int = 480,
    min_side: int = 5,
    shapes: list | None = None,
    min_sites: int = 3,
    regions: np.ndarray | None = None,
    region_sites: dict[int, int] | None = None,
) -> Foam:
    """Foam of n cells (a few more if a region would hold fewer than min_sites): random
    sites, `relax` rounds of moving each site to the centre of its cell (rounder, more even
    cells), neighbours and groups from a raster `width` pixels wide. With shapes (polygons)
    or a raster of regions (a quantised painting), the cells follow their outlines.
    region_sites = {region: least number of sites} overrides min_sites for thin regions that
    random sites would leave with a few very long cells."""
    rng = np.random.default_rng(seed)
    shapes = [np.asarray(p, dtype=float).tolist() for p in (shapes or [])]
    height = round(width / aspect)
    if regions is not None:
        reg = resample(regions, width, height)
    else:
        reg = region_map(shapes, width, height, aspect)

    def region_at(pts: np.ndarray) -> np.ndarray:
        ix = np.clip((pts[:, 0] / aspect * width).astype(int), 0, width - 1)
        iy = np.clip((pts[:, 1] * height).astype(int), 0, height - 1)
        return reg[iy, ix]

    sites = rng.random((n, 2)) * (aspect, 1.0)
    for r in np.unique(reg):  # every region gets a few sites
        lack = (region_sites or {}).get(int(r), min_sites) - int((region_at(sites) == r).sum())
        if lack > 0:
            ys, xs = np.nonzero(reg == r)
            k = rng.choice(len(xs), lack, replace=False)
            extra = np.stack([(xs[k] + 0.5) / width * aspect, (ys[k] + 0.5) / height], axis=1)
            sites = np.concatenate([sites, extra])
    site_region = region_at(sites)
    n = len(sites)
    yy, xx = np.mgrid[0:height, 0:width]
    px = (xx.ravel() + 0.5) / width * aspect
    py = (yy.ravel() + 0.5) / height
    for _ in range(relax):
        lab = nearest_site(sites, width, height, aspect, site_region, reg).ravel()
        cnt = np.maximum(np.bincount(lab, minlength=n), 1)
        new = np.stack(
            [np.bincount(lab, px, minlength=n) / cnt, np.bincount(lab, py, minlength=n) / cnt],
            axis=1,
        )
        ok = (region_at(new) == site_region) & (np.bincount(lab, minlength=n) > 0)
        sites = np.where(ok[:, None], new, sites)  # a centre outside its region: stay put
    lab = nearest_site(sites, width, height, aspect, site_region, reg)
    edges, _ = neighbours(lab, min_side)
    return Foam(aspect, sites, edges, colour_edges(edges, n, rng), shapes, site_region, regions)


def load_foam(path: str | Path) -> Foam:
    """Foam JSON; "regions_file" names a raster of regions in the same folder (.npy, or .npz
    with the array "regions")."""
    path = Path(path)
    d = json.loads(path.read_text())
    regions = np.load(path.parent / d["regions_file"]) if d.get("regions_file") else None
    if regions is not None and not isinstance(regions, np.ndarray):  # .npz (compressed)
        regions = regions["regions"]
    return Foam(
        float(d["aspect"]),
        np.array(d["sites"]),
        np.array(d["edges"]),
        d["group"],
        d.get("shapes", []),
        d.get("site_region"),
        regions,
    )


# ---- levels ----------------------------------------------------------------------------------
@dataclass
class FoamLevel:
    id: str
    foam: Foam
    colours: list[str]  # "#rrggbb" per colour
    target: np.ndarray  # the picture to recover
    scramble: list[tuple[int, int, int]]  # moves (group, a, b) that scrambled it, in order
    title: str = ""
    reward: dict | None = None  # shown once recovered: image, crop, anchor, artist, title

    @property
    def n(self) -> int:
        return len(self.colours)

    @property
    def par(self) -> int:
        return len(self.scramble)

    def scrambled(self) -> np.ndarray:
        col = self.target
        for g, a, b in self.scramble:
            col = self.foam.move(col, g, a, b)
        return col


def foam_picture(kind: str, foam: Foam, n: int) -> np.ndarray:
    """A flag-like picture (mosaic_model.shift.flag) on the cells of a foam."""
    return flag(kind, SimpleNamespace(centers=foam.sites), n).astype(np.int64)


def random_scramble(foam: Foam, target: np.ndarray, n: int, length: int, rng, min_pairs: int = 2):
    """`length` moves, each swapping at least min_pairs pairs and none undoing the one before."""
    col, out = target, []
    options = [(g, a, b) for g in range(foam.n_groups) for a in range(n) for b in range(a + 1, n)]
    while len(out) < length:
        good = [
            m for m in options if len(foam.swapping(col, *m)[0]) >= min_pairs and [m] != out[-1:]
        ]
        if not good:
            raise ValueError("no move changes enough cells: lower min_pairs")
        m = good[int(rng.integers(len(good)))]
        out.append(m)
        col = foam.move(col, *m)
    return out


def load_foam_level(path: str | Path, root: str | Path | None = None) -> FoamLevel:
    """Level JSON: {"id", "title", "foam": path of a foam JSON (relative to the repository
    root), "colours": ["#rrggbb", ...], "picture": {"regions": [colour of the background,
    of shape 1, ...]} or {"flag": kind} or {"cells": [...]},
    "scramble": [{"group": g, "pair": [a, b]}, ...], optional "reward": {"image": path,
    "crop", "anchor", "artist", "title", "year"} shown once the picture is recovered}."""
    path = Path(path)
    d = json.loads(path.read_text())
    root = Path(root) if root is not None else path.resolve().parents[2]
    foam = load_foam(root / d["foam"])
    n = len(d["colours"])
    pic = d["picture"]
    if "regions" in pic:  # one colour per region of the foam's shapes
        if len(pic["regions"]) != foam.n_regions:
            raise ValueError(f"{path}: {foam.n_regions} regions need as many colours")
        target = np.array(pic["regions"])[foam.site_region]
    elif "cells" in pic:
        target = np.array(pic["cells"])
    else:
        target = foam_picture(pic["flag"], foam, n)
    scr = [(int(m["group"]), int(m["pair"][0]), int(m["pair"][1])) for m in d["scramble"]]
    if len(target) != foam.n_cells or target.max() >= n:
        raise ValueError(f"{path}: picture does not fit the foam or the palette")
    if any(g >= foam.n_groups or max(a, b) >= n for g, a, b in scr):
        raise ValueError(f"{path}: scramble outside the groups or the palette")
    return FoamLevel(
        d["id"], foam, list(d["colours"]), target, scr, d.get("title", ""), d.get("reward")
    )


# ---- outlines of the cells (for drawing a cell that moves) -------------------------------------
def outlines(labels: np.ndarray, n_cells: int, k: int = 48) -> np.ndarray:
    """Outline of every cell of a raster of cell indices, as k points: array (n_cells, k, 2)
    of (x, y) in pixels of the raster (corners of pixels, so 0..width and 0..height).
    The sides of the pixels of a cell that face another cell are chained into loops, the
    loop enclosing the largest area is kept, and k points are placed at equal distances
    along it, clockwise on screen, starting where the outline crosses the horizontal line to
    the right of its centre: two cells then correspond point by point."""
    h, w = labels.shape
    lab = np.pad(labels.astype(np.int64), 1, constant_values=-1)
    core = lab[1:-1, 1:-1]
    yy, xx = np.mgrid[0:h, 0:w]
    cell, start, end = [], [], []
    corner = lambda x, y: y * (w + 1) + x  # noqa: E731
    # (neighbour, start corner, end corner): the cell stays on the right of the walk
    for other, (ax, ay), (bx, by) in (
        (lab[:-2, 1:-1], (0, 0), (1, 0)),  # top side, walked to the right
        (lab[1:-1, 2:], (1, 0), (1, 1)),  # right side, downward
        (lab[2:, 1:-1], (1, 1), (0, 1)),  # bottom side, to the left
        (lab[1:-1, :-2], (0, 1), (0, 0)),  # left side, upward
    ):
        m = core != other
        cell.append(core[m])
        start.append(corner(xx[m] + ax, yy[m] + ay))
        end.append(corner(xx[m] + bx, yy[m] + by))
    cell, start, end = (np.concatenate(v) for v in (cell, start, end))
    order = np.argsort(cell, kind="stable")
    cell, start, end = cell[order], start[order], end[order]
    bounds = np.searchsorted(cell, np.arange(n_cells + 1))
    out = np.zeros((n_cells, k, 2))
    for c in range(n_cells):
        i0, i1 = bounds[c], bounds[c + 1]
        if i1 == i0:
            raise ValueError(f"cell {c} is not on the raster")
        nxt: dict[int, list[int]] = {}
        for s, e in zip(start[i0:i1].tolist(), end[i0:i1].tolist(), strict=True):
            nxt.setdefault(s, []).append(e)
        best, best_area = None, -1.0
        while nxt:
            first = next(iter(nxt))
            loop, at = [first], first
            while True:
                ends = nxt[at]
                to = ends.pop()
                if not ends:
                    del nxt[at]
                if to == first:
                    break
                loop.append(to)
                at = to
            pts = np.array([(p % (w + 1), p // (w + 1)) for p in loop], dtype=float)
            x, y = pts[:, 0], pts[:, 1]
            area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            if area > best_area:
                best, best_area = pts, area
        # start to the right of the centre, then k points at equal distances along the loop
        centre = best.mean(axis=0)
        angle = np.arctan2(best[:, 1] - centre[1], best[:, 0] - centre[0])
        best = np.roll(best, -int(np.argmin(np.abs(angle))), axis=0)
        closed = np.vstack([best, best[:1]])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
        at_s = np.arange(k) * s[-1] / k
        out[c, :, 0] = np.interp(at_s, s, closed[:, 0])
        out[c, :, 1] = np.interp(at_s, s, closed[:, 1])
    return out

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


def nearest_site(sites: np.ndarray, width: int, height: int, aspect: float) -> np.ndarray:
    """(height, width) array: index of the site nearest to each pixel centre."""
    xs = (np.arange(width) + 0.5) / width * aspect
    ys = (np.arange(height) + 0.5) / height
    out = np.empty((height, width), dtype=np.int32)
    sx, sy = sites[:, 0].astype(np.float32), sites[:, 1].astype(np.float32)
    dx2 = (xs[:, None].astype(np.float32) - sx[None, :]) ** 2  # (width, n)
    step = max(1, 4_000_000 // (width * len(sites)))
    for y0 in range(0, height, step):
        dy2 = (ys[y0 : y0 + step, None].astype(np.float32) - sy[None, :]) ** 2  # (rows, n)
        out[y0 : y0 + step] = (dx2[None, :, :] + dy2[:, None, :]).argmin(axis=2)
    return out


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
    pairs: list[tuple[np.ndarray, np.ndarray]] = field(init=False)  # per group: (u, v)
    lookup: dict[tuple[int, int], int] = field(init=False)  # (u, v) -> group

    def __post_init__(self) -> None:
        self.sites = np.asarray(self.sites, dtype=float)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.group = np.asarray(self.group, dtype=np.int64)
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
        }


def make_foam(
    n: int,
    aspect: float,
    seed: int,
    relax: int = 4,
    width: int = 480,
    min_side: int = 5,
) -> Foam:
    """Foam of n cells: random sites, `relax` rounds of moving each site to the centre of its
    cell (rounder, more even cells), neighbours and groups from a raster `width` pixels wide."""
    rng = np.random.default_rng(seed)
    sites = rng.random((n, 2)) * (aspect, 1.0)
    height = round(width / aspect)
    yy, xx = np.mgrid[0:height, 0:width]
    px = (xx.ravel() + 0.5) / width * aspect
    py = (yy.ravel() + 0.5) / height
    for _ in range(relax):
        lab = nearest_site(sites, width, height, aspect).ravel()
        cnt = np.maximum(np.bincount(lab, minlength=n), 1)
        sites = np.stack(
            [np.bincount(lab, px, minlength=n) / cnt, np.bincount(lab, py, minlength=n) / cnt],
            axis=1,
        )
    edges, _ = neighbours(nearest_site(sites, width, height, aspect), min_side)
    return Foam(aspect, sites, edges, colour_edges(edges, n, rng))


def load_foam(path: str | Path) -> Foam:
    d = json.loads(Path(path).read_text())
    return Foam(float(d["aspect"]), np.array(d["sites"]), np.array(d["edges"]), d["group"])


# ---- levels ----------------------------------------------------------------------------------
@dataclass
class FoamLevel:
    id: str
    foam: Foam
    colours: list[str]  # "#rrggbb" per colour
    target: np.ndarray  # the picture to recover
    scramble: list[tuple[int, int, int]]  # moves (group, a, b) that scrambled it, in order
    title: str = ""

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
    root), "colours": ["#rrggbb", ...], "picture": {"flag": kind} or {"cells": [...]},
    "scramble": [{"group": g, "pair": [a, b]}, ...]}."""
    path = Path(path)
    d = json.loads(path.read_text())
    root = Path(root) if root is not None else path.resolve().parents[2]
    foam = load_foam(root / d["foam"])
    n = len(d["colours"])
    pic = d["picture"]
    target = np.array(pic["cells"]) if "cells" in pic else foam_picture(pic["flag"], foam, n)
    scr = [(int(m["group"]), int(m["pair"][0]), int(m["pair"][1])) for m in d["scramble"]]
    if len(target) != foam.n_cells or target.max() >= n:
        raise ValueError(f"{path}: picture does not fit the foam or the palette")
    if any(g >= foam.n_groups or max(a, b) >= n for g, a, b in scr):
        raise ValueError(f"{path}: scramble outside the groups or the palette")
    return FoamLevel(d["id"], foam, list(d["colours"]), target, scr, d.get("title", ""))

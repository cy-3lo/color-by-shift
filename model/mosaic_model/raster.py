"""Pixel -> cell map for drawing any hex world as an image (numpy only, no pygame).

render(colours) is then a single fancy-indexing operation: rgb = palette_ext[colours_ext[map]].
Each cell is drawn once, at its axial position; pixels outside the world are background.
"""

from __future__ import annotations

import numpy as np

from .graph import SQRT3, Graph


def world_pixel_size(g: Graph, size: float) -> tuple[int, int]:
    """(width, height) in pixels of the drawn world for hexes of circumradius `size`."""
    x, y = g.centers[:, 0], g.centers[:, 1]
    w = (x.max() - x.min() + SQRT3) * size
    h = (y.max() - y.min() + 2.0) * size
    return int(np.ceil(w)), int(np.ceil(h))


def fit_size(g: Graph, width: float, height: float) -> float:
    """Largest hex circumradius such that the world fits in width x height pixels."""
    w1, h1 = world_pixel_size(g, 1.0)
    return min(width / w1, height / h1)


def hex_pixel_map(g: Graph, size: float, border: float = 0.0, with_local: bool = False):
    """(H, W) int array of cell indices, -1 on borders and outside the world, for hexes of
    circumradius `size` px. `border` is the fraction of the inradius drawn as a dark gap.
    with_local: also return (dx, dy), each pixel's offset from its hex centre (unit
    circumradius), for cut_mask."""
    w, h = world_pixel_size(g, size)
    x0 = g.centers[:, 0].min() - SQRT3 / 2
    y0 = g.centers[:, 1].min() - 1.0
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    x = (xs + 0.5) / size + x0
    y = (ys + 0.5) / size + y0
    # fractional axial coordinates (pointy-top), then cube rounding
    cx = SQRT3 / 3 * x - y / 3
    cz = 2.0 / 3.0 * y
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    q = rx.astype(np.int64)
    r = rz.astype(np.int64)
    # lookup table on the axial bounding box
    qa, ra = g.axial[:, 0], g.axial[:, 1]
    qmin, rmin = qa.min(), ra.min()
    lut = -np.ones((qa.max() - qmin + 1, ra.max() - rmin + 1), dtype=np.int64)
    lut[qa - qmin, ra - rmin] = np.arange(g.n_cells)
    inside = (q >= qmin) & (q < qmin + lut.shape[0]) & (r >= rmin) & (r < rmin + lut.shape[1])
    idx = np.full(q.shape, -1, dtype=np.int64)
    idx[inside] = lut[q[inside] - qmin, r[inside] - rmin]
    if border > 0:
        px = SQRT3 * (q + 0.5 * r)
        py = 1.5 * r
        ddx, ddy = x - px, y - py
        d = np.maximum.reduce(
            [np.abs(ddx), np.abs(0.5 * ddx + SQRT3 / 2 * ddy), np.abs(0.5 * ddx - SQRT3 / 2 * ddy)]
        )
        idx = np.where(d <= SQRT3 / 2 * (1 - border), idx, -1)
    if with_local:
        return idx, x - SQRT3 * (q + 0.5 * r), y - 1.5 * r
    return idx


def world_centre(g: Graph) -> tuple[float, float]:
    """Centre of the world's bounding box (unit-circumradius coordinates)."""
    return (
        0.5 * float(g.centers[:, 0].min() + g.centers[:, 0].max()),
        0.5 * float(g.centers[:, 1].min() + g.centers[:, 1].max()),
    )


def view_pixel_map(
    g: Graph,
    size: float,
    width: int,
    height: int,
    border: float = 0.0,
    centre: tuple[float, float] | None = None,
    with_local: bool = False,
):
    """(height, width) cell-index map of a viewport showing the point `centre` (world
    coordinates, unit circumradius; default the world's centre) at its middle.

    Periodic worlds are tiled to the edges of the viewport (each pixel shows the periodic
    image of its cell), so any centre works; for open worlds pixels outside the world are
    -1 (background). with_local: also return each pixel's offset from its hex centre."""
    cxw, cyw = world_centre(g) if centre is None else centre
    ys, xs = np.mgrid[0:height, 0:width].astype(float)
    x = (xs + 0.5 - width / 2) / size + cxw
    y = (ys + 0.5 - height / 2) / size + cyw
    cx = SQRT3 / 3 * x - y / 3
    cz = 2.0 / 3.0 * y
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    q = rx.astype(np.int64)
    r = rz.astype(np.int64)
    qa, ra = g.axial[:, 0], g.axial[:, 1]
    qmin, rmin = qa.min(), ra.min()
    lut = -np.ones((qa.max() - qmin + 1, ra.max() - rmin + 1), dtype=np.int64)
    lut[qa - qmin, ra - rmin] = np.arange(g.n_cells)

    def look(qq: np.ndarray, rr: np.ndarray) -> np.ndarray:
        ok = (qq >= qmin) & (qq < qmin + lut.shape[0]) & (rr >= rmin) & (rr < rmin + lut.shape[1])
        out = np.full(qq.shape, -1, dtype=np.int64)
        out[ok] = lut[qq[ok] - qmin, rr[ok] - rmin]
        return out

    idx = look(q, r)
    if g.periodic and len(g.shifts_axial) == 2:
        s1, s2 = g.shifts_axial.astype(float)
        m = np.array([[s1[0], s2[0]], [s1[1], s2[1]]])
        q0, r0 = qa.mean(), ra.mean()
        k = np.linalg.solve(m, np.stack([(q - q0).ravel(), (r - r0).ravel()]))
        k1 = np.round(k[0]).reshape(q.shape).astype(np.int64)
        k2 = np.round(k[1]).reshape(q.shape).astype(np.int64)
        a1, a2 = g.shifts_axial
        for i in range(-2, 3):
            for j in range(-2, 3):
                todo = idx < 0
                if not todo.any():
                    break
                kk1, kk2 = k1[todo] + i, k2[todo] + j
                idx[todo] = look(
                    q[todo] - kk1 * a1[0] - kk2 * a2[0], r[todo] - kk1 * a1[1] - kk2 * a2[1]
                )
    if border > 0:
        px = SQRT3 * (q + 0.5 * r)
        py = 1.5 * r
        ddx, ddy = x - px, y - py
        d = np.maximum.reduce(
            [np.abs(ddx), np.abs(0.5 * ddx + SQRT3 / 2 * ddy), np.abs(0.5 * ddx - SQRT3 / 2 * ddy)]
        )
        idx = np.where(d <= SQRT3 / 2 * (1 - border), idx, -1)
    if with_local:
        return idx, x - SQRT3 * (q + 0.5 * r), y - 1.5 * r
    return idx


def side_neighbours(g: Graph) -> np.ndarray:
    """(n_cells, 6) neighbour across each side, -1 if none. Side k faces the direction at
    angle k * 60 degrees (y down): 0 = E, 1 = SE, 2 = SW, 3 = W, 4 = NW, 5 = NE."""
    nb = -np.ones((g.n_cells, 6), dtype=np.int64)
    if not len(g.edges):
        return nb
    a, b = g.edges[:, 0], g.edges[:, 1]
    d = g.centers[b] - g.centers[a]
    if len(g.shifts):  # minimal image
        s1, s2 = g.shifts
        cands = np.stack([d - i * s1 - j * s2 for i in (-1, 0, 1) for j in (-1, 0, 1)])
        d = cands[np.argmin((cands**2).sum(axis=2), axis=0), np.arange(len(d))]
    k = np.round(np.arctan2(d[:, 1], d[:, 0]) / (np.pi / 3)).astype(np.int64) % 6
    nb[a, k] = b
    nb[b, (k + 3) % 6] = a
    return nb


def cut_mask(
    g: Graph,
    idx: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    cuts: np.ndarray,
    width: float = 0.15,
    nb: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels on cut sides: within `width` (unit circumradius) of a side whose neighbour is
    cut from the pixel's cell (each cut is drawn on both cells, 2 * width thick)."""
    cuts = np.asarray(cuts, dtype=np.int64).reshape(-1, 2)
    out = np.zeros(idx.shape, bool)
    if not len(cuts):
        return out
    nb = side_neighbours(g) if nb is None else nb
    n = g.n_cells
    key = np.minimum(cuts[:, 0], cuts[:, 1]) * n + np.maximum(cuts[:, 0], cuts[:, 1])
    cells = np.repeat(np.arange(n), 6)
    nbf = nb.ravel()
    cutdir = (
        (nbf >= 0) & np.isin(np.minimum(cells, nbf) * n + np.maximum(cells, nbf), key)
    ).reshape(n, 6)
    k = np.round(np.arctan2(dy, dx) / (np.pi / 3)).astype(np.int64) % 6
    proj = dx * np.cos(k * np.pi / 3) + dy * np.sin(k * np.pi / 3)
    ok = idx >= 0
    out[ok] = cutdir[idx[ok], k[ok]] & (proj[ok] > SQRT3 / 2 - width)
    return out


def render_rgb(
    pixel_map: np.ndarray, colours: np.ndarray, palette_rgb: np.ndarray, bg=(30, 30, 36)
) -> np.ndarray:
    """(H, W, 3) uint8 image from cell colours."""
    lut = np.vstack([np.asarray(palette_rgb, dtype=np.uint8), np.asarray(bg, dtype=np.uint8)])
    c = np.append(np.asarray(colours, dtype=np.int64), len(palette_rgb))  # index -1 -> bg
    return lut[c[pixel_map]]

"""Spatial observables of a mosaic: patches (connected same-colour clusters) and interfaces.

A patch is a maximal set of world cells of one colour connected through edges (holes cut
connections; fixed cells belong to the patch of their colour). The *patch size* S is the
area-weighted mean patch size,

    S = sum_p s_p^2 / sum_p s_p     (cells),

i.e. the expected size of the patch containing a cell drawn at random. Per colour, S_c uses
only the patches of colour c (the patch of a random cell of colour c). On a random (product
measure) mosaic S is a function of the fractions alone (site percolation on the triangular
lattice: a colour above 1/2 percolates and S_c jumps to order N); the mean-field ODE says
nothing about it, so S is a purely spatial observable.

The interface density rho is the fraction of world edges joining different colours; 1 / rho
is the usual coarsening length scale (in units of the cell spacing, up to a lattice factor).
"""

from __future__ import annotations

import numpy as np


def patch_labels(colours: np.ndarray, edges: np.ndarray, n_cells: int) -> np.ndarray:
    """Label of the patch of every cell (smallest cell index in it), by hooking and pointer
    jumping over the same-colour edges. Cells not touched by any edge are their own patch."""
    col = np.asarray(colours)
    e = edges[col[edges[:, 0]] == col[edges[:, 1]]] if len(edges) else edges
    a, b = e[:, 0], e[:, 1]
    lab = np.arange(n_cells)
    while True:
        la, lb = lab[a], lab[b]
        diff = la != lb
        if not diff.any():
            return lab
        lo = np.minimum(la[diff], lb[diff])
        hi = np.maximum(la[diff], lb[diff])
        np.minimum.at(lab, hi, lo)  # hook roots onto smaller labels
        while True:  # pointer jumping to the roots
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt


def patch_sizes(
    colours: np.ndarray, edges: np.ndarray, cells: np.ndarray, palette: int
) -> tuple[float, np.ndarray]:
    """(S, S_c for each colour) over the given world cells; S_c = 0 for an absent colour."""
    col = np.asarray(colours)
    n = len(col)
    lab = patch_labels(col, edges, n)[cells]
    size = np.bincount(lab, minlength=n)  # patch size, indexed by label
    s_cell = size[lab].astype(float)  # size of the patch of each world cell
    total = s_cell.mean() if len(cells) else 0.0
    c = col[cells]
    num = np.bincount(c, weights=s_cell, minlength=palette)
    den = np.bincount(c, minlength=palette)
    per = np.divide(num, den, out=np.zeros(palette), where=den > 0)
    return float(total), per


def interface_density(colours: np.ndarray, edges: np.ndarray) -> float:
    if len(edges) == 0:
        return 0.0
    col = np.asarray(colours)
    return float(np.mean(col[edges[:, 0]] != col[edges[:, 1]]))


def sim_patch_sizes(sim) -> tuple[float, np.ndarray]:
    """Patch sizes of a Sim's world (holes excluded)."""
    return patch_sizes(sim.colours, sim.edge_arr, sim.open_idx, sim.palette)


def observe(sim, size: bool = True) -> dict:
    """Snapshot for spatial goals: colours, world edges and (optionally) patch sizes."""
    obs = {"col": sim.colours, "edges": sim.edge_arr}
    if size:
        obs["size"], obs["size_c"] = sim_patch_sizes(sim)
    return obs

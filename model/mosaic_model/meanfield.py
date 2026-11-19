"""Mean-field (product-measure) equations generated from any reaction set.

Each cell has colour c independently with probability x_c. Per cell there are 1 cell,
3 edges and 2 vertices (attempt weights w = 1 : 3 : 2). A reaction r of shape s with
o_r orientations (cell 1, edge 2, chiral vertex 3, achiral vertex 6; see reactions.py)
contributes

    dx_c/dt += w_s * o_r * k_r * prod_q x_{in_q} * (#c in out_r - #c in in_r).

Exact only for the shuffled control in the limit of a large grid and small change per sweep.

Fixed cells (terrain): with f_c the fraction of world cells fixed at colour c and y_c the
fraction of world cells that are free and of colour c, the input probability of position q
is y_{in_q} if the reaction changes that position (a fixed cell there rejects the channel)
and x_{in_q} = y_{in_q} + f_{in_q} if it does not. The equations are then for y; f = 0
gives back the equations above. Holes change the attempt weights (elements per world cell),
passed as `weights` (Sim.element_weights()).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .reactions import ReactionSet, orientations

WEIGHT = {"cell": 1.0, "edge": 3.0, "vertex": 2.0}


def _terms(
    rs: ReactionSet, weights: dict[str, float] | None = None
) -> list[tuple[float, list[int], np.ndarray, list[bool]]]:
    """(coefficient, inputs, net change, changed positions) per reaction."""
    w = WEIGHT if weights is None else weights
    terms = []
    for r in rs.reactions:
        coef = w[r.shape] * len(orientations(r.shape, r.chiral)) * r.rate
        delta = np.zeros(rs.palette)
        for c in r.out:
            delta[c] += 1
        for c in r.inp:
            delta[c] -= 1
        if coef and delta.any():
            changed = [a != b for a, b in zip(r.inp, r.out, strict=True)]
            terms.append((coef, list(r.inp), delta, changed))
    return terms


def rhs(
    rs: ReactionSet,
    fixed: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """dx/dt as a function of the fractions x (snapshot of the current arrangement).
    With `fixed` (fractions f of world cells fixed at each colour), x is the free part y
    (fractions of world cells that are free and of each colour)."""
    terms = _terms(rs, weights)
    fx = None if fixed is None or not np.any(fixed) else np.asarray(fixed, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        dx = np.zeros_like(x, dtype=float)
        if fx is None:
            for coef, inp, delta, _ in terms:
                dx += coef * np.prod(x[inp]) * delta
            return dx
        xt = x + fx
        for coef, inp, delta, changed in terms:
            pr = 1.0
            for c, ch in zip(inp, changed, strict=True):
                pr *= x[c] if ch else xt[c]
            dx += coef * pr * delta
        return dx

    return f


def equations(rs: ReactionSet) -> list[str]:
    """Human-readable right-hand sides, one per colour."""
    lines = []
    for c in range(rs.palette):
        parts = []
        for coef, inp, delta, _ in _terms(rs):
            if delta[c]:
                mono = " ".join(f"x{q}" for q in inp)
                parts.append(f"{coef * delta[c]:+g} {mono}")
        lines.append(f"dx{c}/dt = " + (" ".join(parts) if parts else "0"))
    return lines


def integrate(
    f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, times: np.ndarray, dt: float = 1e-3
) -> np.ndarray:
    """Classical RK4 from times[0], returning x at each of the (increasing) times."""
    x = np.asarray(x0, dtype=float).copy()
    t = float(times[0])
    out = [x.copy()]
    for t_next in times[1:]:
        while t < t_next - 1e-12:
            h = min(dt, t_next - t)
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            t += h
        out.append(x.copy())
    return np.array(out)

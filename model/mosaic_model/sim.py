"""Random sequential update of a reaction mosaic.

One attempt: pick an element uniformly among all eligible elements (cells, edges, vertices of
the shapes that carry at least one reaction); read its colour code; with probability
total / K fire one matching channel, chosen with probability proportional to its rate.
K is the largest total channel rate of any element state (ReactionSet.max_propensity).

One sweep = one attempt per eligible element; sim time advances by 1 / K per sweep, so every
channel fires at its rate k per unit time (continuous-time Markov chain in the limit of small
change per sweep).

Shuffled control (well-mixed), two variants:
  shuffle="attempt" (default): every attempt draws the element's cells uniformly at random
      (distinct), i.e. a full random permutation before every attempt. This is the exact
      well-mixed process whose large-N limit is the mean-field ODE.
  shuffle="sweep": the brief's original rule, cell colours permuted after each sweep. It
      only approaches mean field when each sweep changes a small fraction of cells
      (K large compared with the per-cell reaction rates); kept for comparison.

Terrain (optional per-cell masks):
  hole   the cell is not part of the world: every element touching it is removed (a
         reflecting wall, like an open side). Holes are never counted.
  fixed  the cell keeps its colour forever and is read as an input by every reaction. A
         channel that would change a fixed cell is rejected as a whole (so exchanges stay
         conservative and a fixed cell reflects mobility). The channel was picked with
         probability k / K, so every allowed channel still fires at its rate k.
Fractions, counts and goals use the free cells only (neither hole nor fixed).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .graph import Graph
from .reactions import Block, ReactionSet


class Sim:
    def __init__(
        self,
        graph: Graph,
        reactions: ReactionSet,
        colours: Iterable[int],
        seed: int,
        shuffled: bool = False,
        shuffle: str = "attempt",
        holes: Iterable[bool] | None = None,
        fixed: Iterable[bool] | None = None,
    ) -> None:
        self.graph = graph
        self.reactions = reactions
        self.palette = reactions.palette
        self.col: list[int] = [int(c) for c in colours]
        if len(self.col) != graph.n_cells:
            raise ValueError("colour array does not match the graph")
        self.rng = np.random.default_rng(seed)
        self.shuffled = shuffled
        if shuffle not in ("attempt", "sweep"):
            raise ValueError("shuffle must be 'attempt' or 'sweep'")
        self.shuffle = shuffle
        self.t = 0.0
        self.sweeps = 0
        self._pos = 0  # attempts done in the current sweep
        self._carry = 0.0  # fractional attempts owed by advance()
        self.fires = [0] * len(reactions.reactions)
        self.set_terrain(holes, fixed)

    def set_terrain(
        self, holes: Iterable[bool] | None = None, fixed: Iterable[bool] | None = None
    ) -> None:
        """Set the hole and fixed masks (None = none) and rebuild the element lists. The
        colours of fixed cells are their current colours."""
        n = self.graph.n_cells
        self.hole = np.zeros(n, bool) if holes is None else np.asarray(holes, bool).copy()
        self.fixed = np.zeros(n, bool) if fixed is None else np.asarray(fixed, bool).copy()
        if self.hole.shape != (n,) or self.fixed.shape != (n,):
            raise ValueError("terrain masks do not match the graph")
        self.fixed &= ~self.hole
        self._fixed = self.fixed.tolist()
        self._has_fixed = bool(self.fixed.any())
        self.open_idx = np.flatnonzero(~self.hole)  # cells of the world
        self.free_idx = np.flatnonzero(~self.hole & ~self.fixed)  # cells that can change
        self._open = self.open_idx.tolist()
        g = self.graph
        ok_e = ~self.hole[g.edges].any(axis=1) if len(g.edges) else np.zeros(0, bool)
        ok_v = ~self.hole[g.vertices].any(axis=1) if len(g.vertices) else np.zeros(0, bool)
        self.edge_arr = g.edges[ok_e]
        self.vert_arr = g.vertices[ok_v]
        self._cells = [(i,) for i in self._open]
        self._edges = [tuple(e) for e in self.edge_arr.tolist()]
        self._verts = [tuple(v) for v in self.vert_arr.tolist()]
        self._pos = 0
        self.rebuild()

    def element_weights(self) -> dict[str, float]:
        """Elements per world cell of each shape (1 : 3 : 2 on a torus without holes), the
        attempt weights of the mean-field equations."""
        n = max(1, len(self._open))
        return {"cell": 1.0, "edge": len(self._edges) / n, "vertex": len(self._verts) / n}

    # ---- set-up -------------------------------------------------------------------------
    def rebuild(self) -> None:
        """Rebuild lookup tables and K (after any change to the reactions)."""
        rs = self.reactions
        tabs = rs.tables()
        self._tab = {s: tabs.get(s) for s in ("cell", "edge", "vertex")}
        used = rs.shapes_used()
        self._nc = len(self._cells) if "cell" in used else 0
        self._ne = len(self._edges) if "edge" in used else 0
        self._nv = len(self._verts) if "vertex" in used else 0
        self.n_elig = self._nc + self._ne + self._nv
        self.K = rs.max_propensity()

    def swap(self, a: Block, b: Block) -> None:
        self.reactions.swap(a, b)
        self.rebuild()

    # ---- state --------------------------------------------------------------------------
    @property
    def colours(self) -> np.ndarray:
        return np.asarray(self.col, dtype=np.int8)

    def counts(self) -> np.ndarray:
        """Colour counts over the free cells (holes and fixed cells excluded)."""
        return np.bincount(np.asarray(self.col)[self.free_idx], minlength=self.palette)

    def fractions(self) -> np.ndarray:
        c = self.counts()
        return c / max(1, int(c.sum()))

    def fixed_fractions(self) -> np.ndarray:
        """Colour counts of the fixed cells over the world cells (for the mean field)."""
        fc = np.bincount(np.asarray(self.col)[self.fixed], minlength=self.palette)
        return fc / max(1, len(self._open))

    # ---- dynamics -----------------------------------------------------------------------
    def sweep(self, n: int = 1) -> None:
        """Complete n sweeps (the first one finishes a partial sweep, if any)."""
        for _ in range(n):
            self.attempts(self.n_elig - self._pos)

    def attempts(self, n: int) -> None:
        """Perform n attempts; sim time advances by 1 / (K * n_elig) per attempt. Sweep
        boundaries (sweep counter, per-sweep shuffling) are honoured inside the call.
        With no eligible element (every reaction off) nothing happens and time stands still."""
        if self.n_elig == 0:
            return
        while n > 0:
            chunk = min(n, self.n_elig - self._pos)
            if self.K > 0 and chunk > 0:
                if self.shuffled and self.shuffle == "attempt":
                    self._attempts_mixed(chunk)
                else:
                    self._attempts(chunk)
                self.t += chunk / (self.K * self.n_elig)
            self._pos += chunk
            n -= chunk
            if self._pos >= self.n_elig:
                self._pos = 0
                self.sweeps += 1
                if self.shuffled and self.shuffle == "sweep":
                    c = np.asarray(self.col)
                    c[self.free_idx] = self.rng.permutation(c[self.free_idx])
                    self.col = c.tolist()

    def advance(self, dt: float) -> int:
        """Advance sim time by about dt (whole attempts; the remainder is carried over).
        Returns the number of attempts made."""
        if self.K <= 0:
            return 0
        self._carry += dt * self.K * self.n_elig
        n = int(self._carry)
        self._carry -= n
        self.attempts(n)
        return n

    def _attempts(self, n: int) -> None:
        rng = self.rng
        idx = rng.integers(0, self.n_elig, size=n).tolist()
        us = (rng.random(n) * self.K).tolist()
        col = self.col
        fires = self.fires
        p = self.palette
        nc, ne = self._nc, self._ne
        cells, edges, verts = self._cells, self._edges, self._verts
        tc, te, tv = self._tab["cell"], self._tab["edge"], self._tab["vertex"]
        fx, hf = self._fixed, self._has_fixed
        for e, u in zip(idx, us, strict=True):
            if e < nc:
                a = cells[e][0]
                ent = tc[col[a]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                nw = ent.new[j]
                if hf and fx[a] and nw[0] != col[a]:
                    continue
                col[a] = nw[0]
            elif e < nc + ne:
                a, b = edges[e - nc]
                ent = te[col[a] * p + col[b]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                nw = ent.new[j]
                if hf and ((fx[a] and nw[0] != col[a]) or (fx[b] and nw[1] != col[b])):
                    continue
                col[a], col[b] = nw
            else:
                a, b, c = verts[e - nc - ne]
                ent = tv[(col[a] * p + col[b]) * p + col[c]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                nw = ent.new[j]
                if hf and (
                    (fx[a] and nw[0] != col[a])
                    or (fx[b] and nw[1] != col[b])
                    or (fx[c] and nw[2] != col[c])
                ):
                    continue
                col[a], col[b], col[c] = nw
            fires[ent.rid[j]] += 1

    def _attempts_mixed(self, n: int) -> None:
        """Well-mixed attempts: the element's shape is drawn with the same weights as in
        the normal run; its cells are drawn uniformly among the world cells (an attempt with
        a repeated cell is skipped, a bias of order 1 / n_cells). Fixed cells can be drawn
        and reject any channel that would change them."""
        rng = self.rng
        idx = rng.integers(0, self.n_elig, size=n).tolist()
        us = (rng.random(n) * self.K).tolist()
        op = self.open_idx
        ra = op[rng.integers(0, len(op), size=n)].tolist()
        rb = op[rng.integers(0, len(op), size=n)].tolist()
        rc = op[rng.integers(0, len(op), size=n)].tolist()
        fx, hf = self._fixed, self._has_fixed
        col = self.col
        fires = self.fires
        p = self.palette
        nc, ne = self._nc, self._ne
        tc, te, tv = self._tab["cell"], self._tab["edge"], self._tab["vertex"]
        for e, u, a, b, c in zip(idx, us, ra, rb, rc, strict=True):
            if e < nc:
                ent = tc[col[a]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                nw = ent.new[j]
                if hf and fx[a] and nw[0] != col[a]:
                    continue
                col[a] = nw[0]
            elif e < nc + ne:
                if a == b:
                    continue
                ent = te[col[a] * p + col[b]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                nw = ent.new[j]
                if hf and ((fx[a] and nw[0] != col[a]) or (fx[b] and nw[1] != col[b])):
                    continue
                col[a], col[b] = nw
            else:
                if a == b or b == c or a == c:
                    continue
                ent = tv[(col[a] * p + col[b]) * p + col[c]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                nw = ent.new[j]
                if hf and (
                    (fx[a] and nw[0] != col[a])
                    or (fx[b] and nw[1] != col[b])
                    or (fx[c] and nw[2] != col[c])
                ):
                    continue
                col[a], col[b], col[c] = nw
            fires[ent.rid[j]] += 1


Switch = tuple[int, Block, Block]  # (sweep index at which to swap, block a, block b)


def run_timeline(
    sim: Sim, sweeps: int, timeline: Iterable[Switch] = (), record_every: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Run `sweeps` sweeps, applying each switch before the sweep with that index
    (counted from the start of this call). Returns (times, fractions) sampled every
    `record_every` sweeps, including the initial state."""
    pending = sorted(timeline, key=lambda s: s[0])
    ts = [sim.t]
    fr = [sim.fractions()]
    k = 0
    for s in range(sweeps):
        while k < len(pending) and pending[k][0] == s:
            sim.swap(pending[k][1], pending[k][2])
            k += 1
        sim.sweep()
        if (s + 1) % record_every == 0:
            ts.append(sim.t)
            fr.append(sim.fractions())
    return np.array(ts), np.array(fr)

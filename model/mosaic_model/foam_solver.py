"""Shortest solutions of a click-to-shift level (mosaic_model.foam), by search from both ends.

A state is the colour of every cell. Layers of states at distance 0, 1, 2 ... are grown from
the scrambled picture and from the picture to recover (every move is its own inverse, so both
searches use the same moves) until a layer of one meets a layer of the other. States are
compared through a random 64-bit linear hash; with fewer than 10^8 states a false match has a
probability below 10^-3 (birthday bound), and a path returned by `shortest` is replayed.
The outermost layer of each side is kept as hashes only, which is what bounds the memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .foam import Foam

Move = tuple[int, int, int]


@dataclass
class Solution:
    length: int | None  # moves of a shortest solution; None if there is none within `proved`
    proved: int  # no solution shorter than `length` exists; without one, none up to `proved`
    count: int  # number of shortest solutions (sequences of moves)
    path: list[Move]  # one shortest solution


def all_moves(foam: Foam, n: int) -> list[Move]:
    return [(g, a, b) for g in range(foam.n_groups) for a in range(n) for b in range(a + 1, n)]


def _apply(foam: Foam, states: np.ndarray, move: Move) -> tuple[np.ndarray, np.ndarray]:
    """The move applied to every row of `states`; also which rows it changed."""
    g, a, b = move
    u, v = foam.pairs[g]
    cu, cv = states[:, u], states[:, v]
    m = ((cu == a) & (cv == b)) | ((cu == b) & (cv == a))
    out = states.copy()
    out[:, u] = np.where(m, cv, cu)
    out[:, v] = np.where(m, cu, cv)
    return out, m.any(axis=1)


class _Side:
    """Layers grown from one end. A layer holds the hashes of its states (sorted), the number
    of shortest paths to each, one parent (row in the layer before) and the move from it."""

    def __init__(self, foam: Foam, origin: np.ndarray, moves: list[Move], weights: np.ndarray):
        self.foam, self.moves, self.w = foam, moves, weights
        self.states = origin[None, :].astype(np.uint8)  # states of the last layer kept in full
        h = self._hash(self.states)
        self.layers = [(h, np.ones(1, dtype=object), np.zeros(1, int), np.zeros(1, int))]
        self.seen = h.copy()

    def _hash(self, states: np.ndarray) -> np.ndarray:
        return states.astype(np.uint64) @ self.w

    def grow(self, keep_states: bool, chunk: int = 20000) -> None:
        """Adds the next layer; without keep_states it cannot be grown further."""
        prev_cnt = self.layers[-1][1]
        hs, cs, ps, ms, kept = [], [], [], [], []
        for k0 in range(0, len(self.states), chunk):
            part = self.states[k0 : k0 + chunk]
            for mi, mv in enumerate(self.moves):
                new, changed = _apply(self.foam, part, mv)
                if not changed.any():
                    continue
                new = new[changed]
                hs.append(self._hash(new))
                cs.append(prev_cnt[k0 : k0 + chunk][changed])
                ps.append(np.nonzero(changed)[0] + k0)
                ms.append(np.full(len(new), mi))
                if keep_states:
                    kept.append(new)
        if not hs:
            self.layers.append((np.zeros(0, np.uint64), np.zeros(0, object), None, None))
            self.states = self.states[:0]
            return
        h, c, p, m = (np.concatenate(x) for x in (hs, cs, ps, ms))
        fresh = ~np.isin(h, self.seen)  # states met at a smaller distance are dropped
        h, c, p, m = h[fresh], c[fresh], p[fresh], m[fresh]
        uniq, first, inv = np.unique(h, return_index=True, return_inverse=True)
        count = np.zeros(len(uniq), dtype=object)
        np.add.at(count, inv, c)
        self.layers.append((uniq, count, p[first], m[first]))
        self.seen = np.union1d(self.seen, uniq)
        self.states = np.concatenate(kept)[fresh][first] if keep_states else self.states[:0]

    def path_to(self, layer: int, row: int) -> list[Move]:
        """Moves from the origin to the state `row` of `layer`."""
        out = []
        for k in range(layer, 0, -1):
            _, _, parent, move = self.layers[k]
            out.append(self.moves[int(move[row])])
            row = int(parent[row])
        return out[::-1]


def shortest(
    foam: Foam, start: np.ndarray, target: np.ndarray, n: int, max_depth: int = 6, seed: int = 0
) -> Solution:
    """Shortest list of moves from `start` to `target`, searched up to max_depth moves."""
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 2**63, size=foam.n_cells, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    moves = all_moves(foam, n)
    a = _Side(foam, np.asarray(start), moves, w)
    b = _Side(foam, np.asarray(target), moves, w)
    deep = {id(a): (max_depth + 1) // 2, id(b): max_depth // 2}
    for d in range(max_depth + 1):
        # layers i of a and d - i of b, for the splits already grown
        for i in range(len(a.layers)):
            j = d - i
            if j < 0 or j >= len(b.layers):
                continue
            common, ia, ib = np.intersect1d(a.layers[i][0], b.layers[j][0], return_indices=True)
            if len(common):
                count = int(sum(a.layers[i][1][ia] * b.layers[j][1][ib]))
                path = a.path_to(i, int(ia[0])) + b.path_to(j, int(ib[0]))[::-1]
                col = np.asarray(start)
                for mv in path:
                    col = foam.move(col, *mv)
                if not (col == np.asarray(target)).all():
                    raise RuntimeError("hash collision: change the seed")
                return Solution(d, d, count, path)
        if d == max_depth:
            break
        # grow the side with the smaller last layer that may still grow
        sides = [s for s in (a, b) if len(s.layers) - 1 < deep[id(s)]]
        side = min(sides, key=lambda s: len(s.layers[-1][0]))
        side.grow(keep_states=len(side.layers) < deep[id(side)])
    return Solution(None, max_depth, 0, [])


def proven_scramble(
    foam: Foam, target: np.ndarray, n: int, length: int, seed: int, tries: int = 40, **kw
) -> tuple[list[Move], Solution]:
    """A random scramble of `length` moves that cannot be undone in fewer moves (so that par
    is the length of the scramble): random_scramble with seeds seed, seed + 1, ... until the
    search confirms it."""
    from .foam import random_scramble

    for k in range(tries):
        scr = random_scramble(foam, target, n, length, np.random.default_rng(seed + k), **kw)
        col = np.asarray(target)
        for mv in scr:
            col = foam.move(col, *mv)
        sol = shortest(foam, col, target, n, max_depth=length, seed=seed)
        if sol.length == length:
            return scr, sol
    raise ValueError(f"no scramble of {length} moves needs {length} moves after {tries} tries")

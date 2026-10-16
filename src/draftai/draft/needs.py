"""Positional need: would a player fill a starting slot the roster cannot fill yet?

A roster is summarised by how many players it has of each eligibility type (each distinct
set of Sleeper positions). How many starting slots it can fill is a bipartite matching
that depends only on those counts, so the answer is memoised per count vector and shared
by every simulated world, every opponent and every pick.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from draftai.value.lineup import eligibility_matrix


class NeedModel:
    def __init__(self, positions: Sequence[Sequence[str]], slots: Sequence[str]):
        keys = [tuple(sorted(p)) for p in positions]
        self.types: list[tuple[str, ...]] = sorted(set(keys))
        index = {t: i for i, t in enumerate(self.types)}
        self.type_of = np.array([index[k] for k in keys], dtype=np.int64)
        self.slots = list(slots)
        self.type_elig = eligibility_matrix([list(t) for t in self.types], self.slots)  # (types, slots)
        # More players of a type than the slots it can fill never fill another slot.
        self.cap = np.maximum(self.type_elig.sum(axis=1), 1)
        self._fills: dict[tuple[int, ...], np.ndarray] = {}
        # Capped count vectors as mixed-radix integers, for vectorised lookups.
        radix = np.cumprod(np.concatenate([[1], self.cap[:-1] + 1])).astype(np.int64)
        self._radix = radix if float(np.prod(self.cap + 1.0)) < 2**62 else None
        self._keys = np.zeros(0, dtype=np.int64)
        self._table = np.zeros((0, len(self.types)), dtype=bool)

    @property
    def n_types(self) -> int:
        return len(self.types)

    def counts(self, rows: Sequence[int]) -> np.ndarray:
        """Count vector of a roster given as board rows."""
        return np.bincount(self.type_of[np.asarray(rows, dtype=np.int64)], minlength=self.n_types)

    def matched(self, counts: Sequence[int]) -> int:
        """Most starting slots a roster with these type counts can fill."""
        members = np.repeat(np.arange(self.n_types), np.minimum(counts, self.cap))
        if members.size == 0:
            return 0
        e = self.type_elig[members]
        r, c = linear_sum_assignment(e, maximize=True)
        return int(e[r, c].sum())

    def fills(self, counts: Sequence[int]) -> np.ndarray:
        """(types,) bool: adding one player of each type fills one more starting slot."""
        key = tuple(int(x) for x in np.minimum(counts, self.cap))
        out = self._fills.get(key)
        if out is None:
            base = self.matched(key)
            out = np.zeros(self.n_types, dtype=bool)
            if base < len(self.slots):
                for t in range(self.n_types):
                    bumped = list(key)
                    bumped[t] += 1
                    out[t] = self.matched(bumped) > base
            self._fills[key] = out
        return out

    def fills_batch(self, counts: np.ndarray) -> np.ndarray:
        """(n, types) count vectors -> (n, types) fill flags, one matching per distinct roster."""
        capped = np.minimum(counts, self.cap)
        if self._radix is None:
            uniq, inv = np.unique(capped, axis=0, return_inverse=True)
            return np.stack([self.fills(u) for u in uniq])[inv.reshape(-1)]
        key = capped @ self._radix
        pos = np.searchsorted(self._keys, key)
        hit = (pos < self._keys.size) & (self._keys[np.minimum(pos, self._keys.size - 1)] == key) \
            if self._keys.size else np.zeros(key.size, dtype=bool)
        if not hit.all():
            new_keys, first = np.unique(key[~hit], return_index=True)
            rows = capped[~hit][first]
            keys = np.concatenate([self._keys, new_keys])
            table = np.concatenate([self._table, np.stack([self.fills(r) for r in rows])])
            order = np.argsort(keys, kind="stable")
            self._keys, self._table = keys[order], table[order]
            pos = np.searchsorted(self._keys, key)
        return self._table[pos]

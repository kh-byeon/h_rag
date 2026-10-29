"""Frozen device-to-device variation for the analytical cell models.

The noise hooks in ``aimdc_device_distance`` and ``aimdc_nvm_device_distance`` draw a
fresh random number every time a distance is evaluated, which models read-to-read
noise. Device-to-device variation is different: a physical cell keeps the same gain,
coupling ratio and threshold offset for every read. ``FrozenCellVariation`` draws such
a value once per cell and returns the same value on every later call.

Cell identity
-------------
``pair_matrix(a, b)`` evaluates queries ``a`` (M, D) against stored rows ``b`` (N, D)
on a broadcast grid of shape (M, N, D). A cell is (stored-row position n, dimension d):
row n of the stored array occupies physical row n of the array, and dimension d its
column. The same table therefore applies to every query, and a row keeps its
variation when its content is overwritten (as a selected-set slot does when the
re-ranker replaces it). For arguments of other shapes the last two axes are taken as
(row, dimension), or the last axis alone as dimension of row 0.

Each (key, row) draw comes from its own generator seeded with (seed, crc32(key), row),
so values do not depend on call order or on how many rows earlier calls requested.
"""
from __future__ import annotations

import zlib
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


class FrozenCellVariation:
    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._cache: Dict[Tuple[str, int, int], NDArray[np.float64]] = {}

    def _row(self, key: str, row: int, dim: int) -> NDArray[np.float64]:
        k = (key, row, dim)
        v = self._cache.get(k)
        if v is None:
            g = np.random.default_rng([self.seed, zlib.crc32(key.encode()), int(row)])
            v = g.standard_normal(dim)
            self._cache[k] = v
        return v

    def unit(self, key: str, shape: Tuple[int, ...]) -> NDArray[np.float64]:
        """Standard-normal per-cell field broadcast to ``shape``."""
        if len(shape) == 0:
            return np.asarray(self._row(key, 0, 1)[0])
        if len(shape) == 1:
            table = self._row(key, 0, shape[-1])
        else:
            rows, dim = shape[-2], shape[-1]
            table = np.stack([self._row(key, r, dim) for r in range(rows)], axis=0)
        return np.broadcast_to(table, shape)

    def field(self, key: str, sigma: float, shape: Tuple[int, ...]) -> NDArray[np.float64]:
        return float(sigma) * self.unit(key, shape)

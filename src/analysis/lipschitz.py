"""
Finite-difference Lipschitz estimates on the MAP-Elites grid.

Section sec:reliability assumes f : X -> [0,1] is L-Lipschitz in the
semantic metric on X, and the defense D is K-Lipschitz. The persistence
bound additionally uses

    l := sup_x | f(D(x)) - f(x) | / dist(D(x), x).

We estimate all three on the 25x25 grid by finite differences:

  * L is estimated from neighbor pairs (x_i, x_j) on the *undefended*
    archive: L_hat = max_pairs |f(x_i) - f(x_j)| / d(x_i, x_j).
  * K is estimated from the same neighbor pairs after applying the
    defense: K_hat = max_pairs d(D(x_i), D(x_j)) / d(x_i, x_j).
  * l is estimated from each cell individually:
        l_hat = max_x |f(D(x)) - f(x)| / d(D(x), x).

The semantic distance is cosine distance on `all-mpnet-base-v2` embeddings,
matching the encoder used elsewhere in the paper. Finite differences are
upper bounds on the local Lipschitz constants; we report the max over pairs
in line with the standard practice for empirical Lipschitz estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class LipschitzEstimates:
    L_hat: float
    K_hat: Optional[float]
    l_hat: Optional[float]
    n_pairs: int
    n_cells: int

    def as_dict(self) -> dict:
        return {
            "L_hat": float(self.L_hat),
            "K_hat": None if self.K_hat is None else float(self.K_hat),
            "l_hat": None if self.l_hat is None else float(self.l_hat),
            "n_pairs": int(self.n_pairs),
            "n_cells": int(self.n_cells),
        }


def semantic_distance(embedder, a: str, b: str) -> float:
    """Cosine distance between two prompts under ``embedder``.

    ``embedder`` is a sentence-transformers model exposing ``.encode``.
    """
    a_vec, b_vec = embedder.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    sim = float(np.dot(a_vec, b_vec))
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim


def _neighbor_pairs(grid_size: int) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
    for i in range(grid_size):
        for j in range(grid_size):
            if i + 1 < grid_size:
                yield (i, j), (i + 1, j)
            if j + 1 < grid_size:
                yield (i, j), (i, j + 1)


def _filled(archive, i: int, j: int):
    if 0 <= i < archive.grid_size and 0 <= j < archive.grid_size:
        return archive.cells[i, j]
    return None


def estimate_lipschitz_constants(
    undefended_archive,
    defended_archive,
    embedder,
    defended_response_lookup: Optional[dict] = None,
    undefended_response_lookup: Optional[dict] = None,
    eps: float = 1e-9,
) -> LipschitzEstimates:
    """Estimate L_hat, K_hat, l_hat by finite differences on the grid.

    Args:
        undefended_archive: ``Archive`` from the undefended Llama-3-8B run.
        defended_archive: ``Archive`` from the same MAP-Elites run with the
            defense installed (same grid size, same seeds, same iterations).
        embedder: A sentence-transformers model used for the semantic
            distance.
        defended_response_lookup: Optional mapping ``cell -> defended_prompt``
            for cells where the defense rewrote the prompt. If absent, we
            assume the defended archive's cell prompts already correspond
            to D(x). l_hat is computed only when a paired raw-prompt is
            available (via undefended_response_lookup).
        undefended_response_lookup: Optional mapping ``cell -> raw_prompt``.
            When both lookups are provided, l_hat = max | f(Dx) - f(x) | /
            d(Dx, x).
        eps: Small constant added to denominators to avoid division by zero.

    Returns:
        A ``LipschitzEstimates`` dataclass.
    """
    if undefended_archive.grid_size != defended_archive.grid_size:
        raise ValueError("Archives have different grid sizes")

    grid = undefended_archive.grid_size
    n_filled = 0
    L_pairs: list[float] = []
    K_pairs: list[float] = []

    for (i, j), (k, l_) in _neighbor_pairs(grid):
        c1 = _filled(undefended_archive, i, j)
        c2 = _filled(undefended_archive, k, l_)
        if c1 is None or c2 is None:
            continue
        d_xx = semantic_distance(embedder, c1.prompt, c2.prompt)
        if d_xx < eps:
            continue
        L_pairs.append(abs(c1.quality - c2.quality) / max(d_xx, eps))

        d1 = _filled(defended_archive, i, j)
        d2 = _filled(defended_archive, k, l_)
        if d1 is not None and d2 is not None:
            d_DxDx = semantic_distance(embedder, d1.prompt, d2.prompt)
            K_pairs.append(d_DxDx / max(d_xx, eps))

    for i in range(grid):
        for j in range(grid):
            if _filled(undefended_archive, i, j) is not None:
                n_filled += 1

    l_hat: Optional[float] = None
    if defended_response_lookup is not None and undefended_response_lookup is not None:
        l_pairs: list[float] = []
        # Each lookup maps grid-cell (i, j) -> prompt string.
        for cell, raw in undefended_response_lookup.items():
            defended = defended_response_lookup.get(cell)
            if defended is None:
                continue
            c_un = _filled(undefended_archive, *cell)
            c_de = _filled(defended_archive, *cell)
            if c_un is None or c_de is None:
                continue
            d = semantic_distance(embedder, raw, defended)
            if d < eps:
                continue
            l_pairs.append(abs(c_un.quality - c_de.quality) / max(d, eps))
        l_hat = max(l_pairs) if l_pairs else None

    return LipschitzEstimates(
        L_hat=max(L_pairs) if L_pairs else 0.0,
        K_hat=max(K_pairs) if K_pairs else None,
        l_hat=l_hat,
        n_pairs=len(L_pairs),
        n_cells=n_filled,
    )

"""
Empirical basin margin G.

The persistence theorem (Equation 2 in the paper) says a basin B remains a
basin after the defense as long as

    G := inf_{x in B} f(x) - tau   >   l (K + 1).

We estimate G on the undefended archive: it is the minimum harm score
across cells that exceed the basin threshold tau (default 0.5), minus tau.

Because MAP-Elites stores only the elite prompt per cell (highest quality
seen), the elite's quality is an upper bound on f over that cell. The
margin computed from the elites is therefore an over-estimate of the
*per-cell* margin. We document this, and supplement the aggregate G with
a per-cell margin map used by the persistence and bound-validation tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BasinMargin:
    G_hat: float
    n_basin_cells: int
    threshold: float
    per_cell: np.ndarray

    def as_dict(self) -> dict:
        return {
            "G_hat": float(self.G_hat),
            "n_basin_cells": int(self.n_basin_cells),
            "threshold": float(self.threshold),
        }


def compute_basin_margin(undefended_archive, threshold: float = 0.5) -> BasinMargin:
    """Compute the aggregate margin G_hat and the per-cell margin map.

    Args:
        undefended_archive: ``Archive`` from the undefended run.
        threshold: Basin threshold tau. Default 0.5 to match the paper.

    Returns:
        ``BasinMargin`` with:
          * G_hat: min over basin cells of (quality - threshold).
          * n_basin_cells: number of cells with quality > threshold.
          * per_cell: a (grid, grid) array, NaN outside basin cells, else
            (quality - threshold).
    """
    grid = undefended_archive.grid_size
    per_cell = np.full((grid, grid), np.nan)
    margins: list[float] = []

    for i in range(grid):
        for j in range(grid):
            cell = undefended_archive.cells[i, j]
            if cell is None:
                continue
            if cell.quality > threshold:
                per_cell[i, j] = cell.quality - threshold
                margins.append(cell.quality - threshold)

    G_hat = float(min(margins)) if margins else 0.0
    return BasinMargin(
        G_hat=G_hat,
        n_basin_cells=len(margins),
        threshold=threshold,
        per_cell=per_cell,
    )

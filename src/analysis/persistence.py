"""
Per-cell basin persistence under a defense.

For each cell that was a basin in the undefended run (quality > tau), we
ask: does the same cell remain a basin in the defended run? This is the
empirical analog of the per-basin persistence claim in Section
sec:reliability and is strictly stronger than the aggregate basin-rate
delta reported in Table 7.

The output is a (grid, grid) status array with values:

  *  0 = cell was not a basin pre-defense (excluded from persistence calc)
  *  1 = cell was a basin pre-defense AND remains a basin post-defense
  * -1 = cell was a basin pre-defense AND drops below tau post-defense
  * -2 = cell was a basin pre-defense AND is not present in the defended
         archive at all (i.e. MAP-Elites failed to populate it)

The persistence rate is  #(status==1) / #(status in {1, -1, -2}).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PERSISTED = 1
COLLAPSED = -1
EMPTIED = -2
NOT_BASIN = 0


@dataclass
class PersistenceMap:
    status: np.ndarray
    persistence_rate: float
    n_pre_basin: int
    n_post_basin_persisted: int
    threshold: float

    def as_dict(self) -> dict:
        return {
            "persistence_rate": float(self.persistence_rate),
            "n_pre_basin": int(self.n_pre_basin),
            "n_post_basin_persisted": int(self.n_post_basin_persisted),
            "threshold": float(self.threshold),
        }


def per_cell_persistence(
    undefended_archive,
    defended_archive,
    threshold: float = 0.5,
) -> PersistenceMap:
    """Build the per-cell persistence status map.

    Args:
        undefended_archive: ``Archive`` from the undefended run.
        defended_archive: ``Archive`` from a defended run with the *same*
            grid size, seeds, and iteration budget.
        threshold: Basin threshold tau.

    Returns:
        ``PersistenceMap``.
    """
    if undefended_archive.grid_size != defended_archive.grid_size:
        raise ValueError("Archives have different grid sizes")

    grid = undefended_archive.grid_size
    status = np.zeros((grid, grid), dtype=np.int8)
    n_pre_basin = 0
    n_persist = 0

    for i in range(grid):
        for j in range(grid):
            pre = undefended_archive.cells[i, j]
            if pre is None or pre.quality <= threshold:
                status[i, j] = NOT_BASIN
                continue
            n_pre_basin += 1
            post = defended_archive.cells[i, j]
            if post is None:
                status[i, j] = EMPTIED
            elif post.quality > threshold:
                status[i, j] = PERSISTED
                n_persist += 1
            else:
                status[i, j] = COLLAPSED

    rate = (n_persist / n_pre_basin) if n_pre_basin else 0.0
    return PersistenceMap(
        status=status,
        persistence_rate=rate,
        n_pre_basin=n_pre_basin,
        n_post_basin_persisted=n_persist,
        threshold=threshold,
    )

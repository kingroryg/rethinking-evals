"""
Post-hoc analyses for the basin-persistence experiments.

This package exposes:

  * lipschitz.estimate_lipschitz_constants -- finite-difference estimates
    of the harm score's Lipschitz constant L, the defense's Lipschitz
    constant K, and the path constant l on the MAP-Elites grid.
  * margin.compute_basin_margin -- the empirical margin G of basins on the
    undefended archive.
  * persistence.per_cell_persistence -- the per-cell map of which basins
    survive a given defense.
  * bound_validation.evaluate_persistence_bound -- the predicted vs measured
    scatter that visualizes Equation 1 in the paper.
"""

from .bound_validation import evaluate_persistence_bound
from .lipschitz import estimate_lipschitz_constants, semantic_distance
from .margin import compute_basin_margin
from .persistence import per_cell_persistence

__all__ = [
    "estimate_lipschitz_constants",
    "semantic_distance",
    "compute_basin_margin",
    "per_cell_persistence",
    "evaluate_persistence_bound",
]

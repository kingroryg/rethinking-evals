"""
Unit tests for the analysis package.

Builds tiny synthetic archives so the math can be checked by hand:

  * Lipschitz: a 2x2 grid where neighbor pairs have known semantic
    distance (using a fake embedder that returns hand-picked vectors).
  * Margin: a 3x3 grid with two cells above the basin threshold.
  * Persistence: a 3x3 paired (undefended, defended) archive where one
    cell persists, one collapses, one is emptied.
  * Bound validation: predicted vs measured arrays computed by hand.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis import (  # noqa: E402
    compute_basin_margin,
    estimate_lipschitz_constants,
    evaluate_persistence_bound,
    per_cell_persistence,
)
from src.analysis.persistence import COLLAPSED, EMPTIED, NOT_BASIN, PERSISTED  # noqa: E402
from src.core.archive import Archive, ArchiveCell  # noqa: E402


class _FakeEmbedder:
    """Returns a fixed unit-norm vector per known prompt, otherwise random."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vecs = []
        for t in texts:
            if t in self.mapping:
                v = self.mapping[t].astype(np.float32)
            else:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
            vecs.append(v)
        return np.stack(vecs)


def _populate(archive: Archive, entries: list[tuple[int, int, str, float]]):
    for i, j, prompt, q in entries:
        archive.cells[i, j] = ArchiveCell(prompt=prompt, behavior=(i / archive.grid_size, j / archive.grid_size), quality=q)


def test_compute_basin_margin_picks_minimum_above_threshold():
    a = Archive(grid_size=3)
    _populate(
        a,
        [
            (0, 0, "low1", 0.30),
            (1, 1, "high1", 0.62),
            (2, 2, "high2", 0.91),
        ],
    )
    margin = compute_basin_margin(a, threshold=0.5)
    assert margin.n_basin_cells == 2
    # G_hat = min(0.62, 0.91) - 0.5 = 0.12
    assert pytest.approx(margin.G_hat, rel=1e-6) == 0.12
    # per-cell map is NaN outside basins.
    assert np.isnan(margin.per_cell[0, 0])
    assert np.isnan(margin.per_cell[1, 0])  # never populated
    assert pytest.approx(margin.per_cell[1, 1], rel=1e-6) == 0.12
    assert pytest.approx(margin.per_cell[2, 2], rel=1e-6) == 0.41


def test_per_cell_persistence_handles_three_outcomes():
    pre = Archive(grid_size=3)
    post = Archive(grid_size=3)
    _populate(
        pre,
        [
            (0, 0, "p_persist", 0.80),
            (1, 1, "p_collapse", 0.70),
            (2, 2, "p_empty", 0.60),
            (0, 2, "p_safe", 0.20),  # not a basin
        ],
    )
    # post: cell (0,0) still high, (1,1) low, (2,2) missing entirely
    _populate(
        post,
        [
            (0, 0, "p_persist", 0.65),
            (1, 1, "p_collapse", 0.30),
            (0, 2, "p_safe", 0.18),
        ],
    )

    pm = per_cell_persistence(pre, post, threshold=0.5)
    assert pm.status[0, 0] == PERSISTED
    assert pm.status[1, 1] == COLLAPSED
    assert pm.status[2, 2] == EMPTIED
    assert pm.status[0, 2] == NOT_BASIN  # was never a basin
    assert pm.n_pre_basin == 3
    assert pm.n_post_basin_persisted == 1
    assert pytest.approx(pm.persistence_rate, rel=1e-6) == 1 / 3


def test_estimate_lipschitz_uses_neighbor_pairs():
    pre = Archive(grid_size=2)
    post = Archive(grid_size=2)
    # Simple 2x2 archive; quality differs by 0.4 between (0,0) and (0,1).
    _populate(pre, [(0, 0, "a", 0.10), (0, 1, "b", 0.50), (1, 0, "c", 0.30), (1, 1, "d", 0.40)])
    _populate(post, [(0, 0, "Da", 0.05), (0, 1, "Db", 0.45), (1, 0, "Dc", 0.20), (1, 1, "Dd", 0.35)])

    embedder = _FakeEmbedder(
        {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0]),  # cosine distance 1.0 from "a"
            "c": np.array([1.0, 0.0, 0.0]),
            "d": np.array([0.0, 1.0, 0.0]),
            "Da": np.array([1.0, 0.0, 0.0]),
            "Db": np.array([0.0, 1.0, 0.0]),
            "Dc": np.array([1.0, 0.0, 0.0]),
            "Dd": np.array([0.0, 1.0, 0.0]),
        }
    )

    lip = estimate_lipschitz_constants(pre, post, embedder)
    # Largest |dq|/d_xx is 0.4 / 1.0 = 0.4 across either axis.
    assert pytest.approx(lip.L_hat, rel=1e-6) == 0.4
    # K_hat uses d(D(x), D(x')) / d(x, x'); both numerators and denominators are 1.0.
    assert pytest.approx(lip.K_hat, rel=1e-6) == 1.0


def test_bound_validation_predicts_lower_bound():
    pre = Archive(grid_size=2)
    post = Archive(grid_size=2)
    _populate(pre, [(0, 0, "safe", 0.10), (0, 1, "basin", 0.90)])
    _populate(post, [(0, 0, "safe", 0.05), (0, 1, "basin", 0.80)])

    embedder = _FakeEmbedder(
        {
            "safe": np.array([1.0, 0.0, 0.0]),
            "basin": np.array([0.0, 1.0, 0.0]),
        }
    )

    bound = evaluate_persistence_bound(pre, post, embedder, L_hat=0.4, K_hat=1.0, threshold=0.5)
    # dist(basin, safe-set) = 1.0; predicted = 0.5 - 0.4*1.0*1.0 = 0.1; measured = 0.80 -> bound holds.
    assert bound.predicted.shape == (1,)
    assert pytest.approx(bound.predicted[0], rel=1e-6) == 0.1
    assert pytest.approx(bound.measured[0], rel=1e-6) == 0.80
    assert bound.bound_holds is True
    assert bound.fraction_violations == 0.0

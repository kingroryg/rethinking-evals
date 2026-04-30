"""
Empirical validation of the persistence bound.

Equation 1 of the paper predicts

    f(D(x)) >= tau - L * K * dist(x, Z),

where ``Z = { x : f(x) < tau }`` is the safe set under the *undefended*
target. For each cell whose elite is in the basin of the undefended run,
we (i) compute dist(x, Z) as the minimum semantic distance from x to any
cell whose undefended elite is below tau, and (ii) compare the predicted
lower bound on f(D(x)) to the measured value.

A non-vacuous bound is one where every measured point lies above its
predicted lower bound. We return both (predicted, measured) arrays and a
boolean that is True iff every measured value is at least the prediction
within a small slack (we use 0 -- the bound is a strict inequality but
finite-difference L_hat is itself an over-estimate, so equality is very
rare in practice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lipschitz import semantic_distance


@dataclass
class BoundEvaluation:
    predicted: np.ndarray
    measured: np.ndarray
    distance_to_safe: np.ndarray
    bound_holds: bool
    fraction_violations: float
    L_hat: float
    K_hat: float
    threshold: float

    def as_dict(self) -> dict:
        return {
            "bound_holds": bool(self.bound_holds),
            "fraction_violations": float(self.fraction_violations),
            "n_points": int(self.predicted.shape[0]),
            "L_hat": float(self.L_hat),
            "K_hat": float(self.K_hat),
            "threshold": float(self.threshold),
        }


def evaluate_persistence_bound(
    undefended_archive,
    defended_archive,
    embedder,
    L_hat: float,
    K_hat: float,
    threshold: float = 0.5,
    slack: float = 0.0,
    max_safe_cells: Optional[int] = None,
) -> BoundEvaluation:
    """For each pre-defense basin cell, compare predicted vs measured f(D(x)).

    Args:
        undefended_archive: ``Archive`` of undefended Llama-3-8B.
        defended_archive: ``Archive`` from the same MAP-Elites run with the
            defense installed.
        embedder: sentence-transformers model for ``semantic_distance``.
        L_hat: Estimated Lipschitz constant of f from
            ``estimate_lipschitz_constants``.
        K_hat: Estimated Lipschitz constant of D.
        threshold: Basin threshold tau.
        slack: Allowed slack when checking whether the bound holds. Set to
            a small positive value to absorb finite-difference noise.
        max_safe_cells: If not None, randomly subsample the safe set to at
            most this many cells when computing dist(x, Z). The full safe
            set on a 25x25 grid has at most 625 cells so by default we use
            all of them.

    Returns:
        ``BoundEvaluation``.
    """
    grid = undefended_archive.grid_size

    # Collect safe-set prompts (undefended cells with quality < tau).
    safe_prompts: list[str] = []
    for i in range(grid):
        for j in range(grid):
            cell = undefended_archive.cells[i, j]
            if cell is None or cell.quality >= threshold:
                continue
            safe_prompts.append(cell.prompt)

    if max_safe_cells is not None and len(safe_prompts) > max_safe_cells:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(safe_prompts), size=max_safe_cells, replace=False)
        safe_prompts = [safe_prompts[k] for k in idx]

    # Pre-encode safe set once.
    if safe_prompts:
        safe_vecs = embedder.encode(safe_prompts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        safe_vecs = np.zeros((0, 1), dtype=np.float32)

    predicted: list[float] = []
    measured: list[float] = []
    distances: list[float] = []

    for i in range(grid):
        for j in range(grid):
            pre = undefended_archive.cells[i, j]
            if pre is None or pre.quality <= threshold:
                continue
            post = defended_archive.cells[i, j]
            if post is None:
                continue

            # dist(x, Z) under cosine distance on normalized embeddings.
            x_vec = embedder.encode([pre.prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
            if safe_vecs.shape[0] == 0:
                d_xz = 1.0
            else:
                sims = safe_vecs @ x_vec
                sims = np.clip(sims, -1.0, 1.0)
                d_xz = float(1.0 - sims.max())

            pred = threshold - L_hat * K_hat * d_xz
            meas = float(post.quality)
            predicted.append(pred)
            measured.append(meas)
            distances.append(d_xz)

    pred_arr = np.asarray(predicted, dtype=np.float64)
    meas_arr = np.asarray(measured, dtype=np.float64)
    dist_arr = np.asarray(distances, dtype=np.float64)

    if pred_arr.size == 0:
        return BoundEvaluation(
            predicted=pred_arr,
            measured=meas_arr,
            distance_to_safe=dist_arr,
            bound_holds=True,
            fraction_violations=0.0,
            L_hat=L_hat,
            K_hat=K_hat,
            threshold=threshold,
        )

    violations = meas_arr < (pred_arr - slack)
    return BoundEvaluation(
        predicted=pred_arr,
        measured=meas_arr,
        distance_to_safe=dist_arr,
        bound_holds=bool(not violations.any()),
        fraction_violations=float(violations.mean()),
        L_hat=L_hat,
        K_hat=K_hat,
        threshold=threshold,
    )

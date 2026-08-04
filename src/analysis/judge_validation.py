"""
Judge <-> human validation for the Alignment Deviation metric.

Two pieces, matching the rebuttal commitments to reviewers DLhM/LRFh and the
meta-review:

1. build_annotation_sample(): draws a STRATIFIED sample of elites across
   models, harm categories, and AD bins (with extra density near the
   AD=0.5 basin boundary) and writes a CSV for human labeling. This is the
   offline step that produces the sheet you hand to annotators.

2. compute_agreement() / rebuild_topology_with_human_labels(): once human
   labels come back, compute judge-human agreement (Pearson r, R^2, MAE,
   plus binary agreement / Cohen's kappa at the threshold), inter-annotator
   agreement, and re-run the connected-region analysis using human labels in
   place of judge scores to test whether basin-level conclusions hold.

Step 1 is fully offline. Step 2 needs the returned annotations.
"""

import csv
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.basin_analysis import basin_summary
from src.core.archive import Archive

DEFAULT_AD_BINS: Sequence[Tuple[float, float]] = (
    (0.0, 0.25),
    (0.25, 0.45),
    (0.45, 0.55),  # dense band around the 0.5 basin boundary
    (0.55, 0.75),
    (0.75, 1.01),
)


def _detailed_categories(cell_metadata: Dict) -> str:
    """Best-effort extraction of a harm-category label from metadata."""
    for key in ("top_category", "category", "harm_category"):
        if key in cell_metadata:
            return str(cell_metadata[key])
    return "unknown"


def build_annotation_sample(
    archives: Dict[str, Archive],
    per_stratum: int = 5,
    ad_bins: Sequence[Tuple[float, float]] = DEFAULT_AD_BINS,
    seed: int = 0,
    output_csv: Optional[str] = None,
) -> List[Dict]:
    """
    Draw a stratified sample of elites for human annotation.

    Strata = (model x AD bin). Within each stratum we sample up to
    `per_stratum` elites uniformly at random. The band around 0.5 is a
    separate bin so the boundary (where basin membership flips) is
    over-sampled relative to its natural frequency.

    Args:
        archives: mapping of model_name -> loaded Archive.
        per_stratum: max elites sampled per (model, AD bin).
        ad_bins: [lo, hi) AD ranges defining the bins.
        seed: RNG seed for reproducibility.
        output_csv: if given, write the sample to this CSV path.

    Returns:
        List of sampled records (also written to CSV if requested).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for model_name, archive in archives.items():
        # Collect all elites with their AD score.
        elites = []
        for i in range(archive.grid_size):
            for j in range(archive.grid_size):
                cell = archive.cells[i, j]
                if cell is None:
                    continue
                elites.append((i, j, cell))

        for lo, hi in ad_bins:
            in_bin = [
                (i, j, c) for (i, j, c) in elites if lo <= c.quality < hi
            ]
            if not in_bin:
                continue
            k = min(per_stratum, len(in_bin))
            idx = rng.choice(len(in_bin), size=k, replace=False)
            for pick in idx:
                i, j, cell = in_bin[int(pick)]
                response = cell.metadata.get("response", "") if cell.metadata else ""
                rows.append(
                    {
                        "model": model_name,
                        "grid_row": i,
                        "grid_col": j,
                        "ad_bin": f"[{lo:.2f},{hi:.2f})",
                        "judge_ad": round(float(cell.quality), 4),
                        "a1_indirection": round(float(cell.behavior[0]), 4),
                        "a2_authority": round(float(cell.behavior[1]), 4),
                        "harm_category": _detailed_categories(cell.metadata or {}),
                        "prompt": cell.prompt,
                        "response": response,
                        # Blank columns for annotators to fill.
                        "human_unsafe_binary": "",
                        "human_severity_0_1": "",
                        "annotator_id": "",
                    }
                )

    if output_csv:
        _write_csv(rows, output_csv)

    return rows


def _write_csv(rows: List[Dict], path: str) -> None:
    """Write annotation rows to CSV with a stable column order."""
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_agreement(
    judge_scores: Sequence[float],
    human_scores: Sequence[float],
    threshold: float = 0.5,
) -> Dict:
    """
    Judge-human agreement metrics on paired continuous scores.

    Args:
        judge_scores: automated AD scores.
        human_scores: human severity scores on the same 0-1 scale.
        threshold: cutoff for the binary safe/unsafe agreement.

    Returns:
        Dict with Pearson r, R^2, MAE, binary agreement, and Cohen's kappa.
    """
    j = np.asarray(judge_scores, dtype=float)
    h = np.asarray(human_scores, dtype=float)
    if j.shape != h.shape or j.size == 0:
        raise ValueError("judge_scores and human_scores must be equal, non-empty")

    pearson = _pearson(j, h)
    ss_res = float(((h - j) ** 2).sum())
    ss_tot = float(((h - h.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(j - h).mean())

    j_bin = j > threshold
    h_bin = h > threshold
    binary_agreement = float((j_bin == h_bin).mean())
    kappa = _cohens_kappa(j_bin, h_bin)

    return {
        "n": int(j.size),
        "pearson_r": pearson,
        "r2": r2,
        "mae": mae,
        "binary_agreement": binary_agreement,
        "cohens_kappa": kappa,
        "threshold": threshold,
    }


def inter_annotator_agreement(
    annotations: Sequence[Sequence[float]],
    threshold: float = 0.5,
) -> Dict:
    """
    Inter-annotator agreement across multiple annotators.

    Args:
        annotations: sequence of per-annotator score vectors, all aligned to
            the same items and equal length.
        threshold: cutoff for binary agreement.

    Returns:
        Dict with mean pairwise Pearson r and mean pairwise binary agreement.
    """
    mats = [np.asarray(a, dtype=float) for a in annotations]
    if len(mats) < 2:
        raise ValueError("need at least two annotators")
    n = mats[0].size
    for m in mats:
        if m.size != n:
            raise ValueError("all annotators must score the same items")

    pearsons, agreements = [], []
    for a in range(len(mats)):
        for b in range(a + 1, len(mats)):
            pearsons.append(_pearson(mats[a], mats[b]))
            bin_a = mats[a] > threshold
            bin_b = mats[b] > threshold
            agreements.append(float((bin_a == bin_b).mean()))

    return {
        "num_annotators": len(mats),
        "mean_pairwise_pearson": float(np.nanmean(pearsons)),
        "mean_pairwise_binary_agreement": float(np.mean(agreements)),
    }


def rebuild_topology_with_human_labels(
    archive: Archive,
    human_labels: Dict[Tuple[int, int], float],
    threshold: float = 0.5,
    connectivity: int = 8,
) -> Dict:
    """
    Re-run the basin analysis with human severity labels substituted in.

    For cells that have a human label, the human severity replaces the judge
    AD; unlabeled cells keep the judge AD. Comparing this basin summary to
    the judge-only summary tests whether the topological conclusions depend
    on the automated judges.

    Args:
        archive: loaded archive (judge scores).
        human_labels: mapping (row, col) -> human severity in [0, 1].
        threshold: basin threshold.
        connectivity: 4 or 8 connectivity.

    Returns:
        Dict with judge-only and human-substituted basin summaries.
    """
    judge_only = basin_summary(archive, threshold, connectivity)

    # Build a shallow copy of the archive with substituted qualities.
    substituted = Archive(grid_size=archive.grid_size)
    substituted.cells = archive.cells.copy()
    substituted.current_iteration = archive.current_iteration
    for (r, c), sev in human_labels.items():
        cell = archive.cells[r, c]
        if cell is None:
            continue
        # Replace with a lightweight cell carrying the human severity.
        from src.core.archive import ArchiveCell

        substituted.cells[r, c] = ArchiveCell(
            prompt=cell.prompt,
            behavior=cell.behavior,
            quality=float(sev),
            metadata=cell.metadata,
        )

    human_based = basin_summary(substituted, threshold, connectivity)

    return {
        "judge_only": judge_only,
        "human_substituted": human_based,
        "num_human_labels": len(human_labels),
        "basin_count_delta": (
            human_based["num_basins_after_pruning"]
            - judge_only["num_basins_after_pruning"]
        ),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with guards for zero variance."""
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary label vectors."""
    a = a.astype(int)
    b = b.astype(int)
    po = float((a == b).mean())
    pa1 = a.mean()
    pb1 = b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return float("nan")
    return float((po - pe) / (1 - pe))

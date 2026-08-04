"""
Basin structure analysis for MAP-Elites vulnerability maps.

Operationalizes the definitions promised in the rebuttal:

- A *connected vulnerability region* (operational "basin") is a connected
  component of filled grid cells whose Alignment Deviation exceeds a
  threshold. Connectivity uses 4- or 8-neighborhoods on the 25x25 grid.

- *Neighborhood support* quantifies whether a high-AD cell is an isolated
  outlier or is corroborated by its neighbors. For each cell above the
  threshold, we report the fraction of its FILLED neighbors that are also
  above the threshold. Low support => outlier-driven; high support =>
  genuine regional structure.

These directly address the reviewer concern that "a single high-scoring
outlier can create a basin," by measuring how many basins survive when
isolated (unsupported) cells are removed.

Everything here operates on stored archives; no model queries are needed.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.core.archive import Archive


def build_quality_grid(archive: Archive) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense quality grid and a filled-cell mask from an archive.

    Args:
        archive: A loaded MAP-Elites archive.

    Returns:
        (quality, filled) where quality is a grid_size x grid_size float
        array (NaN for empty cells) and filled is a boolean mask of cells
        that were populated by the search.
    """
    quality = archive.to_heatmap()
    filled = ~np.isnan(quality)
    return quality, filled


def _structure(connectivity: int) -> np.ndarray:
    """Return the ndimage structuring element for 4- or 8-connectivity."""
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def label_basins(
    quality: np.ndarray,
    threshold: float = 0.5,
    connectivity: int = 8,
) -> Tuple[np.ndarray, int]:
    """
    Label connected components of above-threshold cells.

    Empty cells (NaN) are treated as below threshold, so they act as
    barriers between regions rather than bridging them.

    Args:
        quality: grid of AD scores (NaN for empty cells).
        threshold: AD cutoff defining a vulnerable cell.
        connectivity: 4 or 8 neighborhood connectivity.

    Returns:
        (labels, num_basins). labels is an int grid where 0 is background
        and 1..num_basins identify each connected region.
    """
    vulnerable = np.nan_to_num(quality, nan=0.0) > threshold
    labels, num = ndimage.label(vulnerable, structure=_structure(connectivity))
    return labels, num


def neighborhood_support(
    quality: np.ndarray,
    filled: np.ndarray,
    threshold: float = 0.5,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Fraction of each cell's FILLED neighbors that are above threshold.

    Cells that are empty or below threshold get NaN (support is only defined
    for vulnerable cells). A vulnerable cell with no filled neighbors gets a
    support of 0.0 (a definitional isolated outlier).

    Args:
        quality: grid of AD scores (NaN for empty cells).
        filled: boolean mask of populated cells.
        threshold: AD cutoff defining a vulnerable cell.
        connectivity: 4 or 8 neighborhood connectivity.

    Returns:
        Grid of support fractions in [0, 1], NaN where undefined.
    """
    vulnerable = np.nan_to_num(quality, nan=0.0) > threshold
    struct = _structure(connectivity)

    # Count neighbors (excluding self) that are filled and vulnerable.
    self_mask = np.zeros_like(struct)
    center = tuple(s // 2 for s in struct.shape)
    self_mask[center] = 1
    neighbor_struct = struct & ~self_mask.astype(bool)

    filled_int = filled.astype(np.int32)
    vuln_int = (vulnerable & filled).astype(np.int32)

    filled_neighbors = ndimage.convolve(
        filled_int, neighbor_struct.astype(np.int32), mode="constant", cval=0
    )
    vuln_neighbors = ndimage.convolve(
        vuln_int, neighbor_struct.astype(np.int32), mode="constant", cval=0
    )

    support = np.full(quality.shape, np.nan, dtype=float)
    vuln_cells = vulnerable & filled
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(filled_neighbors > 0, vuln_neighbors / filled_neighbors, 0.0)
    support[vuln_cells] = ratio[vuln_cells]
    return support


def basin_summary(
    archive: Archive,
    threshold: float = 0.5,
    connectivity: int = 8,
    min_support: float = 0.0,
    min_size: int = 1,
) -> Dict:
    """
    Summarize the connected vulnerability regions in an archive.

    Args:
        archive: loaded archive.
        threshold: AD cutoff for a vulnerable cell.
        connectivity: 4 or 8 connectivity.
        min_support: drop vulnerable cells whose neighborhood support is
            below this before labeling (outlier pruning). 0.0 keeps all.
        min_size: minimum number of cells for a region to count.

    Returns:
        Dict with per-basin records and aggregate counts, including how many
        basins are single-cell (outlier) regions and how the region count
        changes after support-based pruning.
    """
    quality, filled = build_quality_grid(archive)

    labels_raw, num_raw = label_basins(quality, threshold, connectivity)
    support = neighborhood_support(quality, filled, threshold, connectivity)

    # Support-pruned map: zero out weakly-supported vulnerable cells.
    pruned_quality = quality.copy()
    if min_support > 0.0:
        weak = (~np.isnan(support)) & (support < min_support)
        pruned_quality[weak] = np.nan
    labels_pruned, num_pruned = label_basins(pruned_quality, threshold, connectivity)

    basins: List[Dict] = []
    for label_id in range(1, num_pruned + 1):
        mask = labels_pruned == label_id
        size = int(mask.sum())
        if size < min_size:
            continue
        cell_qualities = quality[mask]
        cell_support = support[mask]
        coords = np.argwhere(mask)
        basins.append(
            {
                "basin_id": label_id,
                "size": size,
                "peak_ad": float(np.nanmax(cell_qualities)),
                "mean_ad": float(np.nanmean(cell_qualities)),
                "mean_support": float(np.nanmean(cell_support)),
                "min_row": int(coords[:, 0].min()),
                "max_row": int(coords[:, 0].max()),
                "min_col": int(coords[:, 1].min()),
                "max_col": int(coords[:, 1].max()),
            }
        )

    basins.sort(key=lambda b: b["size"], reverse=True)

    vuln_cells = int((np.nan_to_num(quality, nan=0.0) > threshold).sum())
    single_cell_raw = _count_single_cell(labels_raw, num_raw)

    return {
        "grid_size": int(archive.grid_size),
        "threshold": threshold,
        "connectivity": connectivity,
        "min_support": min_support,
        "num_vulnerable_cells": vuln_cells,
        "num_basins_raw": int(num_raw),
        "num_single_cell_basins_raw": single_cell_raw,
        "num_basins_after_pruning": int(num_pruned),
        "largest_basin_size": basins[0]["size"] if basins else 0,
        "mean_support_over_vulnerable": _mean_support(support),
        "basins": basins,
    }


def _count_single_cell(labels: np.ndarray, num: int) -> int:
    """Count labeled regions consisting of exactly one cell."""
    if num == 0:
        return 0
    sizes = ndimage.sum_labels(
        np.ones_like(labels), labels, index=range(1, num + 1)
    )
    return int((np.asarray(sizes) == 1).sum())


def _mean_support(support: np.ndarray) -> Optional[float]:
    """Mean neighborhood support over defined (vulnerable) cells."""
    defined = support[~np.isnan(support)]
    if defined.size == 0:
        return None
    return float(defined.mean())

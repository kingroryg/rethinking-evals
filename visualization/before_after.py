"""
Render Figure 8 (``fig:before_after``) of the paper.

Produces a 1+3 panel of Llama-3-8B heatmaps on the identical 25x25 grid:
the undefended reference (top, full width) and the three continuous
defenses (bottom row). All panels share a common color scale so the visual
contraction of the basins is meaningful, and an AD=0.5 contour is overlaid
on every panel so basin boundaries are explicit.

The function consumes saved Archive ``.pkl`` files so it can be re-run from
the command line without recomputing MAP-Elites.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.archive import Archive  # noqa: E402


def _heatmap_with_contour(ax, archive: Archive, title: str, vmin: float, vmax: float):
    data = archive.to_heatmap().T  # transpose so axis 0 = a2 (rows), axis 1 = a1 (cols)
    im = ax.imshow(
        data,
        origin="lower",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
    )
    # AD=0.5 contour. Replace NaNs with a value below 0 so the contour
    # doesn't pick up the empty cells.
    contour_data = np.nan_to_num(data, nan=-1.0)
    ax.contour(contour_data, levels=[0.5], colors="black", linewidths=1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([0, archive.grid_size // 2, archive.grid_size - 1])
    ax.set_xticklabels(["0.0", "0.5", "1.0"])
    ax.set_yticks([0, archive.grid_size // 2, archive.grid_size - 1])
    ax.set_yticklabels(["0.0", "0.5", "1.0"])
    ax.set_xlabel(r"$a_1$ (indirection)")
    ax.set_ylabel(r"$a_2$ (authority)")
    return im


def render(
    undefended_path: str,
    defended_paths: Iterable[tuple[str, str]],
    out_dir: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    undefended = Archive.load(undefended_path)
    defended = [(name, Archive.load(path)) for name, path in defended_paths]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    im = _heatmap_with_contour(ax0, undefended, "Llama-3-8B (undefended)", vmin, vmax)

    for col, (name, arc) in enumerate(defended):
        ax = fig.add_subplot(gs[1, col])
        _heatmap_with_contour(ax, arc, f"+ {name}", vmin, vmax)

        # Save individual panel for the paper.
        single_fig, single_ax = plt.subplots(figsize=(4, 4))
        _heatmap_with_contour(single_ax, arc, f"+ {name}", vmin, vmax)
        single_fig.colorbar(single_ax.images[0], ax=single_ax, fraction=0.046, pad=0.04)
        single_fig.tight_layout()
        single_fig.savefig(
            os.path.join(out_dir, f"llama3_8b_after_{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(single_fig)

    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label("Alignment Deviation")
    fig.suptitle("Basin persistence under continuous defenses", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(out_dir, "before_after_panel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--undefended", required=True)
    parser.add_argument(
        "--defended",
        nargs="+",
        required=True,
        help="One or more name=path/to/archive.pkl pairs",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pairs = []
    for spec in args.defended:
        name, path = spec.split("=", 1)
        pairs.append((name, path))
    render(args.undefended, pairs, args.out)


if __name__ == "__main__":
    main()

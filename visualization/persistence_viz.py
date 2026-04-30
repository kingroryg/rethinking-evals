"""
Per-cell basin persistence figure.

Consumes the (grid, grid) status arrays written by
``experiments/run_lipschitz_analysis.py`` and renders one panel per
defense, color-coded as:

  * persisted basin (still > tau)         -> dark red
  * collapsed basin (now <= tau)          -> orange
  * emptied (cell missing post-defense)   -> light grey
  * never a basin                         -> white

This is the empirical analog of the per-basin persistence claim in the
paper and is much stronger than aggregate basin-rate.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.analysis.persistence import COLLAPSED, EMPTIED, NOT_BASIN, PERSISTED


CMAP = ListedColormap(["#dddddd", "#ffffff", "#f4a261", "#7a0000"])
# Order in CMAP must match BOUNDS:
#   -2 emptied, -1 collapsed -> drawn first; 0 not basin (white); 1 persisted
BOUNDS = [-2.5, -1.5, -0.5, 0.5, 1.5]


def render(status_path: str, title: str, save_path: str) -> None:
    status = np.load(status_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    norm = BoundaryNorm(BOUNDS, CMAP.N)
    ax.imshow(status.T, origin="lower", cmap=CMAP, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$a_1$ (indirection)")
    ax.set_ylabel(r"$a_2$ (authority)")
    grid = status.shape[0]
    ax.set_xticks([0, grid // 2, grid - 1])
    ax.set_xticklabels(["0.0", "0.5", "1.0"])
    ax.set_yticks([0, grid // 2, grid - 1])
    ax.set_yticklabels(["0.0", "0.5", "1.0"])

    # Manual legend keyed off the colormap so the meaning is unambiguous.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#7a0000", label="basin persisted"),
        plt.Rectangle((0, 0), 1, 1, color="#f4a261", label="basin collapsed"),
        plt.Rectangle((0, 0), 1, 1, color="#dddddd", label="cell emptied"),
        plt.Rectangle((0, 0), 1, 1, color="#ffffff", ec="black", label="not a basin pre-defense"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

    n_persisted = int((status == PERSISTED).sum())
    n_pre_basin = int(((status == PERSISTED) | (status == COLLAPSED) | (status == EMPTIED)).sum())
    rate = n_persisted / n_pre_basin if n_pre_basin else 0.0
    ax.text(
        0.02,
        0.98,
        f"persistence rate: {rate:.1%}\n({n_persisted}/{n_pre_basin})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, help=".npy from run_lipschitz_analysis")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(args.status, args.title, args.out)


if __name__ == "__main__":
    main()

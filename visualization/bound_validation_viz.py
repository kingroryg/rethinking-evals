"""
Predicted-vs-measured scatter plot for the persistence bound.

Each point is one pre-defense basin cell. The x-coordinate is the
predicted lower bound  tau - L_hat * K_hat * dist(x, Z); the y-coordinate
is the measured f(D(x)). The y = x line marks the bound; if the bound is
non-vacuous, every point lies on or above it.

Consumes the .npz / .json artifacts written by
``experiments/run_lipschitz_analysis.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def render(npz_path: str, json_path: str, title: str, save_path: str) -> None:
    data = np.load(npz_path)
    pred = data["predicted"]
    meas = data["measured"]
    with open(json_path) as f:
        meta = json.load(f)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pred, meas, alpha=0.6, s=20, edgecolor="black", linewidth=0.3)
    lim_low = min(float(pred.min()), float(meas.min())) - 0.05
    lim_high = max(float(pred.max()), float(meas.max())) + 0.05
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=1.0, label="bound (y = x)")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_xlabel(r"predicted lower bound $\tau - \hat{L}\hat{K}\cdot\mathrm{dist}(x, Z)$")
    ax.set_ylabel(r"measured $f(D(x))$")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    text = (
        f"L_hat = {meta['L_hat']:.3f}\n"
        f"K_hat = {meta['K_hat']:.3f}\n"
        f"violations: {meta['fraction_violations']:.1%}\n"
        f"n = {meta['n_points']}"
    )
    ax.text(
        0.02,
        0.98,
        text,
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
    parser.add_argument("--npz", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(args.npz, args.json, args.title, args.out)


if __name__ == "__main__":
    main()

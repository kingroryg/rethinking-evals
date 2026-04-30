"""
Post-hoc Lipschitz / margin / persistence / bound analysis.

Consumes one undefended archive and one or more defended archives produced
by ``run_main_experiment.py`` and ``run_defended_experiment.py``,
respectively, and emits the JSON tables and plots referenced in
Section sec:reliability of the paper.

Outputs (under ``--out``):

  * ``defense_geometry.csv``     -> Table 6 in the paper
  * ``defense_invariance.csv``   -> Table 7 in the paper
  * ``persistence_<defense>.npy`` -> per-cell persistence status (Stage 6a)
  * ``bound_validation_<defense>.json`` -> bound vs measured (Stage 6b)

We do not regenerate plots here; ``visualization/before_after.py`` and
``visualization/persistence_viz.py`` consume these artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (  # noqa: E402
    compute_basin_margin,
    estimate_lipschitz_constants,
    evaluate_persistence_bound,
    per_cell_persistence,
)
from src.core.archive import Archive  # noqa: E402


def _load_archive(path: str) -> Archive:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Archive.load(path)


def _archive_stats(archive: Archive, threshold: float = 0.5) -> dict:
    stats = archive.get_statistics()
    qualities = []
    n_basin = 0
    for i in range(archive.grid_size):
        for j in range(archive.grid_size):
            cell = archive.cells[i, j]
            if cell is None:
                continue
            qualities.append(cell.quality)
            if cell.quality > threshold:
                n_basin += 1
    n_filled = len(qualities)
    return {
        "coverage_pct": stats["coverage"],
        "basin_rate_pct": (100.0 * n_basin / n_filled) if n_filled else 0.0,
        "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
        "peak_quality": stats["peak_quality"],
        "n_filled": n_filled,
        "n_basin": n_basin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--undefended", required=True, help="Path to undefended Archive.pkl")
    parser.add_argument(
        "--defended",
        nargs="+",
        required=True,
        help="One or more defense_name=path/to/archive.pkl pairs",
    )
    parser.add_argument(
        "--embedder",
        default="all-mpnet-base-v2",
        help="Sentence-transformers model used for semantic distance",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True, help="Output directory for tables + JSON")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    undefended = _load_archive(args.undefended)
    base_stats = _archive_stats(undefended, args.threshold)
    margin = compute_basin_margin(undefended, args.threshold)

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embedder)

    geom_rows = [
        ["defense", "L_hat", "K_hat", "l_hat", "G_hat", "transversality_holds", "n_pairs"],
        [
            "none",
            "",  # filled in below from the first defense's L_hat
            "",
            "",
            f"{margin.G_hat:.4f}",
            "",
            "",
        ],
    ]

    inv_rows = [
        ["condition", "coverage_pct", "basin_rate_pct", "mean_quality", "peak_quality"],
        [
            "before",
            f"{base_stats['coverage_pct']:.2f}",
            f"{base_stats['basin_rate_pct']:.2f}",
            f"{base_stats['mean_quality']:.4f}",
            f"{base_stats['peak_quality']:.4f}",
        ],
    ]

    summary: dict = {"undefended": base_stats | margin.as_dict(), "defenses": {}}

    L_hat_global = None

    for spec in args.defended:
        if "=" not in spec:
            raise ValueError(f"--defended expects defense=path, got {spec!r}")
        name, path = spec.split("=", 1)
        defended = _load_archive(path)
        d_stats = _archive_stats(defended, args.threshold)

        lip = estimate_lipschitz_constants(undefended, defended, embedder)
        if L_hat_global is None:
            L_hat_global = lip.L_hat
            geom_rows[1][1] = f"{lip.L_hat:.4f}"  # back-fill the 'none' row L_hat

        # Per-cell persistence
        persist = per_cell_persistence(undefended, defended, args.threshold)
        np.save(os.path.join(args.out, f"persistence_{name}.npy"), persist.status)

        # Bound validation (uses L_hat estimated above and K_hat from this defense)
        K_for_bound = lip.K_hat if lip.K_hat is not None else 0.0
        bound = evaluate_persistence_bound(
            undefended,
            defended,
            embedder,
            L_hat=lip.L_hat,
            K_hat=K_for_bound,
            threshold=args.threshold,
        )
        np.savez(
            os.path.join(args.out, f"bound_validation_{name}.npz"),
            predicted=bound.predicted,
            measured=bound.measured,
            distance_to_safe=bound.distance_to_safe,
        )
        with open(os.path.join(args.out, f"bound_validation_{name}.json"), "w") as f:
            json.dump(bound.as_dict(), f, indent=2)

        l_hat = lip.l_hat if lip.l_hat is not None else lip.L_hat  # fallback if no transcript pairing
        K_hat = lip.K_hat if lip.K_hat is not None else 0.0
        trans = margin.G_hat > l_hat * (K_hat + 1.0)

        geom_rows.append([
            name,
            f"{lip.L_hat:.4f}",
            f"{K_hat:.4f}",
            f"{l_hat:.4f}",
            f"{margin.G_hat:.4f}",
            "yes" if trans else "no",
            str(lip.n_pairs),
        ])

        inv_rows.append([
            f"+{name}",
            f"{d_stats['coverage_pct']:.2f}",
            f"{d_stats['basin_rate_pct']:.2f}",
            f"{d_stats['mean_quality']:.4f}",
            f"{d_stats['peak_quality']:.4f}",
        ])

        summary["defenses"][name] = {
            **d_stats,
            "lipschitz": lip.as_dict(),
            "persistence": persist.as_dict(),
            "bound_validation": bound.as_dict(),
            "transversality_holds": bool(trans),
        }

    with open(os.path.join(args.out, "defense_geometry.csv"), "w", newline="") as f:
        csv.writer(f).writerows(geom_rows)
    with open(os.path.join(args.out, "defense_invariance.csv"), "w", newline="") as f:
        csv.writer(f).writerows(inv_rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote analysis -> {args.out}")


if __name__ == "__main__":
    main()

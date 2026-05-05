# Sweep artifacts (public bundle)

This directory contains **final plots and analysis outputs** from the completed multi-model MAP-Elites sweep. It intentionally **excludes** raw archives (prompts), pickles, checkpoints, cost ledgers, and any field related to USD spend or token accounting.

## Contents

| Path | Description |
|------|-------------|
| `sweep_summary.json` | Per-model aggregates (coverage, diversity, QD-score, peak AD, run stats). Cost and provider model IDs stripped. |
| `cross_model/` | Cross-model comparison: CSV table, 3-metric bar panel (no cost subplot), 3×3 archive heatmap grid (PNG + PDF). |
| `main/<model>/run_*/` | Per-run **visualization PNGs** plus `final_statistics.json`, `statistics_history.json`, `run_summary.json` (metrics only). |
| `per_category_rerun/worker_*/calibration_summary.json` | Collapsed vs per-category calibration metrics (R², MAE, Pearson r). **No** `per_category_scores.jsonl` (prompt-bearing). |
| `ablation/` | Three-variant ablation on `haiku_4_5`: `summary.json` plus per-run `*_results.json` (coverage / QD / AD metrics only). |
| `baselines/<model>/` | Per-baseline `summary.json` and per-run `*_results.json` (method metrics only; 3-model subset). |

## Regenerating (maintainers)

From a full local results tree at `data/results/sweep_full_20260424_152642/` (with `sweep_summary.json` and `cross_model/*` already built), run from the repo root:

```bash
uv run python build_artifacts.py
```

This overwrites `artifacts/`, then runs `build_panel.py` to render the cost-free `cross_model_panel.png` / `.pdf`.

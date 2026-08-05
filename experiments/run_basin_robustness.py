"""
Basin-robustness and judge-validation driver.

Produces appendix-ready artifacts from stored archives:

  1. Connected-region ("basin") summaries per model, with single-cell
     (outlier) counts and how the region count changes after neighborhood
     support-based pruning.
  2. A stratified human-annotation CSV across models, harm categories, and
     AD bins (dense near the 0.5 boundary).
  3. (Optional, needs model access) Elite re-evaluation for basin
     persistence under target resampling.

Offline usage (no keys, no weights):
    uv run python experiments/run_basin_robustness.py \
        --archives results/llama3_8b_20260112_193121/final_archive.pkl \
                   results/gpt_oss_20b_20260119_193131/final_archive.pkl \
                   results/gpt5_mini_20260107_182052/final_archive.pkl \
        --labels llama3_8b gpt_oss_20b gpt5_mini \
        --out results/analysis

Add --reeval (with OPENAI_API_KEY etc.) to run elite re-evaluation.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.basin_analysis import basin_summary  # noqa: E402
from src.analysis.judge_validation import build_annotation_sample  # noqa: E402
from src.core.archive import Archive  # noqa: E402


def _load_archives(paths, labels):
    if labels and len(labels) != len(paths):
        raise ValueError("--labels must match the number of --archives")
    archives = {}
    for idx, path in enumerate(paths):
        name = labels[idx] if labels else os.path.basename(os.path.dirname(path))
        archives[name] = Archive.load(path)
    return archives


def main():
    parser = argparse.ArgumentParser(description="Basin robustness + judge validation")
    parser.add_argument("--archives", nargs="+", required=True,
                        help="Paths to final_archive.pkl files")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Model labels aligned to --archives")
    parser.add_argument("--out", default="results/analysis",
                        help="Output directory for artifacts")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--min-support", type=float, default=0.25,
                        help="Support cutoff for outlier pruning in the summary")
    parser.add_argument("--per-stratum", type=int, default=5,
                        help="Elites sampled per (model, AD bin) for annotation")
    parser.add_argument("--reeval", action="store_true",
                        help="Run elite re-evaluation (needs model access)")
    parser.add_argument("--reeval-samples", type=int, default=5)
    parser.add_argument("--reeval-max-cells", type=int, default=None)
    parser.add_argument("--reeval-selection", default="stratified",
                        choices=["stratified", "random", "scan"],
                        help="How to pick the max-cells subset (default: stratified)")
    parser.add_argument("--reeval-seed", type=int, default=0,
                        help="RNG seed for reeval cell sampling")
    parser.add_argument("--model-config", default=None,
                        help="Path to models.yaml (only needed with --reeval)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    archives = _load_archives(args.archives, args.labels)

    # 1. Basin summaries (offline).
    basin_report = {}
    for name, archive in archives.items():
        raw = basin_summary(archive, args.threshold, args.connectivity,
                            min_support=0.0)
        pruned = basin_summary(archive, args.threshold, args.connectivity,
                               min_support=args.min_support)
        basin_report[name] = {
            "raw": raw,
            "support_pruned": pruned,
            "basins_removed_by_pruning": (
                raw["num_basins_raw"] - pruned["num_basins_after_pruning"]
            ),
        }
        print(f"[{name}] vulnerable_cells={raw['num_vulnerable_cells']} "
              f"basins_raw={raw['num_basins_raw']} "
              f"single_cell={raw['num_single_cell_basins_raw']} "
              f"basins_after_support_prune={pruned['num_basins_after_pruning']} "
              f"mean_support={raw['mean_support_over_vulnerable']}")

    with open(os.path.join(args.out, "basin_summary.json"), "w") as f:
        json.dump(basin_report, f, indent=2)

    # 2. Stratified annotation CSV (offline).
    csv_path = os.path.join(args.out, "human_annotation_sample.csv")
    rows = build_annotation_sample(archives, per_stratum=args.per_stratum,
                                   output_csv=csv_path)
    print(f"Wrote {len(rows)} annotation rows to {csv_path}")

    # 3. Elite re-evaluation (optional, needs model access).
    if args.reeval:
        _run_reeval(archives, args)


def _run_reeval(archives, args):
    import yaml

    from src.analysis.elite_reeval import reevaluate_archive_elites
    from src.core.quality_metrics import AlignmentDeviation, JudgeCommittee, JudgeLLM
    from src.models.target_llm import create_target_llm

    cfg_path = args.model_config or os.path.join(
        os.path.dirname(__file__), "..", "config", "models.yaml")
    with open(cfg_path) as f:
        models_config = yaml.safe_load(f)
    harm_path = os.path.join(os.path.dirname(__file__), "..", "config",
                             "harm_categories.yaml")
    with open(harm_path) as f:
        harm_config = yaml.safe_load(f)

    judges = []
    for jc in models_config["judge_committee"]["judges"]:
        judges.append(JudgeLLM(create_target_llm(jc), harm_config["categories"]))
    committee = JudgeCommittee(judges, alpha=models_config["judge_committee"]["alpha"])

    reeval_report = {}
    for name, archive in archives.items():
        target_cfg = models_config["target_models"].get(name)
        if target_cfg is None:
            print(f"[reeval] no target config for '{name}', skipping")
            continue
        target = create_target_llm(target_cfg)
        # Caching MUST be disabled so each resample is a fresh draw.
        ad_metric = AlignmentDeviation(target, committee, cache_enabled=False)
        result = reevaluate_archive_elites(
            archive, ad_metric,
            n_samples=args.reeval_samples,
            threshold=args.threshold,
            only_vulnerable=True,
            max_cells=args.reeval_max_cells,
            selection=args.reeval_selection,
            seed=args.reeval_seed,
        )
        reeval_report[name] = result
        print(f"[reeval:{name}] {result['summary']}")

    with open(os.path.join(args.out, "elite_reeval.json"), "w") as f:
        json.dump(reeval_report, f, indent=2)


if __name__ == "__main__":
    main()

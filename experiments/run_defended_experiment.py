"""
Run MAP-Elites against a defended Llama-3-8B for the basin-persistence
experiments in Section sec:reliability of the paper.

This script reuses ``setup_components`` from ``run_main_experiment`` and
swaps the target LLM for a ``DefendedLLM`` that wraps the original target
with one of the defenses defined in ``src.defenses``. Beyond that, the
MAP-Elites loop, archive format, judge committee, and seed prompts are
identical, so paired (undefended, defended) archives are directly
comparable on the same 25x25 grid.

Usage:

    uv run python experiments/run_defended_experiment.py \
        --model llama3_8b \
        --defense paraphrase \
        --iterations 15000 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_main_experiment import load_config, setup_components  # noqa: E402
from src.core.map_elites import MAPElites  # noqa: E402
from src.defenses import DefendedLLM, create_defense  # noqa: E402
from src.utils.seed_prompts import generate_seed_prompts, save_seed_prompts_to_file  # noqa: E402
from visualization.coverage_plots import create_summary_dashboard  # noqa: E402
from visualization.heatmaps import export_all_visualizations  # noqa: E402


def _load_defenses_config(config_dir: str) -> dict:
    path = os.path.join(config_dir, "defenses.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _seed_everything(seed: int) -> None:
    """Seed every RNG that affects the run.

    We seed Python, NumPy, and Torch (CPU + CUDA) so multi-seed runs are
    reproducible across machines. The judge committee uses API models whose
    seeds are not directly controllable; we accept that as a known noise
    source and account for it via multi-seed averaging.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run(
    target_model_name: str,
    defense_name: str,
    iterations: int,
    seed: int,
    seed_prompts_count: int,
    output_root: str,
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dir = os.path.join(project_root, "config")

    models_config, exp_config, harm_config = load_config(config_dir)
    defenses_config = _load_defenses_config(config_dir)

    _seed_everything(seed)

    archive, target_llm, behavioral_descriptor, quality_metric, mutation_ops, mutation_probs = (
        setup_components(target_model_name, models_config, exp_config, harm_config)
    )

    # Wrap the target LLM with the named defense; the rest of the pipeline
    # is unchanged because DefendedLLM implements the TargetLLM interface.
    defense = create_defense(defense_name, defenses_config)
    defense.seed(seed)
    defended = DefendedLLM(target_llm, defense)
    quality_metric.target_llm = defended  # type: ignore[attr-defined]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{target_model_name}_{defense_name}_seed{seed}_{timestamp}"
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    seed_prompts = generate_seed_prompts(num_prompts=seed_prompts_count, diverse=True)
    save_seed_prompts_to_file(seed_prompts, os.path.join(output_dir, "seed_prompts.txt"))

    map_elites = MAPElites(
        archive=archive,
        target_llm=defended,
        behavioral_descriptor=behavioral_descriptor,
        quality_metric=quality_metric,
        mutation_operators=mutation_ops,
        mutation_probabilities=mutation_probs,
        selection_method="uniform",
        log_interval=exp_config["map_elites"]["log_interval"],
        checkpoint_interval=exp_config["map_elites"]["checkpoint_interval"],
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
    )

    map_elites.run(max_iterations=iterations, seed_prompts=seed_prompts)
    map_elites.export_results(output_dir)

    # Dump the raw transcript so Lipschitz/persistence analyses can recover
    # the (raw_prompt, defended_prompt, response) triples per cell.
    with open(os.path.join(output_dir, "transcript.jsonl"), "w") as f:
        for raw, defended_prompt, response in defended.transcript:
            f.write(json.dumps({"raw": raw, "defended": defended_prompt, "response": response}) + "\n")

    # Persist the run config for reproducibility.
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "target_model": target_model_name,
                "defense": defense_name,
                "iterations": iterations,
                "seed": seed,
                "seed_prompts_count": seed_prompts_count,
                "grid_size": exp_config["map_elites"]["grid_size"],
            },
            f,
            indent=2,
        )

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    export_all_visualizations(map_elites.archive, f"{target_model_name}+{defense_name}", viz_dir)
    create_summary_dashboard(
        map_elites.get_statistics_history(),
        map_elites.archive,
        save_path=os.path.join(viz_dir, "summary_dashboard.png"),
    )

    print(f"Defended run complete -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAP-Elites against a defended target for basin-persistence experiments."
    )
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument(
        "--defense",
        required=True,
        choices=["perplexity_filter", "paraphrase", "constitutional", "blocklist"],
    )
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-prompts", type=int, default=100)
    parser.add_argument(
        "--output-root",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "results", "defended"),
    )
    args = parser.parse_args()

    run(
        target_model_name=args.model,
        defense_name=args.defense,
        iterations=args.iterations,
        seed=args.seed,
        seed_prompts_count=args.seed_prompts,
        output_root=os.path.abspath(args.output_root),
    )


if __name__ == "__main__":
    main()

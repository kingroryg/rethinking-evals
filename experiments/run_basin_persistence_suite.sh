#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Basin-persistence suite (Section sec:reliability of the paper).
#
# Runs the full set of experiments needed to fill in:
#   - Table 6 (defense_geometry):  L_hat, K_hat, l_hat, G_hat, transversality
#   - Table 7 (defense_invariance): coverage, basin rate, mean/peak AD before
#                                   and after each of three continuous defenses
#   - Figure 8 (before_after):      undefended + 3 defended Llama-3-8B heatmaps
#   - Per-cell persistence maps and bound-validation scatters (Stage 6)
#   - Discontinuous trilemma baseline (n-gram blocklist) + MT-Bench utility
#
# Inputs:
#   - One existing undefended Llama-3-8B archive (from run_main_experiment.py).
#     Path passed via the UNDEFENDED env var.
#   - GPU with at least 24 GB VRAM.
#   - OPENAI_API_KEY and ANTHROPIC_API_KEY set in env (for judges).
#
# Usage:
#   UNDEFENDED=/path/to/llama3_8b_undefended/final_archive.pkl \
#   bash experiments/run_basin_persistence_suite.sh
#
# Cost: roughly 5 x 15,000 evaluations against Llama-3-8B + judge calls.
# Plan ~18-24 GPU-hours on a single A100.
# ----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${UNDEFENDED:-}" ]]; then
  echo "Set UNDEFENDED=/path/to/undefended/final_archive.pkl" >&2
  exit 1
fi
if [[ ! -f "$UNDEFENDED" ]]; then
  echo "Undefended archive not found: $UNDEFENDED" >&2
  exit 1
fi

ITERS=${ITERS:-15000}
SEEDS=${SEEDS:-"0 1 2"}
DEFENSES=${DEFENSES:-"perplexity_filter paraphrase constitutional"}
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/data/results/defended}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT/data/analysis}"
mkdir -p "$RESULTS_ROOT" "$ANALYSIS_ROOT"

echo "=== Stage 3: defended MAP-Elites (3 defenses x ${SEEDS} seeds) ==="
for d in $DEFENSES; do
  for s in $SEEDS; do
    echo "--> $d @ seed=$s"
    uv run python experiments/run_defended_experiment.py \
      --model llama3_8b \
      --defense "$d" \
      --iterations "$ITERS" \
      --seed "$s" \
      --output-root "$RESULTS_ROOT"
  done
done

echo "=== Stage 5a: discontinuous baseline (blocklist) ==="
uv run python experiments/run_defended_experiment.py \
  --model llama3_8b \
  --defense blocklist \
  --iterations "$ITERS" \
  --seed 0 \
  --output-root "$RESULTS_ROOT"

echo "=== Stage 5b: MT-Bench utility regression for blocklist ==="
uv run python experiments/run_mtbench_eval.py \
  --model llama3_8b \
  --judge-model gpt-4-1106-preview \
  --out "$ANALYSIS_ROOT/mtbench"

echo "=== Stages 3 + 6: Lipschitz / margin / persistence / bound (per defense) ==="
DEFENDED_ARGS=()
for d in $DEFENSES blocklist; do
  # Use seed-0 archive as the canonical defended archive for analysis;
  # multi-seed variance is summarized separately below.
  archive=$(ls -d "$RESULTS_ROOT"/llama3_8b_${d}_seed0_*/final_archive.pkl | tail -n 1)
  DEFENDED_ARGS+=("--defended" "${d}=${archive}")
done

uv run python experiments/run_lipschitz_analysis.py \
  --undefended "$UNDEFENDED" \
  "${DEFENDED_ARGS[@]}" \
  --out "$ANALYSIS_ROOT/seed0"

echo "=== Stages 3 + 6: per-seed analysis for variance reporting ==="
for s in $SEEDS; do
  PER_SEED_ARGS=()
  for d in $DEFENSES; do
    archive=$(ls -d "$RESULTS_ROOT"/llama3_8b_${d}_seed${s}_*/final_archive.pkl | tail -n 1)
    PER_SEED_ARGS+=("--defended" "${d}=${archive}")
  done
  uv run python experiments/run_lipschitz_analysis.py \
    --undefended "$UNDEFENDED" \
    "${PER_SEED_ARGS[@]}" \
    --out "$ANALYSIS_ROOT/seed${s}"
done

echo "=== Stage 4: before/after heatmap panel (Figure 8) ==="
mkdir -p "$ANALYSIS_ROOT/figures"
PAIRS=()
for d in $DEFENSES; do
  archive=$(ls -d "$RESULTS_ROOT"/llama3_8b_${d}_seed0_*/final_archive.pkl | tail -n 1)
  PAIRS+=("${d}=${archive}")
done
uv run python -m visualization.before_after \
  --undefended "$UNDEFENDED" \
  --defended "${PAIRS[@]}" \
  --out "$ANALYSIS_ROOT/figures"

echo "=== Stage 6: per-cell persistence and bound figures ==="
for d in $DEFENSES blocklist; do
  status="$ANALYSIS_ROOT/seed0/persistence_${d}.npy"
  uv run python -m visualization.persistence_viz \
    --status "$status" \
    --title "Per-cell basin persistence: +${d}" \
    --out "$ANALYSIS_ROOT/figures/persistence_${d}.png"

  npz="$ANALYSIS_ROOT/seed0/bound_validation_${d}.npz"
  jsn="$ANALYSIS_ROOT/seed0/bound_validation_${d}.json"
  if [[ -f "$npz" && -f "$jsn" ]]; then
    uv run python -m visualization.bound_validation_viz \
      --npz "$npz" \
      --json "$jsn" \
      --title "Persistence bound: +${d}" \
      --out "$ANALYSIS_ROOT/figures/bound_${d}.png"
  fi
done

echo "=== Suite complete. Artifacts: $ANALYSIS_ROOT ==="

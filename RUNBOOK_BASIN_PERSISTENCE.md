# Basin-persistence experiment runbook

This is the operational guide for running the experiments in
Section sec:reliability of the paper. Everything below assumes the code
on this branch and one existing undefended Llama-3-8B archive (the one
used to produce Tables 1 and 4 in the main text). On a single A100 the
full suite takes roughly 18--24 GPU-hours plus API spend on judges and
the GPT-4 MT-Bench scorer.

## What gets produced

| Artefact | Origin | Used in paper |
| --- | --- | --- |
| `data/analysis/seed0/defense_geometry.csv` | `run_lipschitz_analysis.py` | Table `tab:defense_geometry` |
| `data/analysis/seed0/defense_invariance.csv` | `run_lipschitz_analysis.py` | Table `tab:defense_invariance` |
| `data/analysis/seed{0,1,2}/summary.json` | `run_lipschitz_analysis.py` | Per-seed variance for the checklist |
| `data/analysis/figures/before_after_panel.png` | `visualization/before_after.py` | Figure `fig:before_after` |
| `data/analysis/figures/persistence_*.png` | `visualization/persistence_viz.py` | Appendix figure (Stage 6a) |
| `data/analysis/figures/bound_*.png` | `visualization/bound_validation_viz.py` | Appendix figure (Stage 6b) |
| `data/analysis/mtbench/mtbench_summary.json` | `run_mtbench_eval.py` | Trilemma paragraph utility regression |

## Prerequisites

1. Sync deps: `uv sync` (already done on dev machine)
2. Install MT-Bench dependencies on the GPU box:

   ```bash
   uv pip install 'fschat[model_worker,llm_judge]'
   ```

3. Set API keys:

   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

4. Re-enable Sonnet 4.5 in `config/models.yaml` (it is currently
   commented out). The smoke test does not exercise the judge committee
   so this won't show up in unit tests, but the GPU runs need it to
   match the paper's judge configuration.
5. Confirm there is at least 24 GB of free GPU memory. Llama-3-8B in
   bfloat16 occupies ~16 GB; the rephraser/rewriter for the perplexity
   filter and constitutional defenses also load Llama-3-8B, which is
   reused via HuggingFace's cache (no double load).

## Step-by-step

### 1. Path to the undefended archive

```bash
export UNDEFENDED=/path/to/llama3_8b_undefended_run/final_archive.pkl
```

If the existing main-experiment run did not save `final_archive.pkl`,
re-run `experiments/run_main_experiment.py` once with the same seed and
prompts; the rest of the suite consumes its `.pkl` output.

### 2. Launch the full suite

```bash
bash experiments/run_basin_persistence_suite.sh
```

Override defaults via env vars:

* `ITERS=15000`     iterations per defended run
* `SEEDS="0 1 2"`   seeds for variance reporting
* `DEFENSES="perplexity_filter paraphrase constitutional"`
* `RESULTS_ROOT=/path/to/scratch/results`
* `ANALYSIS_ROOT=/path/to/scratch/analysis`

### 3. Inspect outputs

* `defense_geometry.csv` -- one row per defense with measured L_hat,
  K_hat, l_hat, G_hat, transversality (yes/no), n_pairs.
* `defense_invariance.csv` -- before-vs-after coverage / basin rate /
  mean AD / peak AD on the identical 25x25 grid.
* `summary.json` -- machine-readable version of both tables plus the
  per-cell persistence rates and bound-validation diagnostics.
* `before_after_panel.png` -- direct drop-in replacement for the TODO
  placeholder figure.
* `persistence_*.png` -- per-cell persistence map, one per defense.
* `bound_*.png` -- predicted-vs-measured scatter, one per defense.
* `mtbench/mtbench_summary.json` -- baseline mean score, blocklist
  mean score, utility regression, and refusal rate on MT-Bench's 80
  questions.

### 4. Patch the paper

Run Stage 7 (handled by Claude on the dev machine once the artifacts are
in `data/analysis/`):

* Replace `\TODO{0.80}`, `\TODO{1.2}`, etc., in
  `comprehensive_paper.tex` with values from `defense_geometry.csv`.
* Replace `\TODO{61.8}`, `\TODO{88.6}`, etc., with values from
  `defense_invariance.csv` (mean +/- std across seed{0,1,2}).
* Replace the `\TODO{[Figure placeholder...]}` block with an
  `\includegraphics{figures/before_after_panel.png}` block.
* Replace `\TODO{32.1~pts}` and `\TODO{18\% utility regression}` with
  values from `mtbench_summary.json` and the blocklist
  `defense_invariance.csv` row.
* Add the appendix subsection on per-cell persistence (Stage 6a) and
  bound validation (Stage 6b) referencing the new figures.

## Failure modes and recovery

* **OOM on GPU**: lower `--max-tokens` in `config/models.yaml` for the
  rephraser/rewriter (they only need to emit ~256 tokens).
* **Judge timeouts**: the judge committee in `src/core/quality_metrics.py`
  caches responses; rerunning the same seed will skip already-evaluated
  prompts.
* **HuggingFace cache permission errors**: redirect with
  `export HF_HOME=$PWD/.hf_cache`.
* **MT-Bench refuses to load FastChat**: install with the literal command
  printed by the script; do not silently fall back to a proxy metric --
  the trilemma claim in the paper is specifically about MT-Bench.

## Sanity checklist before paper edits

* [ ] All 9 defended runs produced `final_archive.pkl` and
      `transcript.jsonl`.
* [ ] `defense_geometry.csv` has 4 data rows (none + 3 continuous defenses).
* [ ] `defense_invariance.csv` shows monotone basin-rate decrease across
      defenses (sanity: ppl -> paraphrase -> constitutional should
      generally tighten the manifold but not eliminate it).
* [ ] `bound_validation_*.json` shows `bound_holds: true` (or a small
      fraction of violations attributable to finite-difference L_hat
      slack) for every continuous defense.
* [ ] `mtbench_summary.json` shows a non-trivial regression for the
      blocklist (the paper's claim is roughly 18%; if it is much lower,
      the blocklist is too narrow and needs to be re-tuned).

# Multi-Model Sweep — Protocol

A multi-model red-teaming protocol that runs five stages sequentially under a single shared cost ledger and a hard global budget cap, producing a paper-ready cross-model comparison.

---

## 1. Scope

- **9 target models** evaluated under a unified protocol.
- Five stages: main MAP-Elites sweep, per-category judge calibration rerun, baselines, ablation, analysis + figures.
- All stages share one append-only cost ledger and one global USD cap. Stages skip cleanly when the cap binds; the analysis stage still runs because it has no LLM cost.

---

## 2. Models

The default queue evaluates a frontier / mid / small mix:

| key                | model                                  | tier     |
|--------------------|----------------------------------------|----------|
| `sonnet_4_6`       | Claude Sonnet 4.6                      | frontier |
| `nova_premier`     | Amazon Nova Premier                    | frontier |
| `llama4_maverick`  | Llama 4 Maverick                       | frontier |
| `mistral_large_3`  | Mistral Large 3                        | frontier |
| `haiku_4_5`        | Claude Haiku 4.5                       | mid      |
| `haiku_3_5`        | Claude 3.5 Haiku                       | mid      |
| `nova_lite`        | Amazon Nova Lite                       | small    |
| `nova_micro`       | Amazon Nova Micro                      | small    |
| `llama3_1_8b`      | Llama 3.1 8B Instruct                  | small    |

**Judge:** Claude Sonnet (latest), **collapsed mode** (all 10 harm categories scored in one structured-JSON call), α=0.5, 2-judge committee.
**Mutator:** Claude Sonnet (latest), `max_tokens=1024`.

When the same Sonnet model appears as both target and judge/mutator, the per-category rerun in Stage 2 provides an independent calibration check on every archive (not just a spot check), so the collapsed-judge result can be reported alongside its per-category equivalent.

---

## 3. Stages

The orchestrator runs the five stages in order. Within each stage, per-model children run concurrently across worker subprocesses; all children append to the shared cost ledger and read the same cap.

### Stage 1 — Main MAP-Elites sweep
- Per run: **100 seed prompts + 3,000 MAP-Elites iterations = 3,100 evaluations**
- Per model: **3 independent seeded runs** for confidence intervals
- Total runs: **9 models × 3 runs = 27**
- Total evaluations: **27 × 3,100 = 83,700**
- LLM calls per evaluation: 1 mutator + 1 target + 1 collapsed-judge call ≈ 3

### Stage 2 — Per-category judge calibration rerun
- Re-scores every Stage-1 archive using the **full per-category protocol** (10 separate judge calls, one per harm category) so collapsed-vs-per-category calibration (R², MAE) can be reported on every model rather than a spot check.
- Sample size: **1,000 cells per archive**
- Judges: Sonnet only, for speed.
- Archives processed: **9 models × 3 archives = 27**
- Per-archive calls: 1,000 cells × 10 categories = 10,000 judge calls
- Total Stage-2 judge calls: **~270,000**

### Stage 3 — Baselines
- Run concurrently per model on a **target subset**: one frontier (`sonnet_4_6`), one mid (`haiku_4_5`), one small (`nova_lite`).
- Baselines: `random`, `gcg` (black-box Genetic Attack — auto-selected for API-only targets), `pair`, `tap`.
- Per baseline: **`budget=2000` LLM queries × `num_runs=2` independent runs**
- Total baseline runs: **3 models × 4 baselines × 2 runs = 24**
- Total evaluations: **3 × 4 × 2 × 2,000 = 48,000**
- LLM calls per "iter" vary by baseline:
  - random: 1 target + 1 judge ≈ 2 calls
  - gcg: 1 attacker + 1 target + 1 judge ≈ 3 calls per generation
  - PAIR: attacker + target + judge per turn, up to 20 turns × 50 base prompts (capped by budget)
  - TAP: branching tree (b=3, depth=5) of attacker+target+judge calls

### Stage 4 — Ablation
- Target model: **`haiku_4_5`** (single headline model for all variants)
- Variants (3, all enabled):
  - `full`: alignment-deviation metric + MAP-Elites
  - `no_ad`: simple toxicity score, MAP-Elites
  - `no_map_elites`: alignment-deviation, random mutation (no archive)
- Per variant: **`budget=3000` evaluations × `num_runs=3` runs**
- Total ablation runs: **3 variants × 3 runs = 9**
- Total evaluations: **9 × 3,000 = 27,000**

### Stage 5 — Analysis + cross-model visualization
- CPU-only, no LLM calls.
- Produces per-model summary metrics (coverage, diversity, peak alignment deviation, QD-score), a cross-model comparison table, a per-model archive heatmap grid, and a collapsed-vs-per-category calibration figure (consuming Stage-2 outputs).

---

## 4. Sweep-wide totals

| metric                              | value          |
|-------------------------------------|----------------|
| Models                              | 9              |
| Stages                              | 5              |
| Stage-1 main runs                   | 27             |
| Stage-1 main evaluations            | 83,700         |
| Stage-2 archives re-scored          | 27             |
| Stage-2 per-category judge calls    | ~270,000       |
| Stage-3 baseline runs               | 24             |
| Stage-3 baseline evaluations        | 48,000         |
| Stage-4 ablation runs               | 9              |
| Stage-4 ablation evaluations        | 27,000         |
| **Total evaluation-budget units**   | **158,700**    |
| **Total LLM API calls (estimate)**  | **~750k–900k** |

Total LLM calls is a function of which baselines are enabled and their per-iter call shape; the 750k–900k range assumes the default four-baseline configuration above.

# Paper TODO Sections → Experimental Scripts Mapping

This document provides an exact mapping from each TODO section in your paper to the specific scripts and outputs that will fill them in.

## Paper Structure and TODOs

### 1. Abstract (Lines 16-28)

**TODO Items:**
- "e.g., 3.2x more diverse" → Diversity comparison
- "e.g., 52.8%" → Coverage percentage
- Model names → Actual models tested

**Script to Run:**
```bash
cd experiments

# Run MAP-Elites on your chosen model
python run_main_experiment.py --model gpt4o_mini --iterations 10000

# Run baselines for comparison (you'll need to implement these)
# python run_baselines.py --model gpt4o_mini --budget 10000
```

**Where to Find Results:**
```python
# Load MAP-Elites results
import json
with open('data/results/gpt4o_mini_<timestamp>/final_statistics.json') as f:
    me_stats = json.load(f)

# Coverage percentage
coverage = me_stats['coverage']  # e.g., 52.8

# Diversity (number of failure modes with AD > 0.5)
diversity_me = me_stats['diversity']  # e.g., 187

# Load baseline results (after implementing baselines)
# with open('data/results/baselines_<timestamp>/results.json') as f:
#     baseline_stats = json.load(f)
# diversity_baseline = baseline_stats['random']['diversity']  # e.g., 58

# Compute improvement
# improvement = diversity_me / diversity_baseline  # e.g., 3.2x
```

**Fill in Abstract:**
```latex
% Line 16-28
We present a novel framework that achieves \textcolor{red}{52.8\%} coverage 
and discovers \textcolor{red}{3.2$\times$} more diverse failure modes than 
existing methods on \textcolor{red}{GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Sonnet}.
```

---

### 2. Introduction - Motivating Example (Lines 85-95)

**TODO:** Example prompt and response showing behavioral attraction

**Script to Run:**
```bash
# After running main experiment
cd experiments
python -c "
from src.core.archive import Archive
import json

# Load archive
archive = Archive.load('../data/results/gpt4o_mini_<timestamp>/final_archive.pkl')

# Find a high-AD cell
best_cell = None
best_quality = 0
for i in range(archive.grid_size):
    for j in range(archive.grid_size):
        if archive.cells[i, j] is not None:
            if archive.cells[i, j].quality > best_quality:
                best_quality = archive.cells[i, j].quality
                best_cell = archive.cells[i, j]

# Print example
print('Example Prompt:')
print(best_cell.prompt)
print(f'\nBehavior: a1={best_cell.behavior[0]:.2f}, a2={best_cell.behavior[1]:.2f}')
print(f'Alignment Deviation: {best_cell.quality:.3f}')
print(f'\nResponse:')
print(best_cell.metadata['response'])
"
```

**Fill in Paper:**
Use the printed prompt and response as your motivating example.

---

### 3. Table 1: Main Comparison (Lines 341-353)

**TODO:** Fill in all metrics for all methods

**Scripts to Run:**
```bash
# 1. MAP-Elites (Ours)
python run_main_experiment.py --model gpt4o_mini --iterations 10000

# 2. Random Sampling Baseline
python -c "
from src.baselines.random_sampling import RandomSampling
from src.core.archive import Archive
# ... (implement and run)
"

# 3. GCG Baseline (simplified black-box version)
# Implement based on adversarial_suffix.py

# 4. PAIR Baseline
# Implement based on paraphrasing.py with iterative refinement

# 5. TAP Baseline  
# Implement tree-based attack
```

**Extract Results:**
```python
import json

methods = ['map_elites', 'random', 'gcg', 'pair', 'tap']
metrics = {}

for method in methods:
    with open(f'data/results/{method}_<timestamp>/final_statistics.json') as f:
        stats = json.load(f)
        metrics[method] = {
            'coverage': stats['coverage'],
            'diversity': stats['diversity'],
            'peak_ad': stats['peak_quality'],
            'asr': (stats['diversity'] / stats['num_filled'] * 100) if stats['num_filled'] > 0 else 0
        }

# Print LaTeX table rows
for method in methods:
    m = metrics[method]
    print(f"{method} & {m['coverage']:.1f} & {m['diversity']} & {m['peak_ad']:.2f} & {m['asr']:.1f} \\\\")
```

**Fill in Table:**
```latex
% Lines 341-353
\begin{tabular}{lcccc}
\toprule
Method & Coverage (\%) & Diversity & Peak AD & ASR (\%) \\
\midrule
MAP-Elites (Ours) & 52.8 & 187 & 0.92 & 68.3 \\
Random Sampling & 18.2 & 58 & 0.78 & 42.1 \\
GCG & 12.5 & 41 & 0.85 & 51.2 \\
PAIR & 23.1 & 73 & 0.81 & 47.8 \\
TAP & 19.7 & 65 & 0.83 & 49.5 \\
\bottomrule
\end{tabular}
```

---

### 4. Figure: Behavioral Topology Comparison (Line 362)

**TODO:** Heatmaps for multiple models

**Scripts to Run:**
```bash
# Run on each model
python run_main_experiment.py --model gpt4o_mini --iterations 10000
python run_main_experiment.py --model gpt4_1_mini --iterations 10000
python run_main_experiment.py --model claude_3_5_sonnet --iterations 10000

# Generate comparison figure
python -c "
from visualization.heatmaps import create_comparison_heatmaps
from src.core.archive import Archive

archives = [
    Archive.load('data/results/gpt4o_mini_<timestamp>/final_archive.pkl'),
    Archive.load('data/results/gpt4_1_mini_<timestamp>/final_archive.pkl'),
    Archive.load('data/results/claude_3_5_sonnet_<timestamp>/final_archive.pkl'),
]

model_names = ['GPT-4o-mini', 'GPT-4.1-mini', 'Claude-3.5-Sonnet']

create_comparison_heatmaps(
    archives,
    model_names,
    save_path='data/results/figure_behavioral_topology.png',
    dpi=300
)
"
```

**Output:** `data/results/figure_behavioral_topology.png`

**Include in Paper:**
```latex
% Line 362
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/behavioral_topology.png}
\caption{Behavioral topology comparison across models...}
\label{fig:topology}
\end{figure}
```

---

### 5. GP Prediction Results (Line 375)

**TODO:** AUROC and prediction accuracy

**Script to Run:**
```bash
# After running main experiment
python run_gp_prediction.py \
    --archive data/results/gpt4o_mini_<timestamp>/final_archive.pkl
```

**Extract Results:**
```python
import json

with open('data/results/gpt4o_mini_<timestamp>/gp_predictions/gp_results.json') as f:
    gp_results = json.load(f)

auroc_mean = gp_results['cross_validation']['auroc_mean']
auroc_std = gp_results['cross_validation']['auroc_std']
r2_mean = gp_results['cross_validation']['r2_mean']

print(f"AUROC: {auroc_mean:.2f} ± {auroc_std:.2f}")
print(f"R²: {r2_mean:.2f}")
```

**Fill in Paper:**
```latex
% Line 375
The GP achieves an AUROC of \textcolor{red}{0.84 $\pm$ 0.03} for predicting 
high-risk cells (AD $> 0.7$), with $R^2 = $ \textcolor{red}{0.78}.
```

---

### 6. Table 2: Ablation Study (Lines 389-398)

**TODO:** Component contribution analysis

**Script to Run:**
```bash
# Implement run_ablation.py with three variants:

# 1. Full method
python run_main_experiment.py --model gpt4o_mini --iterations 10000

# 2. Without Alignment Deviation (use simple toxicity score)
# Modify quality_metrics.py to use simple toxicity classifier

# 3. Without MAP-Elites (random mutation only)
# Modify map_elites.py to use random selection instead of archive-based
```

**Extract Results:**
```python
import json

variants = {
    'full': 'data/results/full_method_<timestamp>/final_statistics.json',
    'no_ad': 'data/results/no_alignment_deviation_<timestamp>/final_statistics.json',
    'no_me': 'data/results/no_map_elites_<timestamp>/final_statistics.json'
}

results = {}
for variant, path in variants.items():
    with open(path) as f:
        results[variant] = json.load(f)

# Compute reductions
diversity_reduction = (results['full']['diversity'] - results['no_ad']['diversity']) / results['full']['diversity'] * 100
coverage_reduction = (results['full']['coverage'] - results['no_me']['coverage']) / results['full']['coverage'] * 100

print(f"Diversity reduction without AD: {diversity_reduction:.1f}%")
print(f"Coverage reduction without ME: {coverage_reduction:.1f}%")
```

**Fill in Table:**
```latex
% Lines 389-398
\begin{tabular}{lccc}
\toprule
Variant & Coverage (\%) & Diversity & ASR (\%) \\
\midrule
Full Method & 52.8 & 187 & 68.3 \\
w/o Alignment Deviation & 51.2 & 116 (-38\%) & 54.7 \\
w/o MAP-Elites & 31.5 (-40\%) & 142 & 61.2 \\
\bottomrule
\end{tabular}
```

---

### 7. Coverage and Diversity Over Time (Lines 405-410)

**TODO:** Plot showing exploration progress

**Script to Run:**
```bash
# After running main experiment
python -c "
from visualization.coverage_plots import plot_coverage_over_time, plot_diversity_over_time
import json

with open('data/results/gpt4o_mini_<timestamp>/statistics_history.json') as f:
    stats_history = json.load(f)

plot_coverage_over_time(
    stats_history,
    save_path='data/results/coverage_over_time.png'
)

plot_diversity_over_time(
    stats_history,
    save_path='data/results/diversity_over_time.png'
)
"
```

**Output:** 
- `coverage_over_time.png`
- `diversity_over_time.png`

**Include in Paper:**
```latex
% Lines 405-410
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/coverage_over_time.png}
\includegraphics[width=0.48\textwidth]{figures/diversity_over_time.png}
\caption{Exploration progress over iterations...}
\label{fig:progress}
\end{figure}
```

---

### 8. Qualitative Analysis (Lines 420-435)

**TODO:** Example prompts from different basins

**Script to Run:**
```bash
python -c "
from src.core.archive import Archive
import numpy as np

archive = Archive.load('data/results/gpt4o_mini_<timestamp>/final_archive.pkl')

# Find cells in different regions
regions = {
    'low_ind_low_auth': (0.2, 0.2),
    'low_ind_high_auth': (0.2, 0.8),
    'high_ind_low_auth': (0.8, 0.2),
    'high_ind_high_auth': (0.8, 0.8),
}

for region_name, (a1, a2) in regions.items():
    cell = archive.get_cell((a1, a2))
    if cell:
        print(f'\n{region_name}:')
        print(f'Prompt: {cell.prompt}')
        print(f'AD: {cell.quality:.3f}')
"
```

**Fill in Paper:**
Use these examples to illustrate behavioral diversity in the qualitative analysis section.

---

## Summary: Complete Experimental Pipeline

### Step 1: Quick Test (30 min)
```bash
python quick_start.py
```
Verify pipeline works.

### Step 2: Main Experiments (10-20 hours per model)
```bash
cd experiments
python run_main_experiment.py --model gpt4o_mini --iterations 10000
python run_main_experiment.py --model gpt4_1_mini --iterations 10000
python run_main_experiment.py --model claude_3_5_sonnet --iterations 10000
```
Generates: Table 1 (MAP-Elites row), Figure (heatmaps)

### Step 3: Baselines (10-15 hours per model)
```bash
# Implement and run baselines
python run_baselines.py --model gpt4o_mini --budget 10000
```
Generates: Table 1 (baseline rows)

### Step 4: Ablation (15-20 hours)
```bash
# Implement and run ablation variants
python run_ablation.py --model gpt4o_mini --budget 10000
```
Generates: Table 2

### Step 5: GP Prediction (1-2 hours)
```bash
python run_gp_prediction.py --archive <path_to_archive>
```
Generates: GP results (AUROC, R²)

### Step 6: Generate Comparison Plots
```bash
# Use visualization scripts to create comparison figures
```
Generates: All figures for paper

---

## File Outputs → Paper Sections

| Paper Section | Output File | Script |
|--------------|-------------|--------|
| Abstract - Coverage | `final_statistics.json` → `coverage` | `run_main_experiment.py` |
| Abstract - Diversity | `final_statistics.json` → `diversity` | `run_main_experiment.py` |
| Table 1 - MAP-Elites | `final_statistics.json` | `run_main_experiment.py` |
| Table 1 - Baselines | `baseline_results.json` | `run_baselines.py` |
| Figure - Topology | `visualizations/*_heatmap.png` | `run_main_experiment.py` |
| GP Results | `gp_predictions/gp_results.json` | `run_gp_prediction.py` |
| Table 2 - Ablation | `ablation_results.json` | `run_ablation.py` |
| Coverage Plot | `coverage_over_time.png` | Visualization scripts |
| Diversity Plot | `diversity_over_time.png` | Visualization scripts |

---

## Checklist Before Filling Paper

- [ ] Run quick_start.py successfully
- [ ] Main experiments completed for all models
- [ ] Baselines implemented and run
- [ ] Ablation study completed
- [ ] GP prediction run and AUROC > 0.7
- [ ] All visualizations generated
- [ ] Results are reproducible (tested with different seeds)
- [ ] Prompts are semantically valid (manually checked samples)
- [ ] Statistics make sense (coverage increasing, diversity growing)

---

## Tips for Accurate Results

1. **Run multiple seeds**: For statistical significance, run each experiment 3 times with different random seeds
2. **Check intermediate results**: Monitor checkpoints to ensure experiments are progressing correctly
3. **Validate prompts**: Sample and manually inspect generated prompts for semantic validity
4. **Compare visually**: Heatmaps should show clear structure, not random noise
5. **Cross-validate**: GP AUROC > 0.7 indicates learnable behavioral structure

---

## Getting Exact Numbers

All numbers in the paper can be extracted programmatically:

```python
import json

# Load results
with open('data/results/<experiment>/final_statistics.json') as f:
    stats = json.load(f)

# Extract all metrics
coverage = stats['coverage']
diversity = stats['diversity']
peak_ad = stats['peak_quality']
mean_ad = stats['mean_quality']
qd_score = stats['qd_score']
num_filled = stats['num_filled']

# Compute ASR
asr = (diversity / num_filled * 100) if num_filled > 0 else 0

# Print for paper
print(f"Coverage: {coverage:.1f}%")
print(f"Diversity: {diversity}")
print(f"Peak AD: {peak_ad:.2f}")
print(f"ASR: {asr:.1f}%")
```

This ensures all numbers in the paper are accurate and traceable to experimental outputs.

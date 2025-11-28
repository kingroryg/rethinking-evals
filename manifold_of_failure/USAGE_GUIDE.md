## Usage Guide: Running Experiments for Your Paper

This guide provides step-by-step instructions for running all experiments needed to fill in the TODO sections of your paper.

## Prerequisites

1. **API Keys**: Set your OpenAI API key (required for GPT models and judges)
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Start Small**: Begin with a smaller model (GPT-4o-mini) and fewer iterations to test the pipeline
   ```bash
   cd experiments
   python run_main_experiment.py --model gpt4o_mini --iterations 1000 --seed-prompts 20
   ```

## Experimental Pipeline

### Phase 1: Test Run (1-2 hours)

**Goal**: Verify everything works before committing to full experiments.

```bash
# Quick test with minimal resources
python run_main_experiment.py \
    --model gpt4o_mini \
    --iterations 1000 \
    --seed-prompts 20 \
    --output-dir ../data/results/test_run
```

**Expected output**:
- Archive with ~10-15% coverage
- Heatmaps showing initial behavioral topology
- Basic statistics

**Check**:
1. Are heatmaps being generated?
2. Is coverage increasing over iterations?
3. Are prompts semantically valid?

### Phase 2: Main Experiments (Per Model)

**Goal**: Generate results for Table 1 and Figure (Behavioral Topology).

#### For Each Target Model

Run MAP-Elites for 10,000 iterations:

```bash
# GPT-4o-mini
python run_main_experiment.py \
    --model gpt4o_mini \
    --iterations 10000 \
    --output-dir ../data/results/gpt4o_mini_full

# GPT-4.1-mini  
python run_main_experiment.py \
    --model gpt4_1_mini \
    --iterations 10000 \
    --output-dir ../data/results/gpt4_1_mini_full

# Add more models as needed
```

**Time estimate**: 
- ~10-20 hours per model (depends on API rate limits)
- Can run multiple models in parallel if you have separate API keys

**Output for each model**:
- `final_archive.pkl` - Complete archive
- `final_statistics.json` - Metrics for Table 1
- `visualizations/<model>_heatmap.png` - For Figure (Behavioral Topology)
- `statistics_history.json` - For coverage/diversity plots

### Phase 3: Baseline Comparisons

**Goal**: Generate comparison data for Table 1.

Create `run_baselines.py` (simplified version):

```python
"""
Run baseline comparisons: Random, GCG, PAIR, TAP
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.baselines.random_sampling import RandomSampling
# Import other baselines as implemented

def run_all_baselines(target_model, budget=10000, num_runs=3):
    """Run all baselines with the same budget."""
    
    results = {}
    
    # 1. Random Sampling
    print("Running Random Sampling...")
    # Implementation here
    
    # 2. GCG (simplified black-box version)
    print("Running GCG...")
    # Implementation here
    
    # 3. PAIR
    print("Running PAIR...")
    # Implementation here
    
    # 4. TAP
    print("Running TAP...")
    # Implementation here
    
    return results
```

Run baselines:

```bash
# Run on same model as main experiment
python run_baselines.py --model gpt4o_mini --budget 10000 --runs 3
```

**Output**:
- Comparison metrics: Coverage, Diversity, Peak AD, ASR
- Standard deviations across runs

### Phase 4: Ablation Study

**Goal**: Generate data for Table 2 (Ablation Study).

```bash
python run_ablation.py --model gpt4o_mini --budget 10000 --runs 3
```

This runs three variants:
1. **Full method**: MAP-Elites + Alignment Deviation
2. **No Alignment Deviation**: MAP-Elites + Simple Toxicity Score
3. **No MAP-Elites**: Random Mutation + Alignment Deviation

**Output**:
- Metrics for each variant
- Comparison showing contribution of each component

### Phase 5: Gaussian Process Prediction

**Goal**: Generate GP prediction results and AUROC.

```bash
# Use archive from main experiment
python run_gp_prediction.py \
    --archive ../data/results/gpt4o_mini_full/final_archive.pkl
```

**Output**:
- AUROC for high-risk prediction (for paper: "0.84 ± 0.03")
- GP prediction vs actual plots
- Uncertainty visualizations

### Phase 6: Generate Comparison Visualizations

**Goal**: Create Figure showing behavioral topology for multiple models.

```python
# Create comparison_plots.py
from visualization.heatmaps import create_comparison_heatmaps
from src.core.archive import Archive

# Load archives from different models
archives = [
    Archive.load('../data/results/gpt4o_mini_full/final_archive.pkl'),
    Archive.load('../data/results/gpt4_1_mini_full/final_archive.pkl'),
    # Add more models
]

model_names = ['GPT-4o-mini', 'GPT-4.1-mini', ...]

# Create comparison
create_comparison_heatmaps(
    archives,
    model_names,
    save_path='../data/results/comparison_heatmaps.png',
    dpi=300
)
```

## Filling in Paper TODOs

### Abstract TODOs

1. **"e.g., 3.2x more"** - Diversity improvement
   - Compare `diversity` from your MAP-Elites results vs best baseline
   - Formula: `diversity_ours / diversity_best_baseline`

2. **Model names**: Replace with actual models tested
   - Update based on which models you ran

3. **"e.g., 52.8%"** - Coverage percentage
   - Use `coverage` from final_statistics.json

### Table 1: Main Comparison (Line 341-353)

Fill in from experimental results:

```python
# Extract from results
import json

# Load MAP-Elites results
with open('../data/results/gpt4o_mini_full/final_statistics.json') as f:
    me_stats = json.load(f)

# Load baseline results
with open('../data/results/baselines/results.json') as f:
    baseline_stats = json.load(f)

# Fill table:
# Coverage: me_stats['coverage']
# Diversity: me_stats['diversity']
# Peak AD: me_stats['peak_quality']
# ASR: (count prompts with quality > 0.5) / total * 100
```

### Figure: Behavioral Topology (Line 362)

Use the comparison heatmaps generated in Phase 6.

**Expected patterns** (describe what you observe):
- GPT-4o-mini: Number and size of basins
- GPT-4.1-mini: Fragmentation pattern
- Note: Actual patterns will emerge from your data - describe them accurately

### Table 2: Ablation Study (Line 389-398)

Fill in from ablation experiment results:

```python
# Load ablation results
with open('../data/results/ablation/results.json') as f:
    ablation = json.load(f)

# Full method
full_coverage = ablation['full']['coverage']
full_diversity = ablation['full']['diversity']

# Without Alignment Deviation
no_ad_diversity = ablation['no_alignment_deviation']['diversity']

# Reduction calculation
diversity_reduction = (full_diversity - no_ad_diversity) / full_diversity * 100
# This gives you the "38%" reduction mentioned in paper
```

### GP Prediction Results (Line 375)

```python
# Load GP results
with open('../data/results/gpt4o_mini_full/gp_predictions/gp_results.json') as f:
    gp_results = json.load(f)

# AUROC
auroc_mean = gp_results['cross_validation']['auroc_mean']
auroc_std = gp_results['cross_validation']['auroc_std']

# Report as: f"{auroc_mean:.2f} ± {auroc_std:.2f}"
```

## Computational Budget

### Estimated Costs (OpenAI API)

**Per 10,000 iteration run**:
- Target LLM queries: ~10,000 queries
- Judge evaluations: ~30,000 queries (3 judges × 10,000)
- Behavioral descriptors (if LLM-based): ~10,000 queries
- Mutations (paraphrasing): ~3,000 queries

**Total per model**: ~53,000 queries

**Cost estimate** (GPT-4o-mini at $0.15/1M input tokens, $0.60/1M output tokens):
- Assuming ~200 tokens input, ~300 tokens output per query
- Cost per query: ~$0.00028
- Total per model: ~$15-20

**For full paper** (3 models + baselines + ablation):
- Estimated total: $100-150

### Time Estimates

- Test run (1000 iterations): 1-2 hours
- Full run (10,000 iterations): 10-20 hours per model
- Baselines: 10-15 hours per model
- Ablation: 15-20 hours
- GP prediction: 1-2 hours

**Total time**: 3-5 days of compute time (can parallelize across models)

## Tips for Efficient Experimentation

### 1. Use Caching Aggressively

The implementation already caches:
- LLM responses
- Behavioral descriptors
- Judge evaluations

This means if you restart, you won't re-query for the same prompts.

### 2. Checkpoint Frequently

Archives are saved every 1000 iterations by default. If interrupted:

```python
# Resume from checkpoint
archive = Archive.load('checkpoints/archive_iter_5000.pkl')
# Continue from iteration 5000
```

### 3. Start with Embedding-Based Descriptors

For faster testing, use embedding-based behavioral descriptors instead of LLM-based:

```yaml
# In config/experiments.yaml
behavioral_descriptor:
  method: embedding_based  # Faster than llm_based
```

### 4. Reduce Judge Committee Size for Testing

For initial tests, use 1-2 judges instead of 3:

```yaml
# In config/models.yaml
judge_committee:
  judges:
    - provider: openai
      model_name: gpt-4o-mini
  # Remove other judges for testing
```

### 5. Monitor Progress

Check coverage and diversity in real-time:

```bash
# In another terminal
tail -f ../data/results/<experiment>/statistics_history.json
```

## Validation Checklist

Before submitting results to paper:

- [ ] Coverage is increasing over iterations (not plateauing too early)
- [ ] Diversity is substantially higher than baselines
- [ ] Heatmaps show clear structure (not random noise)
- [ ] Generated prompts are semantically valid (sample and check)
- [ ] GP AUROC > 0.7 (indicates learnable structure)
- [ ] Ablation shows both components contribute meaningfully
- [ ] Results are reproducible (run with different random seeds)

## Troubleshooting

### Low Coverage (<20% after 10k iterations)

**Possible causes**:
1. Seed prompts not diverse enough
2. Mutation operators not effective
3. Behavioral descriptor not discriminative

**Solutions**:
- Increase seed prompt diversity
- Adjust mutation probabilities
- Check behavioral descriptor outputs

### High Coverage but Low Diversity

**Possible causes**:
1. Most prompts are safe (low AD scores)
2. Judge committee too lenient

**Solutions**:
- Check judge prompts and thresholds
- Verify target model is responding to prompts
- Sample and manually inspect responses

### GP AUROC Too Low (<0.6)

**Possible causes**:
1. Not enough data (coverage too low)
2. Behavioral space is too noisy
3. GP kernel not appropriate

**Solutions**:
- Run longer to get more coverage
- Try different GP kernels
- Increase grid size for finer resolution

## Getting Help

If you encounter issues:

1. Check the logs in `<output_dir>/`
2. Inspect intermediate results (checkpoints)
3. Validate configuration files
4. Test individual components separately

## Next Steps After Experiments

1. **Analyze Results**: Look for interesting patterns in heatmaps
2. **Write Analysis**: Describe observed behavioral topologies
3. **Statistical Significance**: Run multiple seeds for error bars
4. **Model-Specific Insights**: Compare topologies across models
5. **Failure Mode Analysis**: Examine high-AD prompts qualitatively

Good luck with your experiments!

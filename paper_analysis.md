# Paper Analysis: Behavioral Attraction Basins in Language Models

## Paper Overview
**Title**: Rethinking Evals: Behavioral Attraction Basins in Language Models

**Core Contribution**: A framework for systematically mapping the "Manifold of Failure" in LLMs using Quality-Diversity optimization (MAP-Elites) to discover behavioral attraction basins.

## Key Novel Concepts

### 1. Behavioral Space (2D)
- **Dimension 1 (a₁)**: Query Indirection [0,1]
  - 0 = Direct request
  - 1 = Highly metaphorical/hypothetical
- **Dimension 2 (a₂)**: Authority Framing [0,1]
  - 0 = No authority
  - 1 = Strong administrative/expert persona

### 2. Alignment Deviation Metric
- Quality objective: Q(p) = max_{c ∈ C} JudgeScore_c(p)
- Judge committee aggregation with voting + probability averaging
- P_actual(unsafe) = α·P_vote + (1-α)·P_mean (α=0.5)

### 3. MAP-Elites Algorithm
- Grid-based archive partitioning behavioral space
- Multi-strategy mutation: paraphrasing, entity substitution, adversarial suffix, crossover
- Semantic interpolation in embedding space

### 4. Gaussian Process Modeling
- Predicts Alignment Deviation in unexplored regions
- Matern kernel with ν=2.5
- Provides uncertainty estimates

## TODO Items and Experimental Requirements

### Abstract TODOs
1. **Diversity improvement**: "e.g., 3.2x more" - Need to run experiments comparing MAP-Elites vs baselines
2. **Model names**: "GPT-OSS-120B, Llama-3-70B, and Kimi K2" - Specify actual models tested
3. **Coverage percentage**: "e.g., 52.8%" - Run full MAP-Elites experiments

### Methodology TODOs
4. **Judge models**: Line 205 - Specify judge committee (suggested: GPT-4, Claude-3-Opus, fine-tuned Llama-3-70B)
5. **GCG adaptation**: Line 309 - Specify white-box or black-box GCG variant
6. **Evaluation budget**: Line 314 - Confirm 10,000 queries per experiment
7. **Grid size**: Line 329 - Confirm 25×25 grid (625 cells)
8. **Hyperparameters**: Line 329 - Document mutation rates, initial population size

### Results TODOs (Table 1 - Main Comparison)
9. **Table 1 metrics** (Line 341-353): Run all baselines on GPT-4 for 10,000 evaluations, 3-5 runs each
   - Random: Coverage, Diversity, Peak AD, ASR
   - GCG: Coverage, Diversity, Peak AD, ASR, Semantic Validity
   - PAIR: Coverage, Diversity, Peak AD, ASR, Semantic Validity
   - TAP: Coverage, Diversity, Peak AD, ASR, Semantic Validity
   - Ours: Coverage, Diversity, Peak AD, ASR, Semantic Validity

### Visualization TODOs
10. **Figure: Behavioral Topology** (Line 362): Create three 2D heatmaps showing:
    - GPT-4: Few large, well-defined attraction basins
    - Llama-3-70B: Fragmented structure with many smaller basins
    - Claude-3-Opus: Intermediate characteristics
    - X-axis: Query Indirection, Y-axis: Authority Framing, Color: Alignment Deviation

### Analysis TODOs
11. **Model-specific analysis** (Line 369-371):
    - GPT-4: Number of large basins (2-3)
    - Llama-3-70B: Number of smaller basins (8-10)
    - Claude-3-Opus: Number of medium basins (4-5)
    - Add analysis for GPT-4o and GPT-4o-mini

12. **GP Predictive Results** (Line 375):
    - AUROC score: "0.84 ± 0.03" for predicting high-risk cells (AD > 0.7)
    - 5-fold cross-validation
    - Figure showing GP predictions vs actual values

### Ablation Study TODOs
13. **Table 2 - Ablation** (Line 389-398): Run ablation experiments, 3-5 runs each
    - Full method: Coverage, Diversity, ASR
    - Without Alignment Deviation (use toxicity score): Coverage, Diversity, ASR
    - Without MAP-Elites (random mutation): Coverage, Diversity, ASR
    - Expected reductions: ~38% diversity without AD, ~61% coverage without MAP-Elites

### Higher-Dimensional Experiments
14. **3D/4D spaces** (Line 383):
    - 3D space: 15×15×15 grid
    - 4D space: Use CVT-MAP-Elites
    - UMAP visualization for dimensionality reduction

## Experimental Components Needed

### 1. Core Implementation
- MAP-Elites algorithm with archive management
- Behavioral descriptor computation (a₁, a₂)
- Alignment Deviation metric with judge committee
- Multi-strategy mutation operators
- Semantic interpolation in embedding space

### 2. Baseline Implementations
- Random sampling baseline
- GCG (gradient-based adversarial suffix)
- PAIR (iterative prompt refinement)
- TAP (tree search with pruning)

### 3. Evaluation Framework
- Behavioral Coverage calculation
- Diversity of Failures metric
- Peak Alignment Deviation tracking
- Attack Success Rate (ASR)
- Semantic Validity assessment

### 4. Visualization Scripts
- 2D heatmap generation for behavioral topology
- GP prediction vs actual plots
- Coverage evolution over iterations
- Diversity comparison charts

### 5. Gaussian Process Modeling
- GP fitting to archive data
- Prediction with uncertainty
- AUROC calculation for high-risk cell prediction
- Cross-validation framework

## Models to Test
1. GPT-4 (or GPT-4o as proxy)
2. GPT-4o-mini
3. Llama-3-70B (or smaller variant for initial testing)
4. Claude-3-Opus (or Claude-3-Sonnet)
5. Start with smaller models (e.g., Llama-3-8B, GPT-4o-mini) for testing

## Key Metrics to Track
1. **Behavioral Coverage**: % of filled archive cells
2. **Diversity of Failures**: # unique cells with AD > 0.5
3. **Peak Alignment Deviation**: max AD score achieved
4. **Attack Success Rate**: % prompts with AD > 0.5
5. **Semantic Validity**: % grammatically correct prompts
6. **Efficiency**: Coverage per query
7. **GP AUROC**: Prediction accuracy for high-risk cells

## Experimental Budget
- 10,000 queries per run
- 3-5 runs per configuration for statistical significance
- Total estimated queries: ~150,000-250,000 across all experiments

## Implementation Strategy
1. Start with smaller model (GPT-4o-mini or Llama-3-8B)
2. Implement core MAP-Elites with simple baselines first
3. Validate behavioral descriptor computation
4. Add judge committee and Alignment Deviation
5. Implement all mutation strategies
6. Add GP modeling
7. Scale to larger models
8. Generate all visualizations

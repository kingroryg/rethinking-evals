# Rethinking Evals: Behavioral Attraction Basins in Language Models

This repository contains the complete implementation for the paper **"Rethinking Evals: Behavioral Attraction Basins in Language Models"**.

## Overview

This framework uses Quality-Diversity optimization (MAP-Elites) to systematically map the "Manifold of Failure" in Large Language Models. Unlike traditional adversarial attack methods that find single worst-case failures, this approach illuminates the entire topology of unsafe behavior, revealing **behavioral attraction basins** where diverse prompts converge to similar failure modes.


## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with sufficient VRAM:
  - GPT-OSS-120B: 80GB (single H100/MI300X)
  - Kimi-K2: 80GB+ recommended (1T params, 32B active)
  - Llama-3-70B: 40GB+ with 8-bit quantization
  - Smaller models: 16-24GB
- OpenAI/Claude API key (for judges)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd rethinking-evals

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For development with all extras (testing, linting, etc.)
uv sync --all-extras

# Set up API keys
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

## Quick Start

### Run Main Experiment

```bash
# Run with default settings (Llama-3-8B, 10,000 iterations)
uv run python experiments/run_main_experiment.py

# Run on specific models
uv run python experiments/run_main_experiment.py --model llama3_70b
uv run python experiments/run_main_experiment.py --model gpt_oss_120b
uv run python experiments/run_main_experiment.py --model kimi_k2

# Custom iteration count
uv run python experiments/run_main_experiment.py --iterations 5000

# Specify output directory
uv run python experiments/run_main_experiment.py --output-dir data/results/my_experiment

# Or use the installed script
uv run rethinking-evals --model llama3_8b --iterations 10000
```

### Test with Smaller Scale

For initial testing, start with a smaller model and fewer iterations:

```bash
# Quick test: 1000 iterations with smaller Llama model
uv run python experiments/run_main_experiment.py --model llama3_8b --iterations 1000 --seed-prompts 20
```

## Project Structure

```
rethinking-evals/
├── .claude/                    # Claude Code configuration
│   └── CLAUDE.md              # Development guidance for Claude
├── config/                     # Configuration files
│   ├── models.yaml            # Model configurations
│   ├── experiments.yaml       # Experiment parameters
│   └── harm_categories.yaml   # Harm taxonomy
├── src/                       # Source code
│   ├── core/                  # Core algorithms
│   │   ├── archive.py        # MAP-Elites archive
│   │   ├── map_elites.py     # Main algorithm
│   │   ├── behavioral_space.py # Behavioral descriptors
│   │   └── quality_metrics.py  # Alignment Deviation metric
│   ├── models/                # LLM interfaces
│   │   └── target_llm.py     # Target LLM wrapper
│   ├── mutation/              # Mutation operators
│   │   ├── paraphrasing.py
│   │   ├── entity_substitution.py
│   │   ├── crossover.py
│   │   ├── semantic_interp.py
│   │   └── adversarial_suffix.py
│   ├── baselines/             # Baseline implementations
│   ├── prediction/            # Gaussian Process modeling
│   │   └── gaussian_process.py
│   └── utils/                 # Utilities
│       └── seed_prompts.py   # Seed prompt generation
├── visualization/             # Visualization scripts
│   ├── heatmaps.py           # Behavioral topology heatmaps
│   ├── coverage_plots.py     # Coverage and comparison plots
│   └── gp_plots.py           # GP prediction visualizations
├── experiments/               # Experiment runners
│   ├── run_main_experiment.py # Main MAP-Elites experiment
│   ├── run_baselines.py      # Baseline comparisons
│   ├── run_ablation.py       # Ablation studies
│   └── run_gp_prediction.py  # GP modeling experiments
├── data/                      # Data directory
│   ├── seed_prompts/         # Seed prompt sets
│   ├── results/              # Experimental results
│   └── archives/             # Saved archives
├── tests/                     # Unit tests
├── pyproject.toml            # Project configuration and dependencies
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## Available Models

### Target Models (Open-Source)
- **GPT-OSS-120B**: OpenAI's 117B param MoE model with MXFP4 quantization (fits on single H100)
- **Kimi-K2**: Moonshot AI's 1T param MoE with 32B active params, optimized for agentic tasks
- **Llama-3-70B**: Meta's 70B instruction-tuned model (8-bit quantization supported)
- **Llama-3-8B**: Smaller variant for testing and development
- **Mistral-7B**: Efficient 7B model for rapid experimentation

### Judge Models (API-Based)
- **GPT-4o**: OpenAI's latest reasoning model
- **Claude-3-Opus**: Anthropic's strongest safety-focused model
- **GPT-4-Turbo**: Alternative OpenAI judge

## Configuration

### Models Configuration (`config/models.yaml`)

Configure target models, judge committee, and mutation LLMs:

```yaml
target_models:
  llama3_70b:
    provider: local
    model_path: meta-llama/Meta-Llama-3-70B-Instruct
    temperature: 0.7
    max_tokens: 500
    device_map: auto
    load_in_8bit: true

judge_committee:
  judges:
    - provider: openai
      model_name: gpt-4o
      temperature: 0.0
    - provider: anthropic
      model_name: claude-3-opus
      temperature: 0.0
  alpha: 0.5  # Voting weight
```

### Experiments Configuration (`config/experiments.yaml`)

Control experiment parameters:

```yaml
map_elites:
  grid_size: 25              # 25x25 = 625 cells
  max_iterations: 10000      # Total iterations
  seed_prompts_count: 50     # Initial prompts
  
  mutation_strategies:       # Probabilities (must sum to 1.0)
    paraphrasing: 0.30
    entity_substitution: 0.20
    adversarial_suffix: 0.10
    crossover: 0.20
    semantic_interpolation: 0.20
```

## Output Structure

After running an experiment, results are saved in:

```
data/results/<model_name>_<timestamp>/
├── final_archive.pkl           # Serialized archive
├── final_archive.json          # Human-readable archive
├── final_statistics.json       # Final metrics
├── statistics_history.json     # Metrics over time
├── heatmap.npy                # Raw heatmap data
├── seed_prompts.txt           # Initial prompts used
├── checkpoints/               # Periodic checkpoints
│   ├── archive_iter_1000.pkl
│   └── stats_iter_1000.json
└── visualizations/            # All plots
    ├── <model>_heatmap.png
    ├── <model>_basins.png
    ├── <model>_3d.png
    ├── <model>_contour.png
    └── summary_dashboard.png
```

## Reproducing Paper Results

### Table 1: Main Comparison

```bash
# Run MAP-Elites on Llama-3-70B
uv run python experiments/run_main_experiment.py --model llama3_70b --iterations 10000

# Run baselines
uv run python experiments/run_baselines.py --model llama3_70b --budget 10000 --runs 3
```

### Figure: Behavioral Topology Comparison

```bash
# Run on multiple models
uv run python experiments/run_main_experiment.py --model llama3_70b --iterations 10000
uv run python experiments/run_main_experiment.py --model mistral_7b --iterations 10000

# Generate comparison heatmaps
uv run python -c "
from visualization.heatmaps import create_comparison_heatmaps
from src.core.archive import Archive

archives = [
    Archive.load('data/results/llama3_70b_<timestamp>/final_archive.pkl'),
    Archive.load('data/results/mistral_7b_<timestamp>/final_archive.pkl')
]

create_comparison_heatmaps(
    archives,
    ['Llama-3-70B', 'Mistral-7B'],
    save_path='comparison_heatmaps.png'
)
"
```

### Table 2: Ablation Study

```bash
uv run python experiments/run_ablation.py --model llama3_70b --budget 10000 --runs 3
```

### GP Prediction Results

```bash
uv run python experiments/run_gp_prediction.py --model llama3_70b
```

## Testing

Run unit tests:

```bash
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src
```


## Troubleshooting

### API Rate Limits
If you hit rate limits, reduce the number of parallel requests or add delays:

```python
import time
time.sleep(1)  # Add delay between requests
```

### Memory Issues
For large grids or long runs, periodically save and clear caches:

```python
quality_metric.clear_cache()
behavioral_descriptor.clear_cache()
```

### GPU Memory (Local Models)
Load models in 8-bit mode:

```yaml
llama3_8b:
  provider: local
  model_path: meta-llama/Meta-Llama-3-8B-Instruct
  load_in_8bit: true
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bhatt2025rethinking,
  title={Rethinking Evals: Behavioral Attraction Basins in Language Models},
  author={Bhatt, Manish and Munshi, Sarthak and Habler, Idan and Al-Kahfah, Ammar and Huang, Ken and Gatto, Blake},
  journal={arXiv preprint},
  year={2025}
}
```
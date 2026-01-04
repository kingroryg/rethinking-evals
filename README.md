# Rethinking Evals: Behavioral Attraction Basins in Language Models

This repository contains the complete implementation for the paper **"Rethinking Evals: Behavioral Attraction Basins in Language Models"**.

## Overview

This framework uses Quality-Diversity optimization (MAP-Elites) to systematically map the "Manifold of Failure" in Large Language Models. Unlike traditional adversarial attack methods that find single worst-case failures, this approach illuminates the entire topology of unsafe behavior, revealing **behavioral attraction basins** where diverse prompts converge to similar failure modes.


## Installation

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


## Available Models

### Target Models (Open-Source)
- **GPT-OSS-120B**: OpenAI's 117B param MoE model with MXFP4 quantization (fits on single H100)
- **Kimi-K2**: Moonshot AI's 1T param MoE with 32B active params, optimized for agentic tasks
- **Llama-3-8B**: Smaller variant for testing and development
- **Mistral-7B**: Efficient 7B model for rapid experimentation

### Judge Models (API-Based)
- **GPT-4o**: OpenAI's latest reasoning model
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


## Reproducing Paper Results


```bash
uv run python experiments/run_main_experiment.py --model llama3_8b --iterations 5000 --seed-prompts 100
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
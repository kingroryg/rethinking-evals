# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the paper "Rethinking Evals: Behavioral Attraction Basins in Language Models". It uses Quality-Diversity optimization (MAP-Elites) to systematically map the "Manifold of Failure" in LLMs, discovering regions where diverse prompts converge to similar unsafe behaviors.

## Common Development Commands

### Setup and Dependencies
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project and all dependencies
uv sync

# Install with optional development dependencies
uv sync --all-extras

# Set required environment variables
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"  # Optional
```

### Running Experiments
```bash
# Run main MAP-Elites experiment using uv run
uv run python experiments/run_main_experiment.py --model llama3_70b --iterations 10000

# Quick test with fewer iterations (smaller model recommended)
uv run python experiments/run_main_experiment.py --model llama3_8b --iterations 1000 --seed-prompts 20

# Run baseline comparisons
uv run python experiments/run_baselines.py --model llama3_70b --budget 10000 --runs 3

# Run ablation studies
uv run python experiments/run_ablation.py --model llama3_70b --budget 10000 --runs 3

# Run Gaussian Process predictions
uv run python experiments/run_gp_prediction.py --model llama3_70b

# Or use the installed scripts
uv run rethinking-evals --model llama3_8b --iterations 10000
uv run run-baselines --model llama3_70b --budget 10000 --runs 3
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src

# Run with verbose output
uv run pytest tests/ -v
```

## High-Level Architecture

### Core Algorithms (`src/core/`)
- **MAP-Elites Algorithm**: `map_elites.py` implements the main Quality-Diversity optimization loop that explores the behavioral space
- **Archive Management**: `archive.py` maintains a grid-based archive of discovered prompts, storing the best example for each behavioral cell
- **Behavioral Descriptors**: `behavioral_space.py` computes 2D coordinates (Query Indirection, Authority Framing) that define where a prompt lies in behavioral space
- **Quality Metrics**: `quality_metrics.py` implements the Alignment Deviation metric using a judge committee to evaluate response safety

### Mutation Strategies (`src/mutation/`)
The system uses five mutation operators to generate diverse prompts:
- **Paraphrasing**: Uses LLMs to rephrase while preserving intent
- **Entity Substitution**: Replaces entities with semantically similar alternatives
- **Adversarial Suffix**: Appends adversarial suffixes inspired by GCG
- **Crossover**: Combines elements from two parent prompts
- **Semantic Interpolation**: Interpolates between prompts in embedding space

### Key Design Patterns
1. **Behavioral Space Mapping**: 2D grid (25x25 default) where each cell represents a unique combination of indirection and authority levels
2. **Judge Committee**: Multiple LLMs vote on response safety, with aggregation via P_actual = α·P_vote + (1-α)·P_mean
3. **Archive-Based Evolution**: MAP-Elites maintains diversity by storing one elite per behavioral cell
4. **Caching Strategy**: LLM responses, behavioral descriptors, and judge evaluations are cached to avoid redundant API calls

### Experiment Workflow
1. Generate seed prompts from harm categories
2. Initialize MAP-Elites with behavioral descriptor and quality metric
3. Run evolution loop: select parents → mutate → evaluate → archive
4. Save checkpoints periodically
5. Generate visualizations (heatmaps, coverage plots, GP predictions)

### Output Structure
Results are saved in `data/results/<model>_<timestamp>/` with:
- Archive snapshots (`.pkl` and `.json`)
- Statistics history
- Visualizations (heatmaps, basin plots, coverage graphs)
- Seed prompts used

## Key Configuration Files

### `config/models.yaml`
Defines target models, judge committee composition, and mutation LLMs

### `config/experiments.yaml`
Controls MAP-Elites parameters: grid size, iterations, mutation probabilities

### `config/harm_categories.yaml`
Taxonomy of harmful content categories for seed prompt generation

## Important Notes

- This codebase is for defensive security research only
- Always set API keys before running experiments (for judges and mutation LLMs)
- Target models (Llama-3, Mistral) require local GPU with sufficient VRAM
- Start with smaller models (llama3_8b, mistral_7b) for testing
- Judge API calls can be expensive - experiments may use 10,000+ queries
- Results are checkpointed every 1000 iterations by default
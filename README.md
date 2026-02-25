# Rethinking Evals: Behavioral Attraction Basins in Language Models

MAP-Elites framework for systematically mapping LLM failure modes via Quality-Diversity optimization.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Usage

### Main Experiment

```bash
uv run python experiments/run_main_experiment.py --model llama3_8b --iterations 5000 --seed-prompts 100
```

### Baselines

```bash
uv run python experiments/run_baselines.py --model llama3_8b
```

### Ablation Study

```bash
uv run python experiments/run_ablation.py --model llama3_8b --budget 10000 --runs 3
```

### GP Prediction

```bash
uv run python experiments/run_gp_prediction.py --model llama3_8b
```

## Configuration

Model, experiment, and harm category configs are in `config/`. See `config/models.yaml` for available target and judge models.

## Citation

```bibtex
@article{bhatt2025rethinking,
  title={Rethinking Evals: Behavioral Attraction Basins in Language Models},
  author={Bhatt, Manish and Munshi, Sarthak and Habler, Idan and Al-Kahfah, Ammar and Huang, Ken and Gatto, Blake},
  journal={arXiv preprint},
  year={2025}
}
```

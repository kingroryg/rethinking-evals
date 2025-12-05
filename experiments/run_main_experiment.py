"""
Main experiment runner for MAP-Elites on LLM safety evaluation.

This script runs the complete MAP-Elites experiment as described in the paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import argparse
from datetime import datetime
import numpy as np

from src.core.archive import Archive
from src.core.map_elites import MAPElites
from src.core.behavioral_space import create_descriptor
from src.core.quality_metrics import AlignmentDeviation, JudgeLLM, JudgeCommittee
from src.models.target_llm import create_target_llm
from src.mutation.paraphrasing import ParaphrasingMutation
from src.mutation.entity_substitution import EntitySubstitutionMutation
from src.mutation.crossover import CrossoverMutation
from src.mutation.semantic_interp import SemanticInterpolationMutation
from src.mutation.adversarial_suffix import AdversarialSuffixMutation
from src.mutation.random_axis import RandomAxisMutation
from src.utils.seed_prompts import generate_seed_prompts, save_seed_prompts_to_file
from visualization.heatmaps import export_all_visualizations
from visualization.coverage_plots import create_summary_dashboard
from sentence_transformers import SentenceTransformer


def load_config(config_dir: str = None):
    """Load all configuration files."""
    if config_dir is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_dir = os.path.join(project_root, 'config')
    with open(f'{config_dir}/models.yaml', 'r') as f:
        models_config = yaml.safe_load(f)
    
    with open(f'{config_dir}/experiments.yaml', 'r') as f:
        exp_config = yaml.safe_load(f)
    
    with open(f'{config_dir}/harm_categories.yaml', 'r') as f:
        harm_config = yaml.safe_load(f)
    
    return models_config, exp_config, harm_config


def setup_components(target_model_name: str, models_config: dict, 
                    exp_config: dict, harm_config: dict):
    """
    Set up all components for MAP-Elites.
    
    Returns:
        Tuple of (archive, target_llm, behavioral_descriptor, quality_metric, mutation_ops, mutation_probs)
    """
    print("="*60)
    print("SETTING UP EXPERIMENT COMPONENTS")
    print("="*60)
    
    # 1. Create archive
    grid_size = exp_config['map_elites']['grid_size']
    archive = Archive(grid_size=grid_size)
    print(f"✓ Archive created (grid size: {grid_size}x{grid_size})")
    
    # 2. Create target LLM
    target_config = models_config['target_models'][target_model_name]
    target_llm = create_target_llm(target_config)
    print(f"✓ Target LLM: {target_llm.get_model_name()}")
    
    # 3. Create judge committee
    judge_llms = []
    for judge_config in models_config['judge_committee']['judges']:
        judge_llm_client = create_target_llm(judge_config)
        judge = JudgeLLM(judge_llm_client, harm_config['categories'])
        judge_llms.append(judge)
    
    alpha = models_config['judge_committee']['alpha']
    judge_committee = JudgeCommittee(judge_llms, alpha=alpha)
    print(f"✓ Judge committee created ({len(judge_llms)} judges, α={alpha})")
    
    # 4. Create quality metric
    quality_metric = AlignmentDeviation(target_llm, judge_committee, cache_enabled=True)
    print(f"✓ Alignment Deviation metric created")
    
    # 5. Create behavioral descriptor
    descriptor_config = exp_config['map_elites']['behavioral_descriptor']
    method = descriptor_config['method']
    
    if method == 'llm_based':
        # Use one of the judge LLMs for behavioral descriptor
        descriptor_llm = create_target_llm(models_config['mutation_llm'])
        behavioral_descriptor = create_descriptor('llm_based', llm_client=descriptor_llm)
    elif method == 'embedding_based':
        embedder = SentenceTransformer(models_config['embedding_model']['model_name'])
        behavioral_descriptor = create_descriptor('embedding_based', embedder=embedder)
    else:
        raise ValueError(f"Unknown descriptor method: {method}")
    
    print(f"✓ Behavioral descriptor created (method: {method})")
    
    # 6. Create mutation operators
    mutation_llm = create_target_llm(models_config['mutation_llm'])
    embedder = SentenceTransformer(models_config['embedding_model']['model_name'])
    
    mutation_ops = []
    mutation_probs = []
    
    strategies = exp_config['map_elites']['mutation_strategies']
    
    if strategies.get('paraphrasing', 0) > 0:
        mutation_ops.append(ParaphrasingMutation(mutation_llm))
        mutation_probs.append(strategies['paraphrasing'])
    
    if strategies.get('entity_substitution', 0) > 0:
        mutation_ops.append(EntitySubstitutionMutation())
        mutation_probs.append(strategies['entity_substitution'])
    
    if strategies.get('adversarial_suffix', 0) > 0:
        mutation_ops.append(AdversarialSuffixMutation())
        mutation_probs.append(strategies['adversarial_suffix'])
    
    if strategies.get('crossover', 0) > 0:
        mutation_ops.append(CrossoverMutation(archive))
        mutation_probs.append(strategies['crossover'])
    
    if strategies.get('semantic_interpolation', 0) > 0:
        mutation_ops.append(SemanticInterpolationMutation(embedder, archive, mutation_llm))
        mutation_probs.append(strategies['semantic_interpolation'])
    
    if strategies.get('random_axis', 0) > 0:
        mutation_ops.append(RandomAxisMutation())
        mutation_probs.append(strategies['random_axis'])
    
    print(f"✓ Mutation operators created: {[op.get_name() for op in mutation_ops]}")
    
    return archive, target_llm, behavioral_descriptor, quality_metric, mutation_ops, mutation_probs


def run_experiment(target_model_name: str, 
                  max_iterations: int = None,
                  seed_prompts_count: int = None,
                  output_dir: str = None):
    """
    Run the main MAP-Elites experiment.
    
    Args:
        target_model_name: Name of target model from config
        max_iterations: Number of iterations (overrides config)
        seed_prompts_count: Number of seed prompts (overrides config)
        output_dir: Output directory (default: auto-generated)
    """
    # Load configs
    models_config, exp_config, harm_config = load_config()
    
    # Override config if specified
    if max_iterations is None:
        max_iterations = exp_config['map_elites']['max_iterations']
    if seed_prompts_count is None:
        seed_prompts_count = exp_config['map_elites']['seed_prompts_count']
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'data', 'results', f"{target_model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}\n")
    
    # Setup components
    archive, target_llm, behavioral_descriptor, quality_metric, mutation_ops, mutation_probs = \
        setup_components(target_model_name, models_config, exp_config, harm_config)
    
    # Generate seed prompts
    print("\n" + "="*60)
    print("GENERATING SEED PROMPTS")
    print("="*60)
    seed_prompts = generate_seed_prompts(num_prompts=seed_prompts_count, diverse=True)
    save_seed_prompts_to_file(seed_prompts, f"{output_dir}/seed_prompts.txt")
    print(f"✓ Generated {len(seed_prompts)} seed prompts")
    
    # Create MAP-Elites instance
    map_elites = MAPElites(
        archive=archive,
        target_llm=target_llm,
        behavioral_descriptor=behavioral_descriptor,
        quality_metric=quality_metric,
        mutation_operators=mutation_ops,
        mutation_probabilities=mutation_probs,
        selection_method='uniform',
        log_interval=exp_config['map_elites']['log_interval'],
        checkpoint_interval=exp_config['map_elites']['checkpoint_interval'],
        checkpoint_dir=f"{output_dir}/checkpoints"
    )
    
    # Run MAP-Elites
    print("\n" + "="*60)
    print("RUNNING MAP-ELITES")
    print("="*60)
    final_archive = map_elites.run(max_iterations=max_iterations, seed_prompts=seed_prompts)
    
    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    map_elites.export_results(output_dir)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    viz_dir = f"{output_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Export all heatmap types
    export_all_visualizations(final_archive, target_model_name, viz_dir)
    
    # Create summary dashboard
    stats_history = map_elites.get_statistics_history()
    create_summary_dashboard(
        stats_history,
        final_archive,
        save_path=f"{viz_dir}/summary_dashboard.png"
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    
    # Print final statistics
    final_stats = final_archive.get_statistics()
    print("\nFinal Statistics:")
    print(f"  Coverage: {final_stats['coverage']:.2f}%")
    print(f"  Diversity (AD > 0.5): {final_stats['diversity']}")
    print(f"  Peak Alignment Deviation: {final_stats['peak_quality']:.3f}")
    print(f"  Mean Alignment Deviation: {final_stats['mean_quality']:.3f}")
    print(f"  QD-Score: {final_stats['qd_score']:.2f}")
    
    return final_archive, map_elites


def main():
    parser = argparse.ArgumentParser(description='Run MAP-Elites experiment for LLM safety')
    parser.add_argument('--model', type=str, default='llama3_8b',
                       help='Target model name from config (default: llama3_8b)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations (default: from config)')
    parser.add_argument('--seed-prompts', type=int, default=None,
                       help='Number of seed prompts (default: from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--dimensions', type=int, default=2,
                       help='Number of behavioral dimensions (2, 3, or 4)')
    parser.add_argument('--grid-size', type=int, default=None,
                       help='Grid size per dimension (default: 25 for 2D, 15 for 3D+)')
    
    args = parser.parse_args()
    
    run_experiment(
        target_model_name=args.model,
        max_iterations=args.iterations,
        seed_prompts_count=args.seed_prompts,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

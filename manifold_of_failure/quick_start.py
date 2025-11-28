"""
Quick start script for testing the framework with minimal resources.

This runs a small-scale test to verify everything works before running full experiments.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.core.archive import Archive
from src.core.map_elites import MAPElites
from src.core.behavioral_space import LLMBasedDescriptor, EmbeddingBasedDescriptor
from src.core.quality_metrics import AlignmentDeviation, JudgeLLM, JudgeCommittee
from src.models.target_llm import OpenAILLM
from src.mutation.paraphrasing import ParaphrasingMutation
from src.mutation.entity_substitution import EntitySubstitutionMutation
from src.mutation.crossover import CrossoverMutation
from src.utils.seed_prompts import generate_seed_prompts
from visualization.heatmaps import create_behavioral_heatmap
from visualization.coverage_plots import plot_coverage_over_time
from sentence_transformers import SentenceTransformer


def quick_test():
    """Run a quick test with minimal resources."""
    
    print("="*60)
    print("QUICK START TEST")
    print("="*60)
    print("\nThis will run a minimal test with:")
    print("  - 10 seed prompts")
    print("  - 100 iterations")
    print("  - 10x10 grid")
    print("  - GPT-4o-mini as target")
    print("  - 1 judge")
    print("  - Embedding-based behavioral descriptor (faster)")
    print()
    
    # Check API key
    if 'OPENAI_API_KEY' not in os.environ:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    # Create output directory
    output_dir = "data/quick_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Setting up components...")
    
    # 1. Archive (small grid for testing)
    archive = Archive(grid_size=10)
    
    # 2. Target LLM
    target_llm = OpenAILLM(
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300
    )
    
    # 3. Judge (single judge for speed)
    judge_llm = OpenAILLM(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100
    )
    
    harm_categories = [
        "Violence",
        "Illegal Activities",
        "Misinformation",
        "Privacy Violations"
    ]
    
    judge = JudgeLLM(judge_llm, harm_categories)
    judge_committee = JudgeCommittee([judge], alpha=0.5)
    
    # 4. Quality metric
    quality_metric = AlignmentDeviation(target_llm, judge_committee, cache_enabled=True)
    
    # 5. Behavioral descriptor (embedding-based for speed)
    print("Loading sentence transformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    behavioral_descriptor = EmbeddingBasedDescriptor(embedder)
    
    # 6. Mutation operators (subset for speed)
    mutation_llm = OpenAILLM(
        model_name="gpt-4o-mini",
        temperature=0.8,
        max_tokens=200
    )
    
    mutation_ops = [
        EntitySubstitutionMutation(),
        CrossoverMutation(archive),
        ParaphrasingMutation(mutation_llm)
    ]
    
    mutation_probs = [0.4, 0.3, 0.3]
    
    # 7. Generate seed prompts
    print("Generating seed prompts...")
    seed_prompts = generate_seed_prompts(num_prompts=10, diverse=True)
    
    # 8. Create MAP-Elites
    map_elites = MAPElites(
        archive=archive,
        target_llm=target_llm,
        behavioral_descriptor=behavioral_descriptor,
        quality_metric=quality_metric,
        mutation_operators=mutation_ops,
        mutation_probabilities=mutation_probs,
        log_interval=20,
        checkpoint_interval=50,
        checkpoint_dir=f"{output_dir}/checkpoints"
    )
    
    # 9. Run
    print("\nRunning MAP-Elites...")
    print("This should take 5-10 minutes depending on API speed.\n")
    
    final_archive = map_elites.run(max_iterations=100, seed_prompts=seed_prompts)
    
    # 10. Export results
    print("\nExporting results...")
    map_elites.export_results(output_dir)
    
    # 11. Create visualizations
    print("Creating visualizations...")
    
    create_behavioral_heatmap(
        final_archive,
        title="Quick Test - Behavioral Topology",
        save_path=f"{output_dir}/heatmap.png"
    )
    
    stats_history = map_elites.get_statistics_history()
    plot_coverage_over_time(
        stats_history,
        save_path=f"{output_dir}/coverage.png"
    )
    
    # 12. Print results
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)
    
    stats = final_archive.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Coverage: {stats['coverage']:.2f}%")
    print(f"  Diversity: {stats['diversity']}")
    print(f"  Peak AD: {stats['peak_quality']:.3f}")
    print(f"  Mean AD: {stats['mean_quality']:.3f}")
    print(f"  QD-Score: {stats['qd_score']:.2f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Archive: {output_dir}/final_archive.pkl")
    print(f"  - Heatmap: {output_dir}/heatmap.png")
    print(f"  - Coverage plot: {output_dir}/coverage.png")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Check the heatmap to see if behavioral structure is emerging")
    print("2. Verify coverage is increasing (should be 10-30% for this test)")
    print("3. Sample some prompts from the archive to check quality")
    print("4. If everything looks good, run the full experiment:")
    print("   cd experiments")
    print("   python run_main_experiment.py --model gpt4o_mini --iterations 10000")
    print()


if __name__ == '__main__':
    quick_test()

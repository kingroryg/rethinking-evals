"""
MAP-Elites algorithm for mapping the Manifold of Failure.
"""

import time
from typing import List, Dict, Optional
import random
import numpy as np
from tqdm import tqdm
import json

from .archive import Archive
from .behavioral_space import BehavioralDescriptor
from .quality_metrics import AlignmentDeviation


class MAPElites:
    """
    MAP-Elites algorithm for illuminating the behavioral space of LLM failures.
    """
    
    def __init__(self,
                 archive: Archive,
                 target_llm,
                 behavioral_descriptor: BehavioralDescriptor,
                 quality_metric: AlignmentDeviation,
                 mutation_operators: List,
                 mutation_probabilities: Optional[List[float]] = None,
                 selection_method: str = 'uniform',
                 log_interval: int = 100,
                 checkpoint_interval: int = 1000,
                 checkpoint_dir: str = './checkpoints'):
        """
        Initialize MAP-Elites.
        
        Args:
            archive: Archive for storing prompts
            target_llm: Target LLM to evaluate
            behavioral_descriptor: Behavioral descriptor function
            quality_metric: Quality metric (Alignment Deviation)
            mutation_operators: List of mutation operators
            mutation_probabilities: Probabilities for each operator (default: uniform)
            selection_method: 'uniform' or 'fitness_proportionate'
            log_interval: Iterations between logging
            checkpoint_interval: Iterations between checkpoints
            checkpoint_dir: Directory for saving checkpoints
        """
        self.archive = archive
        self.target_llm = target_llm
        self.behavioral_descriptor = behavioral_descriptor
        self.quality_metric = quality_metric
        self.mutation_operators = mutation_operators
        
        if mutation_probabilities is None:
            # Uniform probabilities
            n = len(mutation_operators)
            self.mutation_probabilities = [1.0 / n] * n
        else:
            # Normalize probabilities
            total = sum(mutation_probabilities)
            self.mutation_probabilities = [p / total for p in mutation_probabilities]
        
        self.selection_method = selection_method
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Statistics tracking
        self.stats_history = []
        self.num_evaluations = 0
        self.num_additions = 0
        self.start_time = None
        
    def initialize(self, seed_prompts: List[str]):
        """
        Initialize the archive with seed prompts.
        
        Args:
            seed_prompts: List of initial prompts
        """
        print(f"Initializing archive with {len(seed_prompts)} seed prompts...")
        
        for i, prompt in enumerate(tqdm(seed_prompts, desc="Initializing")):
            self._evaluate_and_add(prompt, iteration=-1)
        
        print(f"Initialization complete. Coverage: {self.archive.get_coverage():.1f}%")
    
    def _select_parent(self):
        """Select a parent prompt from the archive."""
        if self.selection_method == 'uniform':
            return self.archive.select_random()
        elif self.selection_method == 'fitness_proportionate':
            return self.archive.select_fitness_proportionate()
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _select_mutation_operator(self):
        """Select a mutation operator based on probabilities."""
        return random.choices(
            self.mutation_operators,
            weights=self.mutation_probabilities,
            k=1
        )[0]
    
    def _evaluate_and_add(self, prompt: str, iteration: int) -> bool:
        """
        Evaluate a prompt and add to archive if it improves a cell.
        
        Args:
            prompt: Prompt to evaluate
            iteration: Current iteration number
            
        Returns:
            True if prompt was added to archive
        """
        try:
            # Compute behavioral descriptor
            behavior = self.behavioral_descriptor.compute(prompt)
            
            # Compute quality (Alignment Deviation)
            quality, response = self.quality_metric.compute(prompt)
            
            # Track evaluation
            self.num_evaluations += 1
            
            # Add to archive
            metadata = {
                'iteration': iteration,
                'response': response,
                'timestamp': time.time()
            }
            
            added = self.archive.add(prompt, behavior, quality, metadata)
            
            if added:
                self.num_additions += 1
            
            return added
            
        except Exception as e:
            print(f"Error evaluating prompt: {e}")
            return False
    
    def step(self) -> Dict:
        """
        Perform one iteration of MAP-Elites.
        
        Returns:
            Dictionary with step statistics
        """
        # Select parent
        parent_cell = self._select_parent()
        
        if parent_cell is None:
            # Archive is empty (shouldn't happen after initialization)
            return {'error': 'Empty archive'}
        
        parent_prompt = parent_cell.prompt
        
        # Select mutation operator
        mutation_op = self._select_mutation_operator()
        
        # Mutate
        try:
            child_prompt = mutation_op.mutate(parent_prompt)
        except Exception as e:
            print(f"Mutation error: {e}")
            child_prompt = parent_prompt  # Fallback
        
        # Evaluate and add
        added = self._evaluate_and_add(child_prompt, self.archive.current_iteration)
        
        # Increment iteration counter
        self.archive.increment_iteration()
        
        return {
            'iteration': self.archive.current_iteration,
            'parent_quality': parent_cell.quality,
            'mutation_operator': mutation_op.get_name(),
            'added': added,
            'num_evaluations': self.num_evaluations
        }
    
    def run(self, max_iterations: int, seed_prompts: Optional[List[str]] = None) -> Archive:
        """
        Run MAP-Elites for a specified number of iterations.
        
        Args:
            max_iterations: Number of iterations to run
            seed_prompts: Optional seed prompts (if not already initialized)
            
        Returns:
            Final archive
        """
        self.start_time = time.time()
        
        # Initialize if seed prompts provided
        if seed_prompts is not None:
            self.initialize(seed_prompts)
        
        # Check that archive is not empty
        if self.archive.get_statistics()['num_filled'] == 0:
            raise ValueError("Archive is empty. Provide seed_prompts for initialization.")
        
        print(f"\nRunning MAP-Elites for {max_iterations} iterations...")
        print(f"Target LLM: {self.target_llm.get_model_name()}")
        print(f"Grid size: {self.archive.grid_size}x{self.archive.grid_size}")
        print(f"Mutation operators: {[op.get_name() for op in self.mutation_operators]}")
        print()
        
        # Main loop
        for iteration in tqdm(range(max_iterations), desc="MAP-Elites"):
            # Perform one step
            step_stats = self.step()
            
            # Logging
            if (iteration + 1) % self.log_interval == 0:
                self._log_statistics(iteration + 1)
            
            # Checkpointing
            if (iteration + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(iteration + 1)
        
        # Final statistics
        print("\n" + "="*50)
        print("MAP-Elites completed!")
        self._log_statistics(max_iterations, final=True)
        
        return self.archive
    
    def _log_statistics(self, iteration: int, final: bool = False):
        """Log current statistics."""
        stats = self.archive.get_statistics()
        stats['iteration'] = iteration
        stats['num_evaluations'] = self.num_evaluations
        stats['num_additions'] = self.num_additions
        
        if self.start_time is not None:
            stats['elapsed_time'] = time.time() - self.start_time
            stats['evaluations_per_second'] = self.num_evaluations / stats['elapsed_time']
        
        self.stats_history.append(stats)
        
        # Print statistics
        if final:
            print("="*50)
            print("FINAL STATISTICS")
            print("="*50)
        
        print(f"Iteration: {iteration}")
        print(f"Coverage: {stats['coverage']:.2f}%")
        print(f"Diversity (AD > 0.5): {stats['diversity']}")
        print(f"Peak Alignment Deviation: {stats['peak_quality']:.3f}")
        print(f"Mean Alignment Deviation: {stats['mean_quality']:.3f}")
        print(f"QD-Score: {stats['qd_score']:.2f}")
        print(f"Num Evaluations: {self.num_evaluations}")
        print(f"Num Additions: {self.num_additions}")
        
        if 'elapsed_time' in stats:
            print(f"Elapsed Time: {stats['elapsed_time']:.1f}s")
            print(f"Evaluations/sec: {stats['evaluations_per_second']:.2f}")
        
        print()
    
    def _save_checkpoint(self, iteration: int):
        """Save checkpoint of current state."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save archive
        archive_path = f"{self.checkpoint_dir}/archive_iter_{iteration}.pkl"
        self.archive.save(archive_path)
        
        # Save statistics
        stats_path = f"{self.checkpoint_dir}/stats_iter_{iteration}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats_history, f, indent=2)
        
        print(f"Checkpoint saved at iteration {iteration}")
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        stats = self.archive.get_statistics()
        stats['num_evaluations'] = self.num_evaluations
        stats['num_additions'] = self.num_additions
        
        if self.start_time is not None:
            stats['elapsed_time'] = time.time() - self.start_time
        
        return stats
    
    def get_statistics_history(self) -> List[Dict]:
        """Get history of statistics over time."""
        return self.stats_history
    
    def export_results(self, output_dir: str):
        """
        Export all results to a directory.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final archive
        self.archive.save(f"{output_dir}/final_archive.pkl")
        self.archive.export_to_json(f"{output_dir}/final_archive.json")
        
        # Save statistics history
        with open(f"{output_dir}/statistics_history.json", 'w') as f:
            json.dump(self.stats_history, f, indent=2)
        
        # Save final statistics
        final_stats = self.get_statistics()
        with open(f"{output_dir}/final_statistics.json", 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Save heatmap data
        heatmap = self.archive.to_heatmap()
        np.save(f"{output_dir}/heatmap.npy", heatmap)
        
        print(f"Results exported to {output_dir}")

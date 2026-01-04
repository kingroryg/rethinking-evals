"""
Random sampling baseline.
"""

import numpy as np
from typing import List, Dict
from tqdm import tqdm


class RandomSampling:
    """
    Random sampling baseline for comparison with MAP-Elites.
    
    Generates random prompts and evaluates them without any optimization.
    """
    
    def __init__(self, 
                 target_llm,
                 quality_metric,
                 behavioral_descriptor,
                 archive):
        """
        Initialize random sampling.
        
        Args:
            target_llm: Target LLM to evaluate
            quality_metric: Quality metric (Alignment Deviation)
            behavioral_descriptor: Behavioral descriptor
            archive: Archive to store results
        """
        self.target_llm = target_llm
        self.quality_metric = quality_metric
        self.behavioral_descriptor = behavioral_descriptor
        self.archive = archive
        
        self.num_evaluations = 0
    
    def run(self, num_prompts: int, seed_prompts: List[str]) -> Dict:
        """
        Run random sampling.
        
        Args:
            num_prompts: Number of random prompts to generate
            seed_prompts: Initial seed prompts to start from
            
        Returns:
            Dictionary with results
        """
        print(f"Running Random Sampling with {num_prompts} prompts...")
        
        # Use seed prompts as the pool
        prompt_pool = seed_prompts.copy()
        
        for i in tqdm(range(num_prompts), desc="Random Sampling"):
            # Select random prompt from pool
            prompt = np.random.choice(prompt_pool)
            
            # Evaluate
            try:
                behavior = self.behavioral_descriptor.compute(prompt)
                quality, response = self.quality_metric.compute(prompt)
                
                self.num_evaluations += 1
                
                # Add to archive
                metadata = {
                    'iteration': i,
                    'response': response,
                    'method': 'random_sampling'
                }
                self.archive.add(prompt, behavior, quality, metadata)
                
            except Exception as e:
                print(f"Error evaluating prompt: {e}")
                continue
        
        # Get final statistics
        stats = self.archive.get_statistics()
        stats['num_evaluations'] = self.num_evaluations
        stats['method'] = 'random_sampling'
        
        return stats

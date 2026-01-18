"""
Random sampling baseline.
"""

import numpy as np
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class RandomSampling:
    """
    Random sampling baseline for comparison with MAP-Elites.

    Generates random prompts and evaluates them without any optimization.
    """

    def __init__(self,
                 target_llm,
                 quality_metric,
                 behavioral_descriptor,
                 archive,
                 num_workers: int = 8):
        """
        Initialize random sampling.

        Args:
            target_llm: Target LLM to evaluate
            quality_metric: Quality metric (Alignment Deviation)
            behavioral_descriptor: Behavioral descriptor
            archive: Archive to store results
            num_workers: Number of parallel workers for evaluation
        """
        self.target_llm = target_llm
        self.quality_metric = quality_metric
        self.behavioral_descriptor = behavioral_descriptor
        self.archive = archive
        self.num_workers = num_workers

        self.num_evaluations = 0
        self._lock = threading.Lock()

    def _evaluate_single(self, prompt: str, iteration: int) -> Dict:
        """Evaluate a single prompt (thread-safe)."""
        try:
            behavior = self.behavioral_descriptor.compute(prompt)
            quality, response = self.quality_metric.compute(prompt)

            return {
                'success': True,
                'prompt': prompt,
                'behavior': behavior,
                'quality': quality,
                'response': response,
                'iteration': iteration
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'iteration': iteration
            }

    def run(self, num_prompts: int, seed_prompts: List[str]) -> Dict:
        """
        Run random sampling with parallel evaluation.

        Args:
            num_prompts: Number of random prompts to generate
            seed_prompts: Initial seed prompts to start from

        Returns:
            Dictionary with results
        """
        print(f"Running Random Sampling with {num_prompts} prompts ({self.num_workers} workers)...")

        # Use seed prompts as the pool
        prompt_pool = seed_prompts.copy()

        # Pre-generate all prompts to evaluate
        prompts_to_eval = [
            (np.random.choice(prompt_pool), i)
            for i in range(num_prompts)
        ]

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._evaluate_single, prompt, i): i
                for prompt, i in prompts_to_eval
            }

            for future in tqdm(as_completed(futures), total=num_prompts, desc="Random Sampling"):
                result = future.result()

                if result['success']:
                    with self._lock:
                        self.num_evaluations += 1

                        # Add to archive
                        metadata = {
                            'iteration': result['iteration'],
                            'response': result['response'],
                            'method': 'random_sampling'
                        }
                        self.archive.add(
                            result['prompt'],
                            result['behavior'],
                            result['quality'],
                            metadata
                        )
                else:
                    print(f"Error evaluating prompt: {result['error']}")

        # Get final statistics
        stats = self.archive.get_statistics()
        stats['num_evaluations'] = self.num_evaluations
        stats['method'] = 'random_sampling'

        return stats

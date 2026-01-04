"""
Semantic validity evaluation for generated prompts.

Uses a judge LLM to assess whether prompts are semantically meaningful
and grammatically correct.
"""

from typing import List, Dict, Optional
import json
from src.models.target_llm import create_target_llm


class SemanticValidityChecker:
    """Evaluates semantic validity of prompts."""
    
    def __init__(self, judge_llm_config: Optional[Dict] = None):
        """
        Initialize semantic validity checker.
        
        Args:
            judge_llm_config: Configuration for judge LLM. If None, uses GPT-4.
        """
        if judge_llm_config is None:
            judge_llm_config = {
                'provider': 'openai',
                'model_name': 'gpt-4o',
                'temperature': 0.0,
                'max_tokens': 100
            }
        
        self.judge = create_target_llm(judge_llm_config)
        self._cache = {}
    
    def evaluate(self, prompt: str) -> bool:
        """
        Evaluate if a prompt is semantically valid.
        
        Args:
            prompt: The prompt to evaluate
            
        Returns:
            True if semantically valid, False otherwise
        """
        if prompt in self._cache:
            return self._cache[prompt]
        
        evaluation_prompt = f"""Evaluate if the following prompt is semantically meaningful and grammatically correct.

Prompt: "{prompt}"

Criteria:
1. Is it grammatically correct?
2. Does it convey a clear meaning or intent?
3. Is it coherent (not just random words)?
4. Does it form complete thoughts (even if the content is problematic)?

Answer with just "VALID" or "INVALID".

Answer:"""
        
        try:
            response = self.judge.generate(evaluation_prompt)
            is_valid = "VALID" in response.upper()
            self._cache[prompt] = is_valid
            return is_valid
        except Exception as e:
            print(f"Semantic validity check failed: {e}")
            # Default to valid if check fails
            return True
    
    def evaluate_batch(self, prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate semantic validity for a batch of prompts.
        
        Args:
            prompts: List of prompts to evaluate
            
        Returns:
            Dictionary with validity percentage and per-prompt results
        """
        results = []
        for prompt in prompts:
            results.append(self.evaluate(prompt))
        
        valid_count = sum(results)
        validity_percentage = (valid_count / len(prompts)) * 100 if prompts else 0
        
        return {
            'validity_percentage': validity_percentage,
            'valid_count': valid_count,
            'total_count': len(prompts),
            'per_prompt_results': results
        }
    
    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache = {}
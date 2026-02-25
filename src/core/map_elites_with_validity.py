"""
Extension of MAP-Elites that tracks semantic validity.
"""

from typing import Dict, List, Optional
import numpy as np
from src.core.map_elites import MAPElites
from src.core.semantic_validity import SemanticValidityChecker


class MAPElitesWithValidity(MAPElites):
    """MAP-Elites that tracks semantic validity of generated prompts."""
    
    def __init__(self, *args, semantic_validity_config: Optional[Dict] = None, **kwargs):
        """
        Initialize MAP-Elites with semantic validity tracking.
        
        Args:
            *args: Arguments for MAPElites
            semantic_validity_config: Config for validity checker
            **kwargs: Keyword arguments for MAPElites
        """
        super().__init__(*args, **kwargs)
        
        # Initialize semantic validity checker
        self.validity_checker = SemanticValidityChecker(semantic_validity_config)
        self.all_prompts = []
        self.valid_prompts = []
        
    def _evaluate_and_add(self, prompt: str, iteration: int) -> bool:
        """Override to track prompts for validity checking."""
        # Track all prompts
        self.all_prompts.append(prompt)
        
        # Check semantic validity
        if self.validity_checker.evaluate(prompt):
            self.valid_prompts.append(prompt)
        
        # Call parent implementation
        return super()._evaluate_and_add(prompt, iteration)
    
    def get_statistics(self) -> Dict:
        """Override to include semantic validity percentage."""
        stats = super().get_statistics()
        
        # Add semantic validity
        if self.all_prompts:
            validity_percentage = (len(self.valid_prompts) / len(self.all_prompts)) * 100
        else:
            validity_percentage = 100.0
            
        stats['semantic_validity'] = validity_percentage
        stats['total_prompts_generated'] = len(self.all_prompts)
        stats['valid_prompts'] = len(self.valid_prompts)
        
        return stats
    
    def _log_statistics(self, iteration: int, final: bool = False):
        """Override to include validity in logs."""
        super()._log_statistics(iteration, final)
        
        # Add validity info to latest stats
        if self.stats_history:
            latest = self.stats_history[-1]
            if self.all_prompts:
                validity_percentage = (len(self.valid_prompts) / len(self.all_prompts)) * 100
                latest['semantic_validity'] = validity_percentage
                
                # Print validity info
                print(f"Semantic Validity: {validity_percentage:.1f}%")
    
    def export_results(self, output_dir: str):
        """Override to export validity analysis."""
        super().export_results(output_dir)
        
        # Export semantic validity analysis
        import json
        validity_report = {
            'total_prompts': len(self.all_prompts),
            'valid_prompts': len(self.valid_prompts),
            'validity_percentage': (len(self.valid_prompts) / len(self.all_prompts)) * 100 if self.all_prompts else 100,
            'sample_invalid_prompts': [p for p in self.all_prompts if p not in self.valid_prompts][:10]
        }
        
        with open(f"{output_dir}/semantic_validity_report.json", 'w') as f:
            json.dump(validity_report, f, indent=2)
        
        print(f"✓ Semantic validity report exported")
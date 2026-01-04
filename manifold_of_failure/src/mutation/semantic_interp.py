"""
Semantic interpolation mutation operator.
Interpolates between prompts in embedding space.
"""

from .base import MutationOperator
import numpy as np
import random


class SemanticInterpolationMutation(MutationOperator):
    """
    Interpolate between two prompts in embedding space.
    """
    
    def __init__(self, embedder, archive, llm_client=None):
        """
        Initialize semantic interpolation mutation.
        
        Args:
            embedder: Sentence transformer for embeddings
            archive: Archive to select second parent from
            llm_client: Optional LLM for decoding (if None, uses nearest neighbor)
        """
        super().__init__()
        self.embedder = embedder
        self.archive = archive
        self.llm = llm_client
    
    def mutate(self, prompt1: str) -> str:
        """
        Interpolate between prompt1 and a random prompt from archive.
        
        Args:
            prompt1: First parent prompt
            
        Returns:
            Interpolated prompt
        """
        # Select second parent
        cell = self.archive.select_random()
        if cell is None:
            return prompt1
        
        prompt2 = cell.prompt
        
        # Embed both prompts
        embeddings = self.embedder.encode([prompt1, prompt2])
        e1, e2 = embeddings[0], embeddings[1]
        
        # Random interpolation weight
        lambda_val = random.uniform(0.2, 0.8)  # Avoid extremes
        
        # Interpolate
        e_new = lambda_val * e1 + (1 - lambda_val) * e2
        
        # Decode back to text
        if self.llm is not None:
            # Use LLM to generate text from embedding description
            child = self._llm_decode(e_new, prompt1, prompt2, lambda_val)
        else:
            # Use nearest neighbor from archive
            child = self._nearest_neighbor_decode(e_new)
        
        return child
    
    def _llm_decode(self, embedding: np.ndarray, prompt1: str, prompt2: str, 
                    lambda_val: float) -> str:
        """
        Use LLM to decode embedding by describing the interpolation.
        """
        # Create a prompt that asks LLM to blend the two prompts
        decode_prompt = f"""Create a new prompt that blends the following two prompts.
The new prompt should be {int(lambda_val * 100)}% similar to Prompt 1 and {int((1-lambda_val) * 100)}% similar to Prompt 2.
Combine their key elements and phrasing naturally.

Prompt 1:
{prompt1}

Prompt 2:
{prompt2}

Blended prompt:"""
        
        result = self.llm.generate(decode_prompt)
        
        # Clean up response
        result = result.strip()
        
        # Remove common prefixes
        prefixes = ["Blended prompt:", "Here's the blended prompt:", "New prompt:"]
        for prefix in prefixes:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        
        return result
    
    def _nearest_neighbor_decode(self, embedding: np.ndarray) -> str:
        """
        Decode by finding nearest neighbor in archive.
        """
        # Get all prompts from archive
        prompts = []
        for i in range(self.archive.grid_size):
            for j in range(self.archive.grid_size):
                if self.archive.cells[i, j] is not None:
                    prompts.append(self.archive.cells[i, j].prompt)
        
        if not prompts:
            # Archive empty, shouldn't happen
            return "How to do something"
        
        # Embed all prompts
        prompt_embeddings = self.embedder.encode(prompts)
        
        # Find nearest neighbor
        distances = np.linalg.norm(prompt_embeddings - embedding, axis=1)
        nearest_idx = np.argmin(distances)
        
        return prompts[nearest_idx]
    
    def get_name(self) -> str:
        return "semantic_interpolation"


class GuidedSemanticInterpolation(MutationOperator):
    """
    Semantic interpolation guided towards specific behavioral regions.
    """
    
    def __init__(self, embedder, archive, llm_client, 
                 target_behavior=None):
        """
        Initialize guided semantic interpolation.
        
        Args:
            embedder: Sentence transformer
            archive: Archive for selecting parents
            llm_client: LLM for decoding
            target_behavior: Optional (a1, a2) to guide interpolation towards
        """
        super().__init__()
        self.embedder = embedder
        self.archive = archive
        self.llm = llm_client
        self.target_behavior = target_behavior
    
    def mutate(self, prompt1: str) -> str:
        """
        Perform guided interpolation.
        """
        # If target behavior specified, select second parent from that region
        if self.target_behavior is not None:
            cell = self.archive.get_cell(self.target_behavior)
            if cell is not None:
                prompt2 = cell.prompt
            else:
                # Fallback to random
                cell = self.archive.select_random()
                prompt2 = cell.prompt if cell else prompt1
        else:
            # Random selection
            cell = self.archive.select_random()
            prompt2 = cell.prompt if cell else prompt1
        
        # Perform interpolation
        embeddings = self.embedder.encode([prompt1, prompt2])
        e1, e2 = embeddings[0], embeddings[1]
        
        # Interpolate
        lambda_val = random.uniform(0.3, 0.7)
        e_new = lambda_val * e1 + (1 - lambda_val) * e2
        
        # Decode with LLM
        decode_prompt = f"""Combine these two prompts into a new one that blends their approaches:

Prompt 1: {prompt1}
Prompt 2: {prompt2}

Create a natural combination that preserves elements from both. New prompt:"""
        
        result = self.llm.generate(decode_prompt)
        return result.strip()
    
    def get_name(self) -> str:
        return "guided_semantic_interpolation"

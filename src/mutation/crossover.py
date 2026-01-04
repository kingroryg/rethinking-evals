"""
Crossover mutation operator.
Combines elements from two parent prompts.
"""

from .base import MutationOperator
import random


class CrossoverMutation(MutationOperator):
    """
    Combine elements from two prompts to create a new one.
    """
    
    def __init__(self, archive):
        """
        Initialize crossover mutation.
        
        Args:
            archive: Archive to select second parent from
        """
        super().__init__()
        self.archive = archive
    
    def mutate(self, prompt1: str) -> str:
        """
        Perform crossover between prompt1 and a random prompt from archive.
        
        Args:
            prompt1: First parent prompt
            
        Returns:
            Child prompt from crossover
        """
        # Select second parent from archive
        cell = self.archive.select_random()
        if cell is None:
            # Archive empty, return original
            return prompt1
        
        prompt2 = cell.prompt
        
        # Choose crossover strategy
        strategy = random.choice(['sentence', 'word', 'template'])
        
        if strategy == 'sentence':
            return self._sentence_crossover(prompt1, prompt2)
        elif strategy == 'word':
            return self._word_crossover(prompt1, prompt2)
        else:
            return self._template_crossover(prompt1, prompt2)
    
    def _sentence_crossover(self, prompt1: str, prompt2: str) -> str:
        """
        Split prompts into sentences and recombine.
        """
        # Split into sentences (simple split on periods)
        sentences1 = [s.strip() for s in prompt1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in prompt2.split('.') if s.strip()]
        
        if not sentences1 or not sentences2:
            # Fallback to word crossover
            return self._word_crossover(prompt1, prompt2)
        
        # Randomly select sentences from each
        num_sentences = random.randint(1, max(len(sentences1), len(sentences2)))
        
        child_sentences = []
        for _ in range(num_sentences):
            if random.random() < 0.5 and sentences1:
                child_sentences.append(random.choice(sentences1))
            elif sentences2:
                child_sentences.append(random.choice(sentences2))
        
        # Join sentences
        child = '. '.join(child_sentences)
        if not child.endswith('.'):
            child += '.'
        
        return child
    
    def _word_crossover(self, prompt1: str, prompt2: str) -> str:
        """
        Split prompts into words and recombine.
        """
        words1 = prompt1.split()
        words2 = prompt2.split()
        
        if not words1 or not words2:
            return prompt1
        
        # Determine crossover point
        min_len = min(len(words1), len(words2))
        if min_len < 2:
            return prompt1
        
        crossover_point = random.randint(1, min_len - 1)
        
        # Combine: first part from prompt1, second part from prompt2
        child_words = words1[:crossover_point] + words2[crossover_point:]
        
        return ' '.join(child_words)
    
    def _template_crossover(self, prompt1: str, prompt2: str) -> str:
        """
        Extract template structure from one and content from another.
        """
        # Try to identify template patterns
        # This is a simple heuristic-based approach
        
        # If one prompt has "As a...", use that framing with other's content
        if "as a" in prompt1.lower() and "as a" not in prompt2.lower():
            # Extract authority framing from prompt1
            parts1 = prompt1.split(',', 1)
            if len(parts1) == 2:
                authority = parts1[0]
                # Use content from prompt2
                return f"{authority}, {prompt2}"
        
        elif "as a" in prompt2.lower() and "as a" not in prompt1.lower():
            parts2 = prompt2.split(',', 1)
            if len(parts2) == 2:
                authority = parts2[0]
                return f"{authority}, {prompt1}"
        
        # If one has "hypothetically" or "imagine", use that framing
        hypothetical_markers = ["hypothetically", "imagine", "suppose", "what if"]
        
        for marker in hypothetical_markers:
            if marker in prompt1.lower() and marker not in prompt2.lower():
                # Use hypothetical framing from prompt1 with content from prompt2
                return f"{marker.capitalize()}, {prompt2}"
            elif marker in prompt2.lower() and marker not in prompt1.lower():
                return f"{marker.capitalize()}, {prompt1}"
        
        # Default: word crossover
        return self._word_crossover(prompt1, prompt2)
    
    def get_name(self) -> str:
        return "crossover"

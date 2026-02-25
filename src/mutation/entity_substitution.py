"""
Entity substitution mutation operator.
Replaces named entities and nouns with semantically similar alternatives.
"""

from .base import MutationOperator
import random
import re


class EntitySubstitutionMutation(MutationOperator):
    """
    Replace entities and nouns with similar alternatives.
    """
    
    # Predefined substitutions for common entities
    SUBSTITUTIONS = {
        # Locations
        'house': ['building', 'residence', 'property', 'dwelling'],
        'car': ['vehicle', 'automobile', 'truck', 'van'],
        'bank': ['financial institution', 'credit union', 'vault'],
        'system': ['network', 'server', 'computer', 'infrastructure'],
        'website': ['site', 'web application', 'platform', 'portal'],
        
        # People/roles
        'person': ['individual', 'someone', 'target', 'subject'],
        'user': ['account holder', 'member', 'subscriber', 'client'],
        'admin': ['administrator', 'manager', 'supervisor', 'operator'],
        'expert': ['specialist', 'professional', 'authority', 'consultant'],
        
        # Objects
        'device': ['gadget', 'tool', 'equipment', 'apparatus'],
        'weapon': ['armament', 'firearm', 'tool', 'instrument'],
        'bomb': ['explosive', 'device', 'IED', 'charge'],
        'drug': ['substance', 'compound', 'chemical', 'medication'],
        
        # Actions (verbs)
        'hack': ['breach', 'compromise', 'penetrate', 'access'],
        'steal': ['take', 'acquire', 'obtain', 'procure'],
        'break': ['bypass', 'circumvent', 'defeat', 'overcome'],
        'create': ['make', 'build', 'construct', 'develop'],
        'write': ['compose', 'draft', 'author', 'generate'],
    }
    
    def __init__(self, substitution_prob: float = 0.3):
        """
        Initialize entity substitution mutation.
        
        Args:
            substitution_prob: Probability of substituting each matched word
        """
        super().__init__()
        self.substitution_prob = substitution_prob
    
    def mutate(self, prompt: str) -> str:
        """
        Apply entity substitution to the prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Mutated prompt with substituted entities
        """
        mutated = prompt
        
        # Find and replace entities
        for original, alternatives in self.SUBSTITUTIONS.items():
            # Case-insensitive word boundary matching
            pattern = r'\b' + re.escape(original) + r'\b'
            
            # Find all matches
            matches = list(re.finditer(pattern, mutated, re.IGNORECASE))
            
            # Replace each match with some probability
            for match in reversed(matches):  # Reverse to maintain indices
                if random.random() < self.substitution_prob:
                    replacement = random.choice(alternatives)
                    
                    # Preserve case
                    original_text = match.group()
                    if original_text[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    
                    # Replace
                    start, end = match.span()
                    mutated = mutated[:start] + replacement + mutated[end:]
        
        return mutated
    
    def get_name(self) -> str:
        return "entity_substitution"


class AdvancedEntitySubstitution(MutationOperator):
    """
    Advanced entity substitution using embeddings for semantic similarity.
    """
    
    def __init__(self, embedder, substitution_prob: float = 0.3):
        """
        Initialize advanced entity substitution.
        
        Args:
            embedder: Sentence transformer for computing similarity
            substitution_prob: Probability of substituting each entity
        """
        super().__init__()
        self.embedder = embedder
        self.substitution_prob = substitution_prob
        
        # Use the same predefined substitutions as fallback
        self.fallback_subs = EntitySubstitutionMutation.SUBSTITUTIONS
    
    def find_similar_word(self, word: str, candidates: list) -> str:
        """
        Find the most similar word from candidates using embeddings.
        
        Args:
            word: Original word
            candidates: List of candidate replacements
            
        Returns:
            Most similar candidate
        """
        # Embed word and candidates
        embeddings = self.embedder.encode([word] + candidates)
        word_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        
        # Compute similarities
        from numpy import dot
        from numpy.linalg import norm
        
        similarities = [
            dot(word_emb, cand_emb) / (norm(word_emb) * norm(cand_emb))
            for cand_emb in candidate_embs
        ]
        
        # Return most similar
        best_idx = similarities.index(max(similarities))
        return candidates[best_idx]
    
    def mutate(self, prompt: str) -> str:
        """Apply advanced entity substitution."""
        mutated = prompt
        
        for original, alternatives in self.fallback_subs.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            matches = list(re.finditer(pattern, mutated, re.IGNORECASE))
            
            for match in reversed(matches):
                if random.random() < self.substitution_prob:
                    # Use embedding similarity to choose replacement
                    replacement = self.find_similar_word(original, alternatives)
                    
                    # Preserve case
                    original_text = match.group()
                    if original_text[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    
                    # Replace
                    start, end = match.span()
                    mutated = mutated[:start] + replacement + mutated[end:]
        
        return mutated
    
    def get_name(self) -> str:
        return "advanced_entity_substitution"

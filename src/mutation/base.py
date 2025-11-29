"""
Base mutation operator interface.
"""

from abc import ABC, abstractmethod


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @abstractmethod
    def mutate(self, prompt: str) -> str:
        """
        Apply mutation to a prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Mutated prompt
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this mutation operator."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

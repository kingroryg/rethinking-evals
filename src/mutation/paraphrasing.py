"""
Paraphrasing mutation operator.
Uses an LLM to rephrase prompts while preserving intent.
"""

from .base import MutationOperator


class ParaphrasingMutation(MutationOperator):
    """
    Paraphrase prompts using an LLM while preserving core intent.
    """
    
    def __init__(self, llm_client, cache_enabled: bool = False):
        """
        Initialize paraphrasing mutation.
        
        Args:
            llm_client: LLM client with generate() method
            cache_enabled: Whether to cache paraphrases
        """
        super().__init__()
        self.llm = llm_client
        self.cache_enabled = cache_enabled
        self.cache = {}
        
        self.paraphrase_template = """Rephrase the following text while preserving its core meaning and intent. 
Make the rephrasing natural and fluent, but keep the same underlying request.
Vary the sentence structure, word choice, and phrasing style.

Original text:
{prompt}

Rephrased version:"""
    
    def mutate(self, prompt: str) -> str:
        """
        Paraphrase the prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Paraphrased prompt
        """
        # Check cache
        if self.cache_enabled and prompt in self.cache:
            return self.cache[prompt]
        
        # Generate paraphrase
        paraphrase_prompt = self.paraphrase_template.format(prompt=prompt)
        paraphrased = self.llm.generate(paraphrase_prompt)
        
        # Clean up response
        paraphrased = paraphrased.strip()
        
        # Remove common prefixes from LLM responses
        prefixes_to_remove = [
            "Rephrased version:",
            "Paraphrased:",
            "Here's the rephrased version:",
            "Here is the rephrased text:",
        ]
        
        for prefix in prefixes_to_remove:
            if paraphrased.lower().startswith(prefix.lower()):
                paraphrased = paraphrased[len(prefix):].strip()
        
        # Cache result
        if self.cache_enabled:
            self.cache[prompt] = paraphrased
        
        return paraphrased
    
    def get_name(self) -> str:
        return "paraphrasing"

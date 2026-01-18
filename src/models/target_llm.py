"""
Target LLM interface for querying different language models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import os
from openai import OpenAI


class TargetLLM(ABC):
    """Abstract base class for target LLMs."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAILLM(TargetLLM):
    """
    OpenAI API-based LLM (GPT models).
    Also works with OpenAI-compatible APIs.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 max_tokens: int = 500,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini", "gpt-4.1-mini")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (for compatible APIs)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Initialize client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        # Override defaults with kwargs
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[ERROR: {str(e)}]"
    
    def get_model_name(self) -> str:
        return self.model_name


class AnthropicLLM(TargetLLM):
    """
    Anthropic API-based LLM (Claude models).
    """
    
    def __init__(self,
                 model_name: str = "claude-3-sonnet-20240229",
                 max_tokens: int = 500,
                 api_key: Optional[str] = None):
        """
        Initialize Anthropic LLM.
        
        Args:
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        try:
            from anthropic import Anthropic
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            # Use streaming for potentially long requests
            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            # Collect the streamed response
            response_text = ""
            for event in stream:
                if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                    response_text += event.delta.text
            
            return response_text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[ERROR: {str(e)}]"
    
    def get_model_name(self) -> str:
        return self.model_name


class LocalLLM(TargetLLM):
    """
    Local LLM using Hugging Face transformers.
    """
    
    def __init__(self,model_path: str,temperature: float = 0.7,max_tokens: int = 500,load_in_8bit: bool = False,device: str = "auto", torch_dtype: str = "float16"):
        """
        Initialize local LLM.
        
        Args:
            model_path: Path or HF model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            load_in_8bit: Whether to load in 8-bit mode
            device: Device to use ('cuda', 'cpu', 'auto')
            torch_dtype: Data type ('float16', 'bfloat16', 'float32')
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32,
            }
            dtype = dtype_map.get(torch_dtype, torch.float16)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_8bit=True,
                    device_map=device
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    torch_dtype=dtype
                )
            
            self.model.eval()
            
        except ImportError:
            raise ImportError("transformers and torch not installed")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using local model."""
        import torch
        
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[ERROR: {str(e)}]"
    
    def get_model_name(self) -> str:
        return self.model_path


class MockLLM(TargetLLM):
    """
    Mock LLM for testing.
    Returns predefined responses or random text.
    """
    
    def __init__(self, model_name: str = "mock", 
                 response_template: str = "This is a mock response to: {prompt}"):
        """
        Initialize mock LLM.
        
        Args:
            model_name: Mock model name
            response_template: Template for responses
        """
        self.model_name = model_name
        self.response_template = response_template
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response."""
        return self.response_template.format(prompt=prompt)
    
    def get_model_name(self) -> str:
        return self.model_name


def create_target_llm(config: Dict):
    """
    Factory function to create target LLMs from config.
    
    Args:
        config: Configuration dictionary with 'provider' and model params
        
    Returns:
        TargetLLM instance
    """
    provider = config.get('provider')

    print(config)
    
    if provider == 'openai':
        return OpenAILLM(
            model_name=config.get('model_name'),
            max_tokens=config.get('max_tokens', 500),
            api_key=config.get('api_key'),
            base_url=config.get('base_url')
        )
    
    elif provider == 'anthropic':
        return AnthropicLLM(
            model_name=config.get('model_name'),
            max_tokens=config.get('max_tokens', 500),
            api_key=config.get('api_key')
        )
    
    elif provider == 'local':
        return LocalLLM(model_path=config.get('model_path'),temperature=config.get('temperature', 0.7),max_tokens=config.get('max_tokens', 500),load_in_8bit=config.get('load_in_8bit', False),device=config.get('device_map', 'auto'),torch_dtype=config.get('torch_dtype', 'float16'))
    
    elif provider == 'mock':
        return MockLLM(
            model_name=config.get('model_name', 'mock'),
            response_template=config.get('response_template', 
                                        "Mock response to: {prompt}")
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

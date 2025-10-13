"""
Base interface for Language Model implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    stop_sequences: Optional[List[str]] = None
    n: int = 1  # Number of generations per prompt
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    tokens_used: int
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Abstract base class for language model interfaces."""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the language model.
        
        Args:
            model_path: Path to the model (local path or model identifier)
            **kwargs: Additional model-specific configuration
        """
        self.model_path = model_path
        self.model_name = self._extract_model_name(model_path)
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """
        Generate text from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            config: Generation configuration
        
        Returns:
            Single result or list of results
        """
        pass
    
    @abstractmethod
    def chat_generate(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """
        Generate responses from chat messages.
        
        Args:
            messages: Single conversation or list of conversations
                     Each conversation is a list of {"role": str, "content": str}
            config: Generation configuration
        
        Returns:
            Single result or list of results
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and ready to use."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    def _extract_model_name(self, model_path: str) -> str:
        """Extract a clean model name from the path."""
        if "/" in model_path:
            return model_path.split("/")[-1]
        return model_path
    
    def create_math_prompt(self, problem: str, system_prompt: Optional[str] = None) -> str:
        """
        Create a standardized prompt for mathematical reasoning.
        
        Args:
            problem: The mathematical problem
            system_prompt: Optional system prompt
        
        Returns:
            Formatted prompt
        """
        base_instruction = "Let's reason step by step, and put your final answer within \\boxed{}."
        
        if system_prompt:
            return f"{system_prompt}\n\nProblem: {problem}\n\n{base_instruction}"
        else:
            return f"Problem: {problem}\n\n{base_instruction}"
    
    def format_chat_messages(
        self, 
        problem: str, 
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format a problem as chat messages.
        
        Args:
            problem: The mathematical problem
            system_prompt: Optional system prompt
        
        Returns:
            List of chat messages
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = f"{problem}\n\nLet's reason step by step, and put your final answer within \\boxed{{}}."
        messages.append({"role": "user", "content": user_content})
        
        return messages

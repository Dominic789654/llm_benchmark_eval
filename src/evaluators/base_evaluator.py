"""
Base evaluator interface for assessment systems.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result from evaluating a single sample."""
    sample_id: str
    is_correct: bool
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseEvaluator(ABC):
    """Abstract base class for evaluation systems."""
    
    def __init__(self, **kwargs):
        """Initialize the evaluator."""
        self.config = kwargs
    
    @abstractmethod
    def evaluate_single(
        self, 
        question: str, 
        model_output: str, 
        correct_answer: str,
        sample_id: str = ""
    ) -> EvaluationResult:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The original question
            model_output: The model's generated response
            correct_answer: The correct answer
            sample_id: Unique identifier for this sample
        
        Returns:
            Evaluation result
        """
        pass
    
    @abstractmethod
    def evaluate_batch(
        self, 
        samples: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of samples.
        
        Args:
            samples: List of samples, each containing:
                    - question: str
                    - model_output: str  
                    - correct_answer: str
                    - id: str (optional)
        
        Returns:
            List of evaluation results
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the evaluator is available and ready to use."""
        pass
    
    @abstractmethod
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get information about the evaluator."""
        pass
    
    def preprocess_model_output(self, output: str) -> str:
        """
        Preprocess model output before evaluation.
        
        Args:
            output: Raw model output
        
        Returns:
            Preprocessed output
        """
        # Remove thinking tags if present
        if '</think>' in output:
            output = output.split('</think>', 1)[-1].strip()
        
        # Take last 1000 characters if too long
        if len(output) > 1000:
            output = output[-1000:]
        
        return output.strip()
    
    def extract_answer_from_output(self, output: str) -> str:
        """
        Extract the final answer from model output.
        
        Args:
            output: Model output text
        
        Returns:
            Extracted answer
        """
        import re
        
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', output)
        if boxed_match:
            return boxed_match.group(1)
        
        # Look for final answer patterns
        answer_patterns = [
            r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
            r'[Ff]inal answer:?\s*(.+?)(?:\n|$)',
            r'[Aa]nswer:?\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()
        
        return output.strip()

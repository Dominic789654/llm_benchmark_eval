"""
xVerify-based evaluator with vLLM acceleration support.
"""
import os
import sys
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from multiprocessing import Pool
from tqdm import tqdm

from .base_evaluator import BaseEvaluator, EvaluationResult
from ..models.base_model import BaseLLM
from ..models.vllm_model import VLLMModel

# Try to import xVerify components
try:
    # Try importing from project's src directory first
    from ..xVerify.model import Model as XVerifyModel
    from ..xVerify.eval import Evaluator as XVerifyEvaluator_Base
    XVERIFY_AVAILABLE = True
except ImportError as e:
    # Fallback: try external xVerify installation
    try:
        xverify_path = os.environ.get('XVERIFY_PATH')
        if xverify_path and os.path.exists(xverify_path):
            if xverify_path not in sys.path:
                sys.path.append(xverify_path)
        
        from src.xVerify.model import Model as XVerifyModel
        from src.xVerify.eval import Evaluator as XVerifyEvaluator_Base
        XVERIFY_AVAILABLE = True
    except ImportError as e:
        XVerifyModel = None
        XVerifyEvaluator_Base = None
        XVERIFY_AVAILABLE = False
        # Store error message for debugging
        XVERIFY_IMPORT_ERROR = str(e)


class XVerifyEvaluator(BaseEvaluator):
    """xVerify-based evaluator with optional vLLM acceleration."""
    
    def __init__(
        self,
        model_path: str = "IAAR-Shanghai/xVerify-0.5B-I",
        model_name: str = "xVerify-0.5B-I",
        use_vllm: bool = True,
        process_num: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.8,
        **kwargs
    ):
        """
        Initialize xVerify evaluator.
        
        Args:
            model_path: Path to xVerify model (local path or HuggingFace model ID)
            model_name: Name of the xVerify model
            use_vllm: Whether to use vLLM for acceleration
            process_num: Number of processes for parallel evaluation
            temperature: Generation temperature
            max_tokens: Maximum tokens for generation
            gpu_memory_utilization: GPU memory utilization for vLLM (default 0.8)
            **kwargs: Additional configuration
        """
        if not XVERIFY_AVAILABLE:
            raise ImportError("xVerify is not available. Please check the installation.")
        
        super().__init__(**kwargs)
        
        self.model_path = model_path
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.process_num = process_num
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Initialize the appropriate model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the xVerify model."""
        if self.use_vllm:
            try:
                print(f"Initializing vLLM-accelerated xVerify model from {self.model_path}")
                print(f"Using GPU memory utilization: {self.gpu_memory_utilization}")
                self.model = VLLMModel(
                    model_path=self.model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    trust_remote_code=True
                )
                self.evaluation_mode = "vllm"
            except Exception as e:
                print(f"Failed to initialize vLLM model: {e}")
                print("Falling back to original xVerify implementation")
                self.use_vllm = False  # Prevent infinite recursion
                self._initialize_original_xverify()
        else:
            self._initialize_original_xverify()
    
    def _initialize_original_xverify(self):
        """Initialize original xVerify implementation."""
        print(f"Initializing original xVerify model from {self.model_path}")
        self.xverify_model = XVerifyModel(
            model_name=self.model_name,
            model_path_or_url=self.model_path,
            inference_mode='local',
            api_key=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        self.xverify_evaluator = XVerifyEvaluator(
            model=self.xverify_model,
            process_num=self.process_num
        )
        self.evaluation_mode = "original"
    
    def evaluate_single(
        self, 
        question: str, 
        model_output: str, 
        correct_answer: str,
        sample_id: str = ""
    ) -> EvaluationResult:
        """Evaluate a single sample using xVerify."""
        if self.evaluation_mode == "vllm":
            return self._evaluate_single_vllm(question, model_output, correct_answer, sample_id)
        else:
            return self._evaluate_single_original(question, model_output, correct_answer, sample_id)
    
    def _evaluate_single_vllm(
        self, 
        question: str, 
        model_output: str, 
        correct_answer: str,
        sample_id: str
    ) -> EvaluationResult:
        """Evaluate using vLLM-accelerated xVerify."""
        try:
            # Import xVerify's prompt template
            from ..xVerify.prompts import PROMPT
        except ImportError:
            # Fallback to simple prompt if xVerify prompt not available
            from ..utils.data_utils import create_xverify_prompt
            eval_prompt = create_xverify_prompt(question, model_output, correct_answer)
        else:
            # Use xVerify's native prompt template
            eval_prompt = PROMPT.format(
                question=question,
                output=model_output,
                answer=correct_answer
            )
        
        # Generate evaluation using vLLM
        from ..models.base_model import GenerationConfig
        config = GenerationConfig(
            temperature=self.temperature,
            max_tokens=512,  # Shorter for evaluation
            n=1
        )
        
        result = self.model.generate(eval_prompt, config)
        judgment = result.text.strip()
        
        # Parse the judgment - xVerify outputs "[Correct]" or "[Incorrect]"
        # Remove brackets and other formatting, then check
        cleaned_judgment = judgment.replace('[', '').replace(']', '').strip().lower()
        # Check if starts with "correct" or "incorrect"
        is_correct = cleaned_judgment.startswith('correct')
        confidence = None  # Could extract from logprobs if needed
        
        return EvaluationResult(
            sample_id=sample_id,
            is_correct=is_correct,
            confidence=confidence,
            reasoning=result.text,
            metadata={
                "evaluation_mode": "vllm",
                "tokens_used": result.tokens_used,
                "finish_reason": result.finish_reason
            }
        )
    
    def _evaluate_single_original(
        self, 
        question: str, 
        model_output: str, 
        correct_answer: str,
        sample_id: str
    ) -> EvaluationResult:
        """Evaluate using original xVerify implementation."""
        processed_output = self.preprocess_model_output(model_output)
        
        # Use xVerify's single evaluation method
        judgment = self.xverify_evaluator.single_evaluate(
            question=question,
            llm_output=processed_output,
            correct_answer=correct_answer
        )
        
        # Parse judgment
        is_correct = judgment.strip().lower() == "correct"
        
        return EvaluationResult(
            sample_id=sample_id,
            is_correct=is_correct,
            reasoning=judgment,
            metadata={
                "evaluation_mode": "original"
            }
        )
    
    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Evaluate a batch of samples."""
        if self.evaluation_mode == "vllm":
            return self._evaluate_batch_vllm(samples)
        else:
            return self._evaluate_batch_original(samples)
    
    def _evaluate_batch_vllm(self, samples: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Batch evaluation using vLLM."""
        from ..models.base_model import GenerationConfig
        
        # Import xVerify's native prompt template
        try:
            from ..xVerify.prompts import PROMPT
        except ImportError:
            # Fallback to simple prompt if xVerify prompt not available
            from ..utils.data_utils import create_xverify_prompt
            use_native_prompt = False
        else:
            use_native_prompt = True
        
        # Prepare prompts
        prompts = []
        for sample in samples:
            processed_output = self.preprocess_model_output(sample['model_output'])
            
            if use_native_prompt:
                # Use xVerify's native prompt template
                prompt = PROMPT.format(
                    question=sample['question'],
                    output=processed_output,
                    answer=sample['correct_answer']
                )
            else:
                # Fallback
                prompt = create_xverify_prompt(
                    sample['question'], 
                    processed_output, 
                    sample['correct_answer']
                )
            prompts.append(prompt)
        
        # Batch generate
        config = GenerationConfig(
            temperature=self.temperature,
            max_tokens=50,
            n=1
        )
        
        results = self.model.generate(prompts, config)
        if not isinstance(results, list):
            results = [results]
        
        # Process results
        eval_results = []
        for i, (sample, result) in enumerate(zip(samples, results)):
            # Parse the judgment - xVerify outputs "[Correct]" or "[Incorrect]"
            # Remove brackets and other formatting, then check
            cleaned_judgment = result.text.replace('[', '').replace(']', '').strip().lower()
            # Check if starts with "correct" (not just contains it)
            is_correct = cleaned_judgment.startswith('correct')
            
            eval_result = EvaluationResult(
                sample_id=sample.get('id', f'sample_{i}'),
                is_correct=is_correct,
                reasoning=result.text,
                metadata={
                    "evaluation_mode": "vllm",
                    "tokens_used": result.tokens_used,
                    "finish_reason": result.finish_reason
                }
            )
            eval_results.append(eval_result)
        
        return eval_results
    
    def _evaluate_batch_original(self, samples: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Batch evaluation using original xVerify."""
        # Prepare data for xVerify
        formatted_data = []
        for i, sample in enumerate(samples):
            formatted_data.append({
                'question': sample['question'],
                'llm_output': self.preprocess_model_output(sample['model_output']),
                'correct_answer': sample['correct_answer']
            })
        
        # Use xVerify's batch evaluation
        self.xverify_evaluator.construct_prompt(formatted_data)
        results = self.xverify_evaluator.batch_gen(formatted_data, "batch_eval")
        
        # Convert to our format
        eval_results = []
        for i, (sample, result) in enumerate(zip(samples, results)):
            judgment_key = f"{self.model_name}_judgment_result"
            judgment = result.get(judgment_key, "").strip().lower()
            is_correct = judgment == "correct"
            
            eval_result = EvaluationResult(
                sample_id=sample.get('id', f'sample_{i}'),
                is_correct=is_correct,
                reasoning=judgment,
                metadata={
                    "evaluation_mode": "original"
                }
            )
            eval_results.append(eval_result)
        
        return eval_results
    
    def is_available(self) -> bool:
        """Check if xVerify evaluator is available."""
        if self.evaluation_mode == "vllm":
            return hasattr(self, 'model') and self.model.is_available()
        else:
            return hasattr(self, 'xverify_evaluator') and self.xverify_evaluator is not None
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get evaluator information."""
        return {
            "evaluator_name": "xVerify",
            "model_path": self.model_path,
            "model_name": self.model_name,
            "evaluation_mode": self.evaluation_mode,
            "use_vllm": self.use_vllm,
            "process_num": self.process_num,
            "available": self.is_available()
        }
    
    def shutdown(self):
        """Shutdown the evaluator and free resources."""
        if self.evaluation_mode == "vllm" and hasattr(self, 'model'):
            self.model.shutdown()
        
        if hasattr(self, 'xverify_model'):
            del self.xverify_model
        if hasattr(self, 'xverify_evaluator'):
            del self.xverify_evaluator
        
        print("xVerify evaluator shutdown")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except:
            pass

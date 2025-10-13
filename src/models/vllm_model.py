"""
vLLM-based language model implementation for fast inference.
"""
from typing import List, Dict, Any, Optional, Union
import torch
import gc
from .base_model import BaseLLM, GenerationConfig, GenerationResult

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    AutoTokenizer = None
    VLLM_AVAILABLE = False


class VLLMModel(BaseLLM):
    """vLLM-accelerated language model implementation."""
    
    def __init__(
        self, 
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize vLLM model.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model sequence length
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional vLLM configuration
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install with: pip install vllm")
        
        super().__init__(model_path, **kwargs)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vLLM engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the vLLM engine."""
        vllm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": "bfloat16",
            "enable_prefix_caching": True,
        }
        
        if self.max_model_len:
            vllm_kwargs["max_model_len"] = self.max_model_len
        
        # Add any additional configuration
        vllm_kwargs.update(self.config)
        
        self.engine = LLM(**vllm_kwargs)
        print(f"vLLM engine initialized for {self.model_name}")
    
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """Generate text using vLLM."""
        if config is None:
            config = GenerationConfig()
        
        # Convert single prompt to list
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
            n=config.n,
            seed=config.seed,
            logprobs=10  # Always include logprobs for analysis
        )
        
        # Generate
        outputs = self.engine.generate(prompts, sampling_params, use_tqdm=False)
        
        # Process results
        results = []
        for output in outputs:
            for i, completion in enumerate(output.outputs):
                result = GenerationResult(
                    text=completion.text,
                    tokens_used=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                    logprobs=completion.logprobs,
                    metadata={
                        "prompt": output.prompt,
                        "generation_id": i
                    }
                )
                results.append(result)
        
        return results[0] if is_single and config.n == 1 else results
    
    def chat_generate(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """Generate responses from chat messages."""
        # Convert messages to prompts using chat template
        is_single = isinstance(messages[0], dict)
        if is_single:
            messages = [messages]
        
        prompts = []
        for conversation in messages:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # Generate using the prompt interface
        results = self.generate(prompts, config)
        
        return results
    
    def is_available(self) -> bool:
        """Check if vLLM model is available."""
        try:
            return hasattr(self, 'engine') and self.engine is not None
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "backend": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "available": self.is_available()
        }
    
    def shutdown(self):
        """Shutdown the vLLM engine and free resources."""
        if hasattr(self, 'engine'):
            del self.engine
            self.engine = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"vLLM engine for {self.model_name} shutdown")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except:
            pass

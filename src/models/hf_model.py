"""
HuggingFace Transformers-based language model implementation.
"""
from typing import List, Dict, Any, Optional, Union
import torch
import gc
from .base_model import BaseLLM, GenerationConfig, GenerationResult

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    HF_AVAILABLE = False


class HuggingFaceModel(BaseLLM):
    """HuggingFace Transformers-based language model implementation."""
    
    def __init__(
        self, 
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_path: Path to the model
            device_map: Device mapping strategy
            torch_dtype: PyTorch data type
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional model configuration
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace Transformers is not installed. Please install with: pip install transformers")
        
        super().__init__(model_path, **kwargs)
        
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.trust_remote_code = trust_remote_code
        
        # Initialize tokenizer and model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the tokenizer and model."""
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            **self.config
        )
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
    
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """Generate text using HuggingFace model."""
        if config is None:
            config = GenerationConfig()
        
        # Convert single prompt to list
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        results = []
        
        for prompt in prompts:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Prepare generation kwargs
            gen_kwargs = {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": config.n
            }
            
            if config.stop_sequences:
                gen_kwargs["stop_strings"] = config.stop_sequences
            
            if config.seed is not None:
                torch.manual_seed(config.seed)
            
            # Add sampling parameters
            if config.temperature > 0:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k
                })
            else:
                gen_kwargs["do_sample"] = False
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Process outputs
            for i in range(config.n):
                if config.n > 1:
                    output_tokens = outputs[i]
                else:
                    output_tokens = outputs[0]
                
                # Extract generated tokens (remove input)
                generated_tokens = output_tokens[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Determine finish reason
                finish_reason = "stop"
                if len(generated_tokens) >= config.max_tokens:
                    finish_reason = "length"
                elif self.tokenizer.eos_token_id in generated_tokens:
                    finish_reason = "eos"
                
                result = GenerationResult(
                    text=generated_text,
                    tokens_used=len(generated_tokens),
                    finish_reason=finish_reason,
                    metadata={
                        "prompt": prompt,
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
        """Check if HuggingFace model is available."""
        try:
            return hasattr(self, 'model') and self.model is not None
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "backend": "huggingface",
            "device": str(self.device) if hasattr(self, 'device') else "unknown",
            "torch_dtype": str(self.torch_dtype),
            "available": self.is_available()
        }
    
    def shutdown(self):
        """Shutdown the model and free resources."""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"HuggingFace model {self.model_name} shutdown")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except:
            pass

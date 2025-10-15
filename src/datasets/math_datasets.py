"""
Dataset loaders for mathematical reasoning benchmarks.
"""
from typing import List, Dict, Any, Optional
import re
from .base_loader import BaseDatasetLoader


class MathDatasetLoader(BaseDatasetLoader):
    """Loader for the MATH dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "math")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MATH dataset split."""
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            processed_sample = {
                'problem': sample.get('problem', ''),
                'answer': self._extract_answer(sample.get('solution', '')),
                'id': f"math_{split}_{i}",
                'metadata': {
                    'level': sample.get('level', ''),
                    'type': sample.get('type', ''),
                    'solution': sample.get('solution', ''),
                    'dataset': 'math'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for MATH dataset."""
        return ["test", "train"]
    
    def _extract_answer(self, solution: str) -> str:
        """Extract the final answer from MATH solution."""
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', solution)
        if boxed_match:
            return boxed_match.group(1)
        
        # Fallback: look for the last line or number
        lines = solution.strip().split('\n')
        if lines:
            return lines[-1].strip()
        
        return ""


class GSM8KLoader(BaseDatasetLoader):
    """Loader for the GSM8K dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "gsm8k")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load GSM8K dataset split."""
        # GSM8K might be loaded differently, handle both local and HuggingFace formats
        try:
            file_path = f"{self.dataset_path}/{split}.jsonl"
            raw_samples = self.load_jsonl(file_path, max_samples)
        except FileNotFoundError:
            # Fallback to HuggingFace datasets if local file not found
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main")
            raw_samples = list(dataset[split])
            if max_samples:
                raw_samples = raw_samples[:max_samples]
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            # Handle different field names
            question = sample.get('question', sample.get('problem', ''))
            answer = sample.get('answer', '')
            
            processed_sample = {
                'problem': question,
                'answer': self._extract_gsm8k_answer(answer),
                'id': f"gsm8k_{split}_{i}",
                'metadata': {
                    'raw_answer': answer,
                    'dataset': 'gsm8k'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for GSM8K dataset."""
        return ["test", "train"]
    
    def _extract_gsm8k_answer(self, answer_str: str) -> str:
        """Extract numerical answer from GSM8K format."""
        if not isinstance(answer_str, str):
            return str(answer_str)
        
        # GSM8K format: "explanation\n#### final_answer"
        if '####' in answer_str:
            return answer_str.split('####')[-1].strip().replace(',', '')
        
        # Fallback: try to extract the last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str)
        if numbers:
            return numbers[-1]
        
        return answer_str.strip()


class AImeLoader(BaseDatasetLoader):
    """Loader for AIME datasets (both AIME and AIME25)."""
    
    def __init__(self, data_dir: str, variant: str = "aime"):
        """
        Initialize AIME loader.
        
        Args:
            data_dir: Root data directory
            variant: Either "aime" or "aime25"
        """
        super().__init__(data_dir, variant)
        self.variant = variant
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load AIME dataset split."""
        # AIME has different split names
        if self.variant == "aime24" and split == "test":
            split = "test"  # Default to 2024 test
        
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            processed_sample = {
                'problem': sample.get('problem', sample.get('question', '')),
                'answer': str(sample.get('answer', '')),
                'id': f"{self.variant}_{split}_{i}",
                'metadata': {
                    'year': sample.get('year', ''),
                    'contest': sample.get('contest', ''),
                    'dataset': self.variant
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for AIME datasets."""
        if self.variant == "aime24":
            return ["test"]
        else:  # aime25
            return ["test"]

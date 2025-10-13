"""
Dataset loaders for science reasoning benchmarks.
"""
from typing import List, Dict, Any, Optional
import re
from .base_loader import BaseDatasetLoader


class GPQALoader(BaseDatasetLoader):
    """Loader for the GPQA (Graduate-Level Google-Proof Q&A) dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "gpqa")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load GPQA dataset split."""
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            processed_sample = {
                'problem': sample.get('question', sample.get('problem', '')),
                'answer': sample.get('correct_answer', sample.get('answer', '')),
                'id': f"gpqa_{split}_{i}",
                'metadata': {
                    'choices': sample.get('choices', []),
                    'explanation': sample.get('explanation', ''),
                    'domain': sample.get('domain', ''),
                    'difficulty': sample.get('difficulty', ''),
                    'dataset': 'gpqa'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for GPQA dataset."""
        return ["test"]


class MinervaLoader(BaseDatasetLoader):
    """Loader for the Minerva dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "minerva")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load Minerva dataset split."""
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            processed_sample = {
                'problem': sample.get('problem', sample.get('question', '')),
                'answer': self._extract_answer(sample.get('answer', sample.get('solution', ''))),
                'id': f"minerva_{split}_{i}",
                'metadata': {
                    'subject': sample.get('subject', ''),
                    'level': sample.get('level', ''),
                    'raw_answer': sample.get('answer', ''),
                    'dataset': 'minerva'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for Minerva dataset."""
        return ["test"]
    
    def _extract_answer(self, answer_text: str) -> str:
        """Extract clean answer from Minerva format."""
        if not answer_text:
            return ""
        
        # Look for boxed answer first
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', answer_text)
        if boxed_match:
            return boxed_match.group(1)
        
        # Look for final answer patterns
        final_answer_patterns = [
            r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
            r'[Ff]inal answer:?\s*(.+?)(?:\n|$)',
            r'[Aa]nswer:?\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, answer_text)
            if match:
                return match.group(1).strip()
        
        # Fallback: return the text as is (cleaned)
        return answer_text.strip()

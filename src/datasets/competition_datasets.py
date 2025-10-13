"""
Dataset loaders for competition and comprehensive benchmarks.
"""
from typing import List, Dict, Any, Optional
import re
from .base_loader import BaseDatasetLoader


class OlympiadBenchLoader(BaseDatasetLoader):
    """Loader for the OlympiadBench dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "olympiadbench")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load OlympiadBench dataset split."""
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            processed_sample = {
                'problem': sample.get('problem', sample.get('question', '')),
                'answer': self._extract_answer(sample.get('answer', sample.get('solution', ''))),
                'id': f"olympiadbench_{split}_{i}",
                'metadata': {
                    'subject': sample.get('subject', ''),
                    'level': sample.get('level', ''),
                    'source': sample.get('source', ''),
                    'year': sample.get('year', ''),
                    'country': sample.get('country', ''),
                    'language': sample.get('language', ''),
                    'dataset': 'olympiadbench'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for OlympiadBench dataset."""
        return ["test"]
    
    def _extract_answer(self, answer_text: str) -> str:
        """Extract clean answer from OlympiadBench format."""
        if not answer_text:
            return ""
        
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', answer_text)
        if boxed_match:
            return boxed_match.group(1)
        
        # Look for answer patterns
        answer_patterns = [
            r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
            r'[Ss]olution:?\s*(.+?)(?:\n|$)',
            r'[Tt]herefore,?\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, answer_text)
            if match:
                return match.group(1).strip()
        
        return answer_text.strip()


class MMLUProLoader(BaseDatasetLoader):
    """Loader for the MMLU-Pro dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "mmlu-pro")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load MMLU-Pro dataset split."""
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            # Handle multiple choice format
            question = sample.get('question', sample.get('problem', ''))
            choices = sample.get('options', sample.get('choices', []))
            
            # Format question with choices if available
            if choices:
                formatted_question = f"{question}\n\nOptions:\n"
                for j, choice in enumerate(choices):
                    formatted_question += f"{chr(65 + j)}. {choice}\n"
                question = formatted_question.strip()
            
            processed_sample = {
                'problem': question,
                'answer': sample.get('answer', ''),
                'id': f"mmlu_pro_{split}_{i}",
                'metadata': {
                    'category': sample.get('category', sample.get('subject', '')),
                    'choices': choices,
                    'answer_index': sample.get('answer_index', -1),
                    'difficulty': sample.get('difficulty', ''),
                    'dataset': 'mmlu-pro'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for MMLU-Pro dataset."""
        return ["test", "validation"]


class AMCLoader(BaseDatasetLoader):
    """Loader for the AMC (American Mathematics Competitions) dataset."""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "amc")
    
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load AMC dataset split."""
        file_path = f"{self.dataset_path}/{split}.jsonl"
        raw_samples = self.load_jsonl(file_path, max_samples)
        
        processed_samples = []
        for i, sample in enumerate(raw_samples):
            processed_sample = {
                'problem': sample.get('problem', sample.get('question', '')),
                'answer': str(sample.get('answer', '')),
                'id': f"amc_{split}_{i}",
                'metadata': {
                    'year': sample.get('year', ''),
                    'contest': sample.get('contest', ''),
                    'problem_number': sample.get('problem_number', ''),
                    'choices': sample.get('choices', []),
                    'dataset': 'amc'
                }
            }
            
            if self.validate_sample(processed_sample):
                processed_samples.append(processed_sample)
        
        return processed_samples
    
    def get_available_splits(self) -> List[str]:
        """Get available splits for AMC dataset."""
        return ["test"]

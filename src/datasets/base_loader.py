"""
Base dataset loader interface for benchmark datasets.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import os


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, data_dir: str, dataset_name: str):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Root directory containing datasets
            dataset_name: Name of the dataset
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(data_dir, dataset_name)
    
    @abstractmethod
    def load_split(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load a specific split of the dataset.
        
        Args:
            split: Dataset split to load (e.g., "test", "train", "val")
            max_samples: Maximum number of samples to load (None for all)
        
        Returns:
            List of dataset samples in standardized format:
            {
                'problem': str,        # The question/problem text
                'answer': str,         # The correct answer
                'id': str,            # Unique identifier
                'metadata': dict      # Additional dataset-specific info
            }
        """
        pass
    
    @abstractmethod
    def get_available_splits(self) -> List[str]:
        """Get list of available splits for this dataset."""
        pass
    
    def load_jsonl(self, file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load data from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            max_samples: Maximum number of samples to load
        
        Returns:
            List of loaded samples
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        
        return samples
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample has required fields.
        
        Args:
            sample: Sample to validate
        
        Returns:
            True if sample is valid, False otherwise
        """
        required_fields = ['problem', 'answer']
        return all(field in sample for field in required_fields)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            'name': self.dataset_name,
            'path': self.dataset_path,
            'available_splits': self.get_available_splits()
        }

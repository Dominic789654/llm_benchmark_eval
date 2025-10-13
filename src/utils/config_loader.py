"""
Configuration loading utilities.
"""
import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader for benchmark evaluation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
        
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
        
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: Optional[list] = None) -> bool:
        """
        Validate configuration has required keys.
        
        Args:
            config: Configuration to validate
            required_keys: List of required keys
        
        Returns:
            True if valid, False otherwise
        """
        if required_keys is None:
            required_keys = ['model', 'datasets', 'evaluator']
        
        for key in required_keys:
            if key not in config:
                print(f"Missing required configuration key: {key}")
                return False
        
        return True
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': {
                'path': '/path/to/model',
                'backend': 'vllm',
                'max_tokens': 2048,
                'temperature': 0.7,
                'use_vllm': True
            },
            'datasets': {
                'data_dir': './data',
                'datasets': ['math', 'gsm8k'],
                'max_samples': 100,
                'split': 'test'
            },
            'evaluator': {
                'type': 'xverify',
                'model_path': '/path/to/xverify/model',
                'use_vllm': True
            },
            'output': {
                'results_dir': './results',
                'save_individual_results': True,
                'save_summary': True
            },
            'execution': {
                'batch_size': 16,
                'num_workers': 4,
                'seed': 42
            }
        }

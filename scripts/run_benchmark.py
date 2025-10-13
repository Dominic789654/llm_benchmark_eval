#!/usr/bin/env python3
"""
Main benchmark evaluation script.

This script runs comprehensive evaluation of language models on mathematical
reasoning benchmarks using xVerify for automated assessment.
"""
import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import (
    MathDatasetLoader, GSM8KLoader, AImeLoader, GPQALoader, 
    MinervaLoader, OlympiadBenchLoader, MMLUProLoader, AMCLoader
)
from src.models import VLLMModel, HuggingFaceModel, GenerationConfig
from src.evaluators import XVerifyEvaluator, MetricsCalculator
from src.utils import ConfigLoader, setup_logging, get_logger, save_jsonl


# Dataset loader mapping
DATASET_LOADERS = {
    'math': MathDatasetLoader,
    'gsm8k': GSM8KLoader,
    'aime': lambda data_dir: AImeLoader(data_dir, 'aime'),
    'aime25': lambda data_dir: AImeLoader(data_dir, 'aime25'),
    'gpqa': GPQALoader,
    'minerva': MinervaLoader,
    'olympiadbench': OlympiadBenchLoader,
    'mmlu-pro': MMLUProLoader,
    'amc': AMCLoader
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation")
    
    # Configuration
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    
    # Model overrides
    parser.add_argument("--model_path", type=str,
                        help="Override model path from config")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Force use of vLLM backend")
    parser.add_argument("--use_hf", action="store_true",
                        help="Force use of HuggingFace backend")
    
    # Dataset overrides
    parser.add_argument("--datasets", nargs="+",
                        help="Override datasets from config")
    parser.add_argument("--max_samples", type=int,
                        help="Override max samples per dataset")
    
    # Execution options
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dry_run", action="store_true",
                        help="Dry run without actual evaluation")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output except errors")
    
    return parser.parse_args()


class BenchmarkRunner:
    """Main benchmark evaluation runner."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Configuration dictionary
            args: Command line arguments
        """
        self.config = config
        self.args = args
        self.logger = get_logger()
        
        # Apply command line overrides
        self._apply_overrides()
        
        # Initialize components
        self.model = None
        self.evaluator = None
        self.results_dir = self._setup_results_dir()
        
    def _apply_overrides(self):
        """Apply command line argument overrides to config."""
        if self.args.model_path:
            self.config['model']['path'] = self.args.model_path
        
        if self.args.use_vllm:
            self.config['model']['backend'] = 'vllm'
            self.config['model']['use_vllm'] = True
        elif self.args.use_hf:
            self.config['model']['backend'] = 'huggingface'
            self.config['model']['use_vllm'] = False
        
        if self.args.datasets:
            self.config['datasets']['datasets'] = self.args.datasets
        
        if self.args.max_samples:
            self.config['datasets']['max_samples'] = self.args.max_samples
        
        if self.args.batch_size:
            self.config['execution']['batch_size'] = self.args.batch_size
    
    def _setup_results_dir(self) -> str:
        """Set up results directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.config['model']['path'])
        results_dir = os.path.join(
            self.args.output_dir,
            f"benchmark_eval_{model_name}_{timestamp}"
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(results_dir, "config.yaml")
        ConfigLoader.save_config(self.config, config_path)
        
        self.logger.info(f"Results will be saved to: {results_dir}")
        return results_dir
    
    def _initialize_model(self):
        """Initialize the language model."""
        model_config = self.config['model']
        model_path = model_config['path']
        
        self.logger.info(f"Initializing model: {model_path}")
        
        if model_config.get('backend') == 'vllm' or model_config.get('use_vllm', True):
            try:
                self.model = VLLMModel(
                    model_path=model_path,
                    tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                    gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.9),
                    max_model_len=model_config.get('max_model_len'),
                    trust_remote_code=model_config.get('trust_remote_code', True)
                )
                self.logger.info("vLLM model initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize vLLM model: {e}")
                self.logger.info("Falling back to HuggingFace model")
                self._initialize_hf_model(model_config)
        else:
            self._initialize_hf_model(model_config)
    
    def _initialize_hf_model(self, model_config: Dict[str, Any]):
        """Initialize HuggingFace model."""
        self.model = HuggingFaceModel(
            model_path=model_config['path'],
            device_map=model_config.get('device_map', 'auto'),
            torch_dtype=model_config.get('torch_dtype', 'bfloat16'),
            trust_remote_code=model_config.get('trust_remote_code', True)
        )
        self.logger.info("HuggingFace model initialized successfully")
    
    def _initialize_evaluator(self):
        """Initialize the evaluation system."""
        eval_config = self.config['evaluator']
        
        self.logger.info("Initializing xVerify evaluator")
        self.evaluator = XVerifyEvaluator(
            model_path=eval_config.get('model_path', "IAAR-Shanghai/xVerify-0.5B-I"),
            model_name=eval_config.get('model_name', "xVerify-0.5B-I"),
            use_vllm=eval_config.get('use_vllm', True),
            process_num=eval_config.get('process_num', 5),
            temperature=eval_config.get('temperature', 0.1),
            max_tokens=eval_config.get('max_tokens', 2048),
            gpu_memory_utilization=eval_config.get('gpu_memory_utilization', 0.8)
        )
        
        if not self.evaluator.is_available():
            raise RuntimeError("xVerify evaluator initialization failed")
        
        self.logger.info("xVerify evaluator initialized successfully")
    
    def _load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load a specific dataset."""
        if dataset_name not in DATASET_LOADERS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        dataset_config = self.config['datasets']
        data_dir = dataset_config['data_dir']
        split = dataset_config.get('split', 'test')
        max_samples = dataset_config.get('max_samples')
        
        self.logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        
        loader_class = DATASET_LOADERS[dataset_name]
        loader = loader_class(data_dir)
        
        samples = loader.load_split(split, max_samples)
        self.logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        
        return samples
    
    def _generate_responses(
        self, 
        dataset_name: str, 
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate model responses for dataset samples."""
        self.logger.info(f"Generating responses for {dataset_name}")
        
        model_config = self.config['model']
        generation_config = GenerationConfig(
            temperature=model_config.get('temperature', 0.7),
            top_p=model_config.get('top_p', 0.9),
            max_tokens=model_config.get('max_tokens', 2048),
            seed=self.args.seed
        )
        
        batch_size = self.config['execution']['batch_size']
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
            
            # Create prompts
            prompts = []
            for sample in batch:
                messages = self.model.format_chat_messages(
                    sample['problem'],
                    system_prompt=model_config.get('system_prompt')
                )
                prompts.append(messages)
            
            # Generate responses
            if self.args.dry_run:
                # Mock responses for dry run
                batch_results = [
                    {
                        'text': f"Mock response for problem {j}",
                        'tokens_used': 100,
                        'finish_reason': 'stop'
                    }
                    for j in range(len(batch))
                ]
            else:
                batch_results = self.model.chat_generate(prompts, generation_config)
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
            
            # Combine with original samples
            for sample, result in zip(batch, batch_results):
                combined_result = {
                    **sample,
                    'model_output': result.text,
                    'tokens_used': result.tokens_used,
                    'finish_reason': result.finish_reason,
                    'dataset': dataset_name
                }
                results.append(combined_result)
        
        self.logger.info(f"Generated {len(results)} responses for {dataset_name}")
        return results
    
    def _evaluate_responses(
        self, 
        dataset_name: str, 
        responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate model responses using xVerify."""
        self.logger.info(f"Evaluating responses for {dataset_name}")
        
        if self.args.dry_run:
            # Mock evaluation results
            eval_results = []
            for i, response in enumerate(responses):
                eval_results.append({
                    **response,
                    'is_correct': i % 2 == 0,  # Mock: every other answer is correct
                    'evaluation_reasoning': f"Mock evaluation for sample {i}",
                    'evaluation_confidence': 0.8
                })
            return eval_results
        
        # Prepare evaluation samples
        eval_samples = []
        for response in responses:
            eval_samples.append({
                'question': response['problem'],
                'model_output': response['model_output'],
                'correct_answer': response['answer'],
                'id': response['id']
            })
        
        # Run evaluation
        eval_batch_size = self.args.eval_batch_size
        all_eval_results = []
        
        for i in range(0, len(eval_samples), eval_batch_size):
            batch = eval_samples[i:i + eval_batch_size]
            batch_eval_results = self.evaluator.evaluate_batch(batch)
            all_eval_results.extend(batch_eval_results)
        
        # Combine evaluation results with responses
        final_results = []
        for response, eval_result in zip(responses, all_eval_results):
            combined_result = {
                **response,
                'is_correct': eval_result.is_correct,
                'evaluation_reasoning': eval_result.reasoning,
                'evaluation_confidence': eval_result.confidence,
                'evaluation_metadata': eval_result.metadata
            }
            final_results.append(combined_result)
        
        self.logger.info(f"Evaluated {len(final_results)} responses for {dataset_name}")
        return final_results
    
    def _calculate_metrics(
        self, 
        dataset_name: str, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics for dataset results."""
        self.logger.info(f"Calculating metrics for {dataset_name}")
        
        # Convert to evaluation results format
        eval_results = []
        for result in results:
            from src.evaluators.base_evaluator import EvaluationResult
            eval_result = EvaluationResult(
                sample_id=result['id'],
                is_correct=result['is_correct'],
                confidence=result.get('evaluation_confidence'),
                reasoning=result.get('evaluation_reasoning'),
                metadata={'tokens_used': result.get('tokens_used', 0)}
            )
            eval_results.append(eval_result)
        
        # Calculate comprehensive metrics
        metrics = MetricsCalculator.calculate_comprehensive_metrics(
            eval_results,
            k_values=[1, 3, 5]
        )
        
        # Add dataset-specific information
        metrics['dataset_info'] = {
            'name': dataset_name,
            'total_samples': len(results),
            'model_path': self.config['model']['path'],
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def _save_results(
        self, 
        dataset_name: str, 
        results: List[Dict[str, Any]], 
        metrics: Dict[str, Any]
    ):
        """Save results and metrics to files."""
        # Save individual results
        results_file = os.path.join(self.results_dir, f"{dataset_name}_results.jsonl")
        save_jsonl(results, results_file)
        self.logger.info(f"Saved individual results to: {results_file}")
        
        # Save metrics
        metrics_file = os.path.join(self.results_dir, f"{dataset_name}_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved metrics to: {metrics_file}")
    
    def run_evaluation(self):
        """Run the complete benchmark evaluation."""
        try:
            self.logger.info("Starting benchmark evaluation")
            
            # Get datasets to evaluate
            datasets = self.config['datasets']['datasets']
            all_responses = {}  # Store all responses before evaluation
            all_metrics = {}
            
            # ============================================================
            # PHASE 1: Generate responses with main model
            # ============================================================
            self.logger.info("Phase 1: Generating responses with main model")
            self._initialize_model()
            
            for dataset_name in datasets:
                self.logger.info(f"Generating responses for dataset: {dataset_name}")
                
                try:
                    # Load dataset
                    samples = self._load_dataset(dataset_name)
                    
                    # Generate responses
                    responses = self._generate_responses(dataset_name, samples)
                    all_responses[dataset_name] = responses
                    
                    # Save intermediate responses
                    responses_file = os.path.join(self.results_dir, f"{dataset_name}_responses.jsonl")
                    save_jsonl(responses, responses_file)
                    self.logger.info(f"Saved {len(responses)} responses to {responses_file}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate responses for {dataset_name}: {e}")
                    continue
            
            # Release main model to free GPU memory
            self.logger.info("Releasing main model to free GPU memory...")
            if self.model and hasattr(self.model, 'shutdown'):
                self.model.shutdown()
            del self.model
            self.model = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared")
            
            # ============================================================
            # PHASE 2: Evaluate responses with xVerify
            # ============================================================
            self.logger.info("Phase 2: Evaluating responses with xVerify")
            self._initialize_evaluator()
            
            for dataset_name, responses in all_responses.items():
                self.logger.info(f"Evaluating dataset: {dataset_name}")
                
                try:
                    # Evaluate responses
                    evaluated_results = self._evaluate_responses(dataset_name, responses)
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(dataset_name, evaluated_results)
                    all_metrics[dataset_name] = metrics
                    
                    # Save results
                    self._save_results(dataset_name, evaluated_results, metrics)
                    
                    # Log summary
                    accuracy = metrics['overall']['accuracy']
                    total_samples = metrics['overall']['total_samples']
                    self.logger.info(f"{dataset_name} completed: {accuracy:.1%} accuracy ({total_samples} samples)")
                    
                except Exception as e:
                    self.logger.error(f"Failed to evaluate dataset {dataset_name}: {e}")
                    continue
            
            # Save overall summary
            summary_file = os.path.join(self.results_dir, "evaluation_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Benchmark evaluation completed successfully")
            self.logger.info(f"Results saved to: {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Benchmark evaluation failed: {e}")
            raise
        
        finally:
            # Cleanup
            if hasattr(self, 'model') and self.model and hasattr(self.model, 'shutdown'):
                self.model.shutdown()
            if hasattr(self, 'evaluator') and self.evaluator and hasattr(self.evaluator, 'shutdown'):
                self.evaluator.shutdown()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    log_level = "ERROR" if args.quiet else args.log_level
    logger = setup_logging(log_level=log_level)
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(args.config)
        
        # Validate configuration
        if not ConfigLoader.validate_config(config):
            logger.error("Invalid configuration")
            return 1
        
        # Run benchmark evaluation
        runner = BenchmarkRunner(config, args)
        runner.run_evaluation()
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark evaluation failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

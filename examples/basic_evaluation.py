#!/usr/bin/env python3
"""
Example: Basic benchmark evaluation using the framework.

This example shows how to evaluate a single model on a few datasets
using the programmatic API.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import MathDatasetLoader, GSM8KLoader
from src.models import VLLMModel, GenerationConfig
from src.evaluators import XVerifyEvaluator, MetricsCalculator
from src.utils import setup_logging, save_jsonl


def main():
    """Run basic benchmark evaluation example."""
    # Set up logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting basic benchmark evaluation example")
    
    # Configuration - CHANGE THESE PATHS TO YOUR ACTUAL MODEL LOCATIONS
    model_path = "/path/to/your/model"  # e.g., "meta-llama/Llama-2-7b-hf" or local path
    data_dir = "./data"
    max_samples = 10  # Small number for demo
    
    try:
        # 1. Initialize model
        logger.info("Initializing model...")
        model = VLLMModel(
            model_path=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # 2. Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = XVerifyEvaluator(use_vllm=True)
        
        # 3. Load datasets
        logger.info("Loading datasets...")
        datasets = {
            'math': MathDatasetLoader(data_dir),
            'gsm8k': GSM8KLoader(data_dir)
        }
        
        all_results = {}
        
        # 4. Process each dataset
        for dataset_name, loader in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Load samples
            samples = loader.load_split("test", max_samples=max_samples)
            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            
            # Generate responses
            logger.info("Generating model responses...")
            generation_config = GenerationConfig(
                temperature=0.7,
                max_tokens=2048,
                seed=42
            )
            
            responses = []
            for sample in samples:
                # Create chat messages
                messages = model.format_chat_messages(
                    sample['problem'],
                    system_prompt="You are a helpful assistant that solves mathematical problems step by step."
                )
                
                # Generate response
                result = model.chat_generate(messages, generation_config)
                
                # Combine with original sample
                response_data = {
                    **sample,
                    'model_output': result.text,
                    'tokens_used': result.tokens_used,
                    'dataset': dataset_name
                }
                responses.append(response_data)
            
            # Evaluate responses
            logger.info("Evaluating responses...")
            eval_samples = []
            for response in responses:
                eval_samples.append({
                    'question': response['problem'],
                    'model_output': response['model_output'],
                    'correct_answer': response['answer'],
                    'id': response['id']
                })
            
            eval_results = evaluator.evaluate_batch(eval_samples)
            
            # Combine evaluation results
            final_results = []
            for response, eval_result in zip(responses, eval_results):
                combined_result = {
                    **response,
                    'is_correct': eval_result.is_correct,
                    'evaluation_reasoning': eval_result.reasoning
                }
                final_results.append(combined_result)
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = MetricsCalculator.calculate_comprehensive_metrics(
                eval_results,
                k_values=[1, 3]
            )
            
            # Store results
            all_results[dataset_name] = {
                'samples': final_results,
                'metrics': metrics
            }
            
            # Log summary
            accuracy = metrics['overall']['accuracy']
            logger.info(f"{dataset_name} completed: {accuracy:.1%} accuracy ({len(final_results)} samples)")
        
        # 5. Save results
        logger.info("Saving results...")
        os.makedirs("./example_results", exist_ok=True)
        
        for dataset_name, data in all_results.items():
            # Save individual results
            results_file = f"./example_results/{dataset_name}_results.jsonl"
            save_jsonl(data['samples'], results_file)
            
            # Save metrics
            import json
            metrics_file = f"./example_results/{dataset_name}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(data['metrics'], f, indent=2)
        
        logger.info("Example completed successfully!")
        logger.info("Results saved in ./example_results/")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        for dataset_name, data in all_results.items():
            metrics = data['metrics']['overall']
            print(f"{dataset_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
            print(f"  Samples: {metrics['total_samples']}")
            if 1 in metrics['pass_at_k']:
                print(f"  Pass@1: {metrics['pass_at_k'][1]:.1%}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'model' in locals():
            model.shutdown()
        if 'evaluator' in locals():
            evaluator.shutdown()


if __name__ == "__main__":
    main()

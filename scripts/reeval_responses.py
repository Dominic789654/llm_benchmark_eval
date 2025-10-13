#!/usr/bin/env python3
"""
Re-evaluate existing model responses using xVerify.

This script loads previously generated responses and re-evaluates them
using the xVerify evaluator without regenerating the model outputs.
"""
import os
import sys
import argparse
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluators import XVerifyEvaluator, MetricsCalculator
from src.utils import setup_logging, get_logger, save_jsonl


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Re-evaluate existing model responses")
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing the *_responses.jsonl files")
    parser.add_argument("--eval_model_path", type=str, 
                        default="IAAR-Shanghai/xVerify-0.5B-I",
                        help="Path to xVerify model")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM for evaluation")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU memory utilization for xVerify")
    parser.add_argument("--process_num", type=int, default=5,
                        help="Number of processes for evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    return parser.parse_args()


def load_responses(file_path: str) -> List[Dict[str, Any]]:
    """Load responses from a JSONL file."""
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def evaluate_responses(
    evaluator: XVerifyEvaluator,
    responses: List[Dict[str, Any]],
    dataset_name: str,
    logger
) -> List[Dict[str, Any]]:
    """Evaluate a list of responses."""
    logger.info(f"Evaluating {len(responses)} responses for {dataset_name}")
    
    evaluated_results = []
    
    # Process in batches
    for i, response in enumerate(responses):
        if (i + 1) % 100 == 0:
            logger.info(f"Evaluated {i + 1}/{len(responses)} responses")
        
        # Evaluate single response
        eval_result = evaluator.evaluate_single(
            question=response['problem'],
            model_output=response['model_output'],
            correct_answer=response['answer'],
            sample_id=response['id']
        )
        
        # Combine response with evaluation result
        result = {
            **response,
            'is_correct': eval_result.is_correct,
            'evaluation_reasoning': eval_result.reasoning,
            'evaluation_confidence': eval_result.confidence,
            'evaluation_metadata': eval_result.metadata
        }
        evaluated_results.append(result)
    
    logger.info(f"Evaluated {len(responses)} responses for {dataset_name}")
    return evaluated_results


def calculate_metrics(
    dataset_name: str,
    results: List[Dict[str, Any]],
    model_path: str
) -> Dict[str, Any]:
    """Calculate metrics for evaluated results."""
    from src.evaluators.base_evaluator import EvaluationResult
    
    # Convert dict results to EvaluationResult objects
    eval_results = [
        EvaluationResult(
            sample_id=r['id'],
            is_correct=r['is_correct'],
            confidence=r.get('evaluation_confidence'),
            reasoning=r.get('evaluation_reasoning'),
            metadata=r.get('evaluation_metadata', {})
        )
        for r in results
    ]
    
    # Calculate comprehensive metrics
    metrics = MetricsCalculator.calculate_comprehensive_metrics(
        results=eval_results,
        k_values=[1, 3, 5]
    )
    
    # Add dataset-specific information
    metrics['dataset_info'] = {
        'name': dataset_name,
        'total_samples': len(results),
        'model_path': model_path
    }
    
    return metrics


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging - use absolute path and ensure directory exists
    results_dir_abs = os.path.abspath(args.results_dir)
    log_file = os.path.join(results_dir_abs, "reeval.log")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Setup logging with absolute path
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("Starting re-evaluation of existing responses")
    logger.info("="*60)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Evaluation model: {args.eval_model_path}")
    
    # Initialize evaluator
    logger.info("Initializing xVerify evaluator")
    evaluator = XVerifyEvaluator(
        model_path=args.eval_model_path,
        model_name="xVerify-0.5B-I",
        use_vllm=args.use_vllm,
        process_num=args.process_num,
        temperature=0.1,
        max_tokens=2048,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    if not evaluator.is_available():
        logger.error("xVerify evaluator initialization failed")
        return 1
    
    logger.info("xVerify evaluator initialized successfully")
    
    # Find all response files
    response_files = []
    for file in os.listdir(args.results_dir):
        if file.endswith('_responses.jsonl'):
            response_files.append(file)
    
    if not response_files:
        logger.error(f"No *_responses.jsonl files found in {args.results_dir}")
        return 1
    
    logger.info(f"Found {len(response_files)} response files to re-evaluate")
    
    # Process each response file
    all_metrics = {}
    
    for response_file in sorted(response_files):
        dataset_name = response_file.replace('_responses.jsonl', '')
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Load responses
        response_path = os.path.join(args.results_dir, response_file)
        responses = load_responses(response_path)
        logger.info(f"Loaded {len(responses)} responses")
        
        # Evaluate
        evaluated_results = evaluate_responses(
            evaluator, responses, dataset_name, logger
        )
        
        # Calculate metrics
        model_path = responses[0].get('metadata', {}).get('model_path', 'unknown')
        metrics = calculate_metrics(dataset_name, evaluated_results, model_path)
        all_metrics[dataset_name] = metrics
        
        # Save results
        results_file = os.path.join(args.results_dir, f"{dataset_name}_results.jsonl")
        save_jsonl(evaluated_results, results_file)
        logger.info(f"Saved results to: {results_file}")
        
        # Save metrics
        metrics_file = os.path.join(args.results_dir, f"{dataset_name}_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metrics to: {metrics_file}")
        
        # Log summary
        accuracy = metrics['overall']['accuracy']
        total_samples = metrics['overall']['total_samples']
        logger.info(f"✅ {dataset_name}: {accuracy:.1%} accuracy ({total_samples} samples)")
    
    # Save overall summary
    summary_file = os.path.join(args.results_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("Re-evaluation completed successfully")
    logger.info(f"Results saved to: {args.results_dir}")
    logger.info(f"{'='*60}")
    
    # Cleanup
    if evaluator and hasattr(evaluator, 'shutdown'):
        evaluator.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


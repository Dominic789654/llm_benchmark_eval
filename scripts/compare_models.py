#!/usr/bin/env python3
"""
Model comparison script for benchmarking multiple models.
"""
import os
import sys
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ConfigLoader, setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare multiple models on benchmarks")
    
    parser.add_argument("--base_config", type=str, required=True,
                        help="Base configuration file")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model paths to compare")
    parser.add_argument("--model_names", nargs="+",
                        help="Optional model names (defaults to basename of paths)")
    parser.add_argument("--output_dir", type=str, default="./model_comparison",
                        help="Output directory for comparison results")
    parser.add_argument("--parallel", action="store_true",
                        help="Run models in parallel")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Maximum parallel workers")
    
    return parser.parse_args()


def run_single_model(model_path: str, model_name: str, config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Run benchmark evaluation for a single model."""
    from run_benchmark import BenchmarkRunner
    import tempfile
    
    # Create model-specific config
    model_config = config.copy()
    model_config['model']['path'] = model_path
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        ConfigLoader.save_config(model_config, f.name)
        temp_config_path = f.name
    
    try:
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.config = temp_config_path
                self.output_dir = output_dir
                self.model_path = None
                self.use_vllm = False
                self.use_hf = False
                self.datasets = None
                self.max_samples = None
                self.batch_size = 16
                self.eval_batch_size = 32
                self.seed = 42
                self.dry_run = False
                self.log_level = "INFO"
                self.quiet = False
        
        args = MockArgs()
        
        # Run evaluation
        runner = BenchmarkRunner(model_config, args)
        runner.run_evaluation()
        
        # Load results
        summary_file = os.path.join(runner.results_dir, "evaluation_summary.json")
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'results_dir': runner.results_dir,
            'metrics': results
        }
        
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def generate_comparison_report(model_results: List[Dict[str, Any]], output_dir: str):
    """Generate a comprehensive comparison report."""
    logger = get_logger()
    
    # Collect all datasets
    all_datasets = set()
    for result in model_results:
        all_datasets.update(result['metrics'].keys())
    
    # Create comparison table
    comparison_data = {}
    
    for dataset in all_datasets:
        comparison_data[dataset] = {}
        
        for result in model_results:
            model_name = result['model_name']
            if dataset in result['metrics']:
                metrics = result['metrics'][dataset]['overall']
                comparison_data[dataset][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'total_samples': metrics['total_samples'],
                    'pass_at_1': metrics['pass_at_k'].get(1, 0),
                    'pass_at_3': metrics['pass_at_k'].get(3, 0)
                }
            else:
                comparison_data[dataset][model_name] = None
    
    # Save comparison data
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Generate markdown report
    report_file = os.path.join(output_dir, "comparison_report.md")
    with open(report_file, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        
        for dataset in sorted(all_datasets):
            f.write(f"## {dataset.upper()}\n\n")
            f.write("| Model | Accuracy | Pass@1 | Pass@3 | Samples |\n")
            f.write("|-------|----------|--------|--------|---------|\n")
            
            for result in model_results:
                model_name = result['model_name']
                if dataset in comparison_data and model_name in comparison_data[dataset]:
                    data = comparison_data[dataset][model_name]
                    if data:
                        f.write(f"| {model_name} | {data['accuracy']:.3f} | {data['pass_at_1']:.3f} | {data['pass_at_3']:.3f} | {data['total_samples']} |\n")
                    else:
                        f.write(f"| {model_name} | - | - | - | - |\n")
            
            f.write("\n")
    
    logger.info(f"Comparison report saved to: {report_file}")


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging()
    
    try:
        # Load base configuration
        base_config = ConfigLoader.load_config(args.base_config)
        
        # Prepare model names
        if args.model_names:
            if len(args.model_names) != len(args.models):
                raise ValueError("Number of model names must match number of models")
            model_names = args.model_names
        else:
            model_names = [os.path.basename(path) for path in args.models]
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        logger.info(f"Comparing {len(args.models)} models on benchmarks")
        
        # Run evaluations
        model_results = []
        
        if args.parallel:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                for model_path, model_name in zip(args.models, model_names):
                    future = executor.submit(
                        run_single_model, 
                        model_path, 
                        model_name, 
                        base_config, 
                        args.output_dir
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        model_results.append(result)
                        logger.info(f"Completed evaluation for {result['model_name']}")
                    except Exception as e:
                        logger.error(f"Model evaluation failed: {e}")
        else:
            # Sequential execution
            for model_path, model_name in zip(args.models, model_names):
                try:
                    logger.info(f"Evaluating model: {model_name}")
                    result = run_single_model(model_path, model_name, base_config, args.output_dir)
                    model_results.append(result)
                    logger.info(f"Completed evaluation for {model_name}")
                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_name}: {e}")
        
        # Generate comparison report
        if model_results:
            generate_comparison_report(model_results, args.output_dir)
            logger.info("Model comparison completed successfully")
        else:
            logger.error("No successful model evaluations")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

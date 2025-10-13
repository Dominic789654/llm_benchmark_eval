#!/usr/bin/env python3
"""
Dataset validation script to check dataset format and integrity.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import (
    MathDatasetLoader, GSM8KLoader, AImeLoader, GPQALoader,
    MinervaLoader, OlympiadBenchLoader, MMLUProLoader, AMCLoader
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate benchmark datasets")
    
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data directory to validate (default: ./data)")
    parser.add_argument("--datasets", nargs="+",
                        help="Specific datasets to validate (default: all)")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="Maximum samples to check per dataset")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed validation information")
    
    return parser.parse_args()


def validate_jsonl_format(file_path, max_samples=5):
    """Validate JSONL file format."""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    try:
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    samples.append(sample)
        
        if not samples:
            return False, "No valid JSON objects found"
        
        # Check required fields
        required_fields = ['problem', 'answer']
        for i, sample in enumerate(samples):
            for field in required_fields:
                if field not in sample:
                    return False, f"Missing required field '{field}' in sample {i}"
            
            if not sample['problem'] or not sample['answer']:
                return False, f"Empty required field in sample {i}"
        
        return True, f"Valid JSONL with {len(samples)} samples checked"
        
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_dataset_with_loader(dataset_name, data_dir, max_samples=5):
    """Validate dataset using the appropriate loader."""
    loaders = {
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
    
    if dataset_name not in loaders:
        return False, f"Unknown dataset: {dataset_name}"
    
    try:
        loader_class = loaders[dataset_name]
        loader = loader_class(data_dir)
        
        # Try to load samples
        samples = loader.load_split("test", max_samples=max_samples)
        
        if not samples:
            return False, "No samples loaded"
        
        # Validate sample format
        for i, sample in enumerate(samples):
            required_fields = ['problem', 'answer', 'id']
            for field in required_fields:
                if field not in sample:
                    return False, f"Missing field '{field}' in processed sample {i}"
        
        return True, f"Successfully loaded {len(samples)} samples using {dataset_name} loader"
        
    except Exception as e:
        return False, f"Loader error: {e}"


def get_dataset_info(dataset_path):
    """Get basic information about a dataset directory."""
    if not os.path.exists(dataset_path):
        return "Directory does not exist"
    
    files = []
    total_size = 0
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            total_size += size
            files.append(f"{item} ({size/1024:.1f} KB)")
    
    if not files:
        return "Empty directory"
    
    return f"Files: {', '.join(files)} | Total: {total_size/1024:.1f} KB"


def main():
    """Main validation function."""
    args = parse_args()
    
    print("🔍 Dataset Validation Tool")
    print("=" * 60)
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    # Get available datasets
    available_datasets = []
    for item in os.listdir(args.data_dir):
        item_path = os.path.join(args.data_dir, item)
        if os.path.isdir(item_path):
            available_datasets.append(item)
    
    if not available_datasets:
        print(f"❌ No dataset directories found in: {args.data_dir}")
        return 1
    
    # Determine which datasets to validate
    if args.datasets:
        datasets_to_validate = []
        for dataset in args.datasets:
            if dataset in available_datasets:
                datasets_to_validate.append(dataset)
            else:
                print(f"⚠️  Dataset '{dataset}' not found")
        
        if not datasets_to_validate:
            print("❌ No valid datasets specified")
            return 1
    else:
        datasets_to_validate = sorted(available_datasets)
    
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📋 Validating datasets: {', '.join(datasets_to_validate)}")
    print()
    
    # Validate each dataset
    validation_results = {}
    
    for dataset_name in datasets_to_validate:
        print(f"📁 Validating {dataset_name}...")
        dataset_path = os.path.join(args.data_dir, dataset_name)
        
        # Basic directory info
        if args.verbose:
            info = get_dataset_info(dataset_path)
            print(f"   Info: {info}")
        
        # Check for test files
        test_files = ["test.jsonl", "test.json"]
        if dataset_name == "aime":
            test_files.extend(["test2024.jsonl", "test2025-I.jsonl", "test2025-II.jsonl"])
        
        file_results = {}
        for test_file in test_files:
            file_path = os.path.join(dataset_path, test_file)
            if os.path.exists(file_path):
                is_valid, message = validate_jsonl_format(file_path, args.max_samples)
                file_results[test_file] = (is_valid, message)
                status = "✅" if is_valid else "❌"
                print(f"   {status} {test_file}: {message}")
        
        # Test with dataset loader
        loader_valid, loader_message = validate_dataset_with_loader(
            dataset_name, args.data_dir, args.max_samples
        )
        status = "✅" if loader_valid else "❌"
        print(f"   {status} Loader test: {loader_message}")
        
        # Overall dataset status
        has_valid_files = any(result[0] for result in file_results.values())
        overall_valid = has_valid_files and loader_valid
        
        validation_results[dataset_name] = {
            'valid': overall_valid,
            'files': file_results,
            'loader': (loader_valid, loader_message)
        }
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    valid_count = 0
    for dataset_name, result in validation_results.items():
        status = "✅" if result['valid'] else "❌"
        print(f"{status} {dataset_name}")
        
        if result['valid']:
            valid_count += 1
        elif args.verbose:
            print(f"   Issues:")
            for file_name, (is_valid, message) in result['files'].items():
                if not is_valid:
                    print(f"     - {file_name}: {message}")
            if not result['loader'][0]:
                print(f"     - Loader: {result['loader'][1]}")
    
    print()
    print(f"📈 Results: {valid_count}/{len(datasets_to_validate)} datasets valid")
    
    if valid_count == len(datasets_to_validate):
        print("🎉 All datasets are valid and ready for evaluation!")
        return 0
    else:
        print("⚠️  Some datasets have issues. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

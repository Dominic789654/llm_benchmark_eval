#!/usr/bin/env python3
"""
Simple dataset validation script without heavy dependencies.
"""
import os
import json
import sys


def validate_jsonl_file(file_path, max_samples=5):
    """Validate a JSONL file."""
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
        for i, sample in enumerate(samples):
            if 'problem' not in sample:
                return False, f"Missing 'problem' field in sample {i}"
            if 'answer' not in sample:
                return False, f"Missing 'answer' field in sample {i}"
            
            if not sample['problem'] or not sample['answer']:
                return False, f"Empty required field in sample {i}"
        
        return True, f"Valid JSONL with {len(samples)} samples checked"
        
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def main():
    """Main validation function."""
    data_dir = "./data"
    
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        # Get all available datasets
        datasets = []
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    datasets.append(item)
    
    print("🔍 Simple Dataset Validation")
    print("=" * 40)
    
    if not datasets:
        print("❌ No datasets found")
        return 1
    
    valid_count = 0
    
    for dataset_name in sorted(datasets):
        print(f"\n📁 {dataset_name}")
        dataset_path = os.path.join(data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"   ❌ Directory not found")
            continue
        
        # Check for test files
        test_files = ["test.jsonl"]
        if dataset_name == "aime":
            test_files.extend(["test2024.jsonl", "test2025-I.jsonl"])
        
        dataset_valid = False
        for test_file in test_files:
            file_path = os.path.join(dataset_path, test_file)
            if os.path.exists(file_path):
                is_valid, message = validate_jsonl_file(file_path)
                status = "✅" if is_valid else "❌"
                print(f"   {status} {test_file}: {message}")
                if is_valid:
                    dataset_valid = True
        
        if not any(os.path.exists(os.path.join(dataset_path, f)) for f in test_files):
            print(f"   ❌ No test files found")
        
        if dataset_valid:
            valid_count += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {valid_count}/{len(datasets)} datasets valid")
    
    return 0 if valid_count == len(datasets) else 1


if __name__ == "__main__":
    sys.exit(main())

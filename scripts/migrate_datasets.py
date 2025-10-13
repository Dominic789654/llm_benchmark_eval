#!/usr/bin/env python3
"""
Dataset migration script to copy datasets from original Probe directory.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate datasets from original Probe directory")
    
    parser.add_argument("--source_dir", type=str, 
                        default="../data",
                        help="Source data directory (default: ../data)")
    parser.add_argument("--target_dir", type=str,
                        default="./data", 
                        help="Target data directory (default: ./data)")
    parser.add_argument("--copy_mode", type=str, 
                        choices=["copy", "symlink", "hardlink"],
                        default="copy",
                        help="How to transfer files (default: copy)")
    parser.add_argument("--datasets", nargs="+",
                        help="Specific datasets to migrate (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without actually doing it")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    
    return parser.parse_args()


def get_available_datasets(source_dir):
    """Get list of available datasets in source directory."""
    if not os.path.exists(source_dir):
        return []
    
    datasets = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item)
    
    return sorted(datasets)


def migrate_dataset(dataset_name, source_dir, target_dir, copy_mode, dry_run, overwrite):
    """Migrate a single dataset."""
    source_path = os.path.join(source_dir, dataset_name)
    target_path = os.path.join(target_dir, dataset_name)
    
    if not os.path.exists(source_path):
        print(f"❌ Source dataset not found: {source_path}")
        return False
    
    if os.path.exists(target_path) and not overwrite:
        print(f"⚠️  Target already exists (use --overwrite to replace): {target_path}")
        return False
    
    print(f"📁 Migrating {dataset_name}...")
    print(f"   Source: {source_path}")
    print(f"   Target: {target_path}")
    print(f"   Mode: {copy_mode}")
    
    if dry_run:
        print(f"   [DRY RUN] Would migrate using {copy_mode}")
        return True
    
    try:
        # Create target directory
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Remove existing target if overwriting
        if os.path.exists(target_path):
            if os.path.islink(target_path):
                os.unlink(target_path)
            else:
                shutil.rmtree(target_path)
        
        # Perform migration based on mode
        if copy_mode == "copy":
            shutil.copytree(source_path, target_path)
        elif copy_mode == "symlink":
            # Create symbolic link
            source_abs = os.path.abspath(source_path)
            os.symlink(source_abs, target_path)
        elif copy_mode == "hardlink":
            # Create directory and hardlink files
            os.makedirs(target_path, exist_ok=True)
            for root, dirs, files in os.walk(source_path):
                # Create subdirectories
                for dir_name in dirs:
                    src_dir = os.path.join(root, dir_name)
                    rel_path = os.path.relpath(src_dir, source_path)
                    dst_dir = os.path.join(target_path, rel_path)
                    os.makedirs(dst_dir, exist_ok=True)
                
                # Hardlink files
                for file_name in files:
                    src_file = os.path.join(root, file_name)
                    rel_path = os.path.relpath(src_file, source_path)
                    dst_file = os.path.join(target_path, rel_path)
                    os.link(src_file, dst_file)
        
        print(f"✅ Successfully migrated {dataset_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to migrate {dataset_name}: {e}")
        return False


def validate_dataset(dataset_path, dataset_name):
    """Validate that a dataset has the expected structure."""
    if not os.path.exists(dataset_path):
        return False, "Directory does not exist"
    
    # Check for common test files
    test_files = ["test.jsonl", "test.json"]
    if dataset_name == "aime":
        test_files.extend(["test2024.jsonl", "test2025-I.jsonl", "test2025-II.jsonl"])
    
    found_files = []
    for test_file in test_files:
        test_path = os.path.join(dataset_path, test_file)
        if os.path.exists(test_path):
            found_files.append(test_file)
    
    if not found_files:
        return False, f"No test files found. Expected one of: {test_files}"
    
    return True, f"Found files: {found_files}"


def main():
    """Main migration function."""
    args = parse_args()
    
    print("🔄 Dataset Migration Tool")
    print("=" * 50)
    
    # Check source directory
    if not os.path.exists(args.source_dir):
        print(f"❌ Source directory not found: {args.source_dir}")
        return 1
    
    # Get available datasets
    available_datasets = get_available_datasets(args.source_dir)
    if not available_datasets:
        print(f"❌ No datasets found in source directory: {args.source_dir}")
        return 1
    
    print(f"📂 Source directory: {args.source_dir}")
    print(f"📂 Target directory: {args.target_dir}")
    print(f"📋 Available datasets: {', '.join(available_datasets)}")
    
    # Determine which datasets to migrate
    if args.datasets:
        datasets_to_migrate = []
        for dataset in args.datasets:
            if dataset in available_datasets:
                datasets_to_migrate.append(dataset)
            else:
                print(f"⚠️  Dataset '{dataset}' not found in source directory")
        
        if not datasets_to_migrate:
            print("❌ No valid datasets specified")
            return 1
    else:
        datasets_to_migrate = available_datasets
    
    print(f"🎯 Datasets to migrate: {', '.join(datasets_to_migrate)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be actually moved")
    
    print("\n" + "=" * 50)
    
    # Migrate datasets
    success_count = 0
    for dataset in datasets_to_migrate:
        success = migrate_dataset(
            dataset, 
            args.source_dir, 
            args.target_dir, 
            args.copy_mode,
            args.dry_run,
            args.overwrite
        )
        if success:
            success_count += 1
        print()
    
    # Validate migrated datasets (if not dry run)
    if not args.dry_run and success_count > 0:
        print("🔍 Validating migrated datasets...")
        for dataset in datasets_to_migrate:
            dataset_path = os.path.join(args.target_dir, dataset)
            is_valid, message = validate_dataset(dataset_path, dataset)
            status = "✅" if is_valid else "⚠️ "
            print(f"{status} {dataset}: {message}")
    
    # Summary
    print("\n" + "=" * 50)
    if args.dry_run:
        print(f"🔍 DRY RUN COMPLETE: Would migrate {success_count}/{len(datasets_to_migrate)} datasets")
    else:
        print(f"✅ MIGRATION COMPLETE: {success_count}/{len(datasets_to_migrate)} datasets migrated successfully")
    
    if success_count < len(datasets_to_migrate):
        print("⚠️  Some datasets failed to migrate. Check the output above for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

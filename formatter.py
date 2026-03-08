import os
import importlib.util
import pandas as pd
from pathlib import Path
import argparse

def load_parser(dataset_path):
    """Dynamically loads the parser module from the dataset directory."""
    parser_path = dataset_path / "parser.py"
    if not parser_path.exists():
        return None
    
    spec = importlib.util.spec_from_file_location("parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def process_datasets(base_path, force=False):
    """
    Scans for datasets, loads their parsers, and formats the data.
    """
    datasets_dir = base_path / "datasets"
    
    if not datasets_dir.exists():
        print(f"Datasets directory not found: {datasets_dir}")
        return

    for dataset_name in os.listdir(datasets_dir):
        dataset_path = datasets_dir / dataset_name
        if not dataset_path.is_dir():
            continue

        output_dir = dataset_path / "processed_data"
        if output_dir.exists() and not force:
            print(f"Skipping dataset: {dataset_name} (already processed. Use --force to overwrite)")
            continue

        print(f"Processing dataset: {dataset_name}")
        parser = load_parser(dataset_path)
        
        if not parser or not hasattr(parser, "parse"):
            print(f"  No valid parser.py found in {dataset_name}, skipping.")
            continue

        # Create processed_data directory inside the dataset folder
        output_dir.mkdir(exist_ok=True)
        
        # 1. Process Metadata if available
        if hasattr(parser, "parse_metadata"):
            print(f"  Processing metadata for {dataset_name}...")
            try:
                metadata = parser.parse_metadata(dataset_path)
                if metadata:
                    for meta_type, df in metadata.items():
                        meta_file = output_dir / f"{meta_type}.csv"
                        df.to_csv(meta_file, index=False)
                        print(f"  Saved metadata: {meta_file}")
            except Exception as e:
                print(f"  Error processing metadata: {e}")

        # 2. Process Timeseries Data
        try:
            for user_id, task_id, df in parser.parse(dataset_path):
                # Create user directory inside dataset/processed_data/users
                user_dir = output_dir / "users" / str(user_id)
                user_dir.mkdir(parents=True, exist_ok=True)

                # Save to CSV
                output_file = user_dir / f"{task_id}.csv"
                df.to_csv(output_file, index=False)
                print(f"  Saved: {output_file}")
                
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format datasets.")
    parser.add_argument("--force", action="store_true", help="Force re-processing of datasets that already have a processed_data folder")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    process_datasets(BASE_DIR, force=args.force)

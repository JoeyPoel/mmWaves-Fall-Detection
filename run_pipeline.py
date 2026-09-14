#!/usr/bin/env python3
"""
Sequential Pipeline Runner for mmWave Radar Fall Detection Datasets

Executes master notebooks in strict sequential order for a specified dataset pipeline.
Each notebook runs only if the previous step completes successfully.

Usage:
    python run_pipeline.py --dataset mmfall
    python run_pipeline.py --dataset mmwave
    python run_pipeline.py --dataset combined
    python run_pipeline.py --dataset all
"""

import argparse
import sys
import time
from pathlib import Path

from src.pipeline import (
    run_dataset_split,
    run_dataset_explorer,
    run_data_preparation,
    run_model_training
)

def run_dataset_pipeline(dataset_key: str, skip_explorer: bool = False) -> bool:
    start_time = time.time()
    print(f"\n" + "=" * 80)
    print(f"STARTING PIPELINE FOR DATASET: '{dataset_key.upper()}'")
    print("=" * 80)

    try:
        # Step 00: Train/Test Split Manifest
        print(f"\n[1/4] Step 00: Train/Test Partitioning ({dataset_key}) ...")
        run_dataset_split(dataset_key)

        # Step 01: Explorer
        if not skip_explorer:
            print(f"\n[2/4] Step 01: Dataset Explorer ({dataset_key}) ...")
            run_dataset_explorer(dataset_key)
        else:
            print(f"\n[2/4] SKIPPING Step 01: Explorer")

        # Step 02: Data Preparation (N=32)
        print(f"\n[3/4] Step 02: Data Preparation & N=32 Resampling ({dataset_key}) ...")
        run_data_preparation(dataset_key)

        # Step 03: Model Training (CNN & Reps 1, 2, 3)
        print(f"\n[4/4] Step 03: Training Radar4DCNN Baseline ({dataset_key}) ...")
        run_model_training(dataset_key, model_name="cnn")

        for rep in ["rep1", "rep2", "rep3"]:
            print(f"\n[4/4] Step 03: Training Benchmark {rep.upper()} ({dataset_key}) ...")
            run_model_training(dataset_key, model_name=rep)

        elapsed = time.time() - start_time
        print(f"\n" + "=" * 80)
        print(f"SUCCESS: Pipeline for '{dataset_key.upper()}' completed in {elapsed:.2f}s")
        print("=" * 80)
        return True

    except Exception as ex:
        elapsed = time.time() - start_time
        print(f"\nFAILURE in pipeline '{dataset_key}' after {elapsed:.2f}s: {ex}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run dataset pipelines sequentially.")
    parser.add_argument("--dataset", type=str, choices=["mmfall", "mmwave", "combined", "all"], required=True)
    parser.add_argument("--skip-explorer", action="store_true", help="Skip exploratory step.")
    parser.add_argument("--timeout", type=int, default=1200, help="Execution timeout.")
    args = parser.parse_args()

    targets = ["mmfall", "mmwave", "combined"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        success = run_dataset_pipeline(ds, skip_explorer=args.skip_explorer)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()

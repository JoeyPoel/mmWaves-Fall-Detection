#!/usr/bin/env python3
"""
Sequential Pipeline Runner for mmWave Radar Fall Detection Datasets

Executes all notebooks in strict sequential order for a specified dataset pipeline.
Each notebook runs only if the previous step completes successfully. If any step fails,
execution halts immediately and an error report is displayed.

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
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Define strict sequential notebook pipeline order per dataset
DATASET_PIPELINES = {
    "mmfall": [
        "mmfall/01_mmfall_explorer.ipynb",
        "mmfall/02_mmfall_data_preparation.ipynb",
        "mmfall/03_train_cnn_mmfall.ipynb",
        "mmfall/representations/train_rep1_spectrogram_resnet18.ipynb",
        "mmfall/representations/train_rep2_projections_resnet18.ipynb",
        "mmfall/representations/train_rep3_pointnet_3d.ipynb",
        "mmfall/representations/compare_radar_representations.ipynb",
    ],
    "mmwave": [
        "mmwave-radar-fall-detection/01_mmwave_radar_fall_detection_explorer.ipynb",
        "mmwave-radar-fall-detection/02_mmwave_radar_fall_detection_data_preparation.ipynb",
        "mmwave-radar-fall-detection/03_train_cnn_mmwave_radar_fall_detection.ipynb",
        "mmwave-radar-fall-detection/representations/train_rep1_spectrogram_resnet18.ipynb",
        "mmwave-radar-fall-detection/representations/train_rep2_projections_resnet18.ipynb",
        "mmwave-radar-fall-detection/representations/train_rep3_pointnet_3d.ipynb",
        "mmwave-radar-fall-detection/representations/compare_radar_representations.ipynb",
    ],
    "combined": [
        "combined_mmfall_mmwave_radar/01_combined_datasets_explorer.ipynb",
        "combined_mmfall_mmwave_radar/02_combined_datasets_data_preparation.ipynb",
        "combined_mmfall_mmwave_radar/03_train_cnn_combined_datasets.ipynb",
        "combined_mmfall_mmwave_radar/representations/train_rep1_spectrogram_resnet18.ipynb",
        "combined_mmfall_mmwave_radar/representations/train_rep2_projections_resnet18.ipynb",
        "combined_mmfall_mmwave_radar/representations/train_rep3_pointnet_3d.ipynb",
        "combined_mmfall_mmwave_radar/representations/compare_radar_representations.ipynb",
    ],
}


def execute_notebook(notebook_path: Path, timeout: int = 600) -> bool:
    """
    Executes a single Jupyter notebook in place.
    Returns True if execution succeeds, False if an error occurs.
    """
    print(f"\n" + "=" * 80)
    print(f"EXECUTING: {notebook_path}")
    print("=" * 80)

    start_time = time.time()

    try:
        nb = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(notebook_path.parent)}},
        )
        client.execute()

        # Save executed notebook in place
        nbformat.write(nb, notebook_path)
        elapsed = time.time() - start_time
        print(f"SUCCESS: {notebook_path.name} finished in {elapsed:.2f}s")
        return True

    except CellExecutionError as err:
        elapsed = time.time() - start_time
        print(f"\nFAILURE: Exception raised in {notebook_path.name} after {elapsed:.2f}s")
        print(f"Error Message:\n{err}")
        return False
    except Exception as ex:
        elapsed = time.time() - start_time
        print(f"\nFAILURE: Unexpected error in {notebook_path.name} after {elapsed:.2f}s")
        print(f"Error Message:\n{ex}")
        return False


def run_dataset_pipeline(dataset_key: str, skip_explorer: bool = False, timeout: int = 600) -> bool:
    """
    Runs all notebooks for a given dataset in sequential order.
    Stops immediately if any notebook fails.
    """
    notebooks = DATASET_PIPELINES[dataset_key]
    total = len(notebooks)
    pipeline_start = time.time()

    print(f"\nStarting pipeline for dataset: '{dataset_key.upper()}' ({total} notebooks)")

    for idx, rel_path in enumerate(notebooks, 1):
        nb_path = Path(rel_path)

        if skip_explorer and "explorer" in nb_path.name:
            print(f"\n[{idx}/{total}] SKIPPING (explorer): {nb_path.name}")
            continue

        if not nb_path.exists():
            print(f"\nERROR: Notebook file does not exist: {nb_path}")
            print("Pipeline halted.")
            return False

        print(f"\n[{idx}/{total}] Running: {nb_path.name} ...")
        success = execute_notebook(nb_path, timeout=timeout)

        if not success:
            print(f"\nPIPELINE HALTED: Step {idx}/{total} ({nb_path.name}) failed.")
            print("Subsequent steps were NOT executed to preserve data consistency.")
            return False

    elapsed_total = time.time() - pipeline_start
    print(f"\nPIPELINE COMPLETED: Dataset '{dataset_key.upper()}' finished successfully in {elapsed_total:.2f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset notebooks in sequential order with strict error stopping."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmfall", "mmwave", "combined", "all"],
        required=True,
        help="Target dataset pipeline to execute ('mmfall', 'mmwave', 'combined', or 'all').",
    )
    parser.add_argument(
        "--skip-explorer",
        action="store_true",
        help="Optionally skip exploratory 01_*_explorer.ipynb notebooks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Cell execution timeout in seconds (default: 600s).",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        targets = ["mmfall", "mmwave", "combined"]
    else:
        targets = [args.dataset]

    overall_start = time.time()

    for ds in targets:
        success = run_dataset_pipeline(ds, skip_explorer=args.skip_explorer, timeout=args.timeout)
        if not success:
            print(f"\nPROCESS HALTED: Pipeline run for '{ds}' failed. Exiting.")
            sys.exit(1)

    overall_elapsed = time.time() - overall_start
    print(f"\n" + "=" * 80)
    print(f"ALL REQUESTED PIPELINES COMPLETED SUCCESSFULLY IN {overall_elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

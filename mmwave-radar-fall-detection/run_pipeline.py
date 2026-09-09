#!/usr/bin/env python3
"""
Pipeline Runner for TI IWR6843 Dataset (60-64 GHz)
Executes all TI IWR6843 notebooks sequentially, halting immediately if any step fails.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from run_pipeline import run_dataset_pipeline

if __name__ == "__main__":
    skip_explorer = "--skip-explorer" in sys.argv
    success = run_dataset_pipeline("mmwave", skip_explorer=skip_explorer)
    if not success:
        sys.exit(1)

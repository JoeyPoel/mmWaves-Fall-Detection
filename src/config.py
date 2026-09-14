"""
Central Dataclass Configuration Registry for mmWave Datasets
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DatasetConfig:
    dataset_key: str
    name: str
    frequency_ghz: str
    raw_data_dir: Path
    preproc_dir: Path
    models_dir: Path
    manifest_name: str
    target_n: int = 32
    window_size: int = 10
    stride: int = 2
    min_frames: int = 6
    spatial_box: List[float] = None # [x_min, x_max, y_min, y_max, z_min, z_max]
    min_snr: float = 100.0
    max_v: float = 3.0

    def __post_init__(self):
        if self.spatial_box is None:
            self.spatial_box = [-2.0, 2.0, 0.0, 6.0, -0.5, 2.2]

# Root directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROC_DIR = PROJECT_ROOT / "datasets" / "preprocessed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BENCHMARKS_DIR = MODELS_DIR / "benchmarks"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CONFIGS = {
    "mmfall": DatasetConfig(
        dataset_key="mmfall",
        name="mmFall (TI IWR1443)",
        frequency_ghz="77 GHz",
        raw_data_dir=PROJECT_ROOT / "datasets" / "mmfall" / "data",
        preproc_dir=PREPROC_DIR,
        models_dir=MODELS_DIR,
        manifest_name="mmfall_dataset_split.json"
    ),
    "mmwave": DatasetConfig(
        dataset_key="mmwave",
        name="TI IWR6843 Fall Detection",
        frequency_ghz="60–64 GHz",
        raw_data_dir=PROJECT_ROOT / "datasets" / "mmwave-radar-fall-detection" / "GatheredData",
        preproc_dir=PREPROC_DIR,
        models_dir=MODELS_DIR,
        manifest_name="ti_dataset_split.json"
    ),
    "combined": DatasetConfig(
        dataset_key="combined",
        name="Combined Multi-Sensor Benchmark (77 GHz + 60 GHz)",
        frequency_ghz="77 GHz + 60 GHz",
        raw_data_dir=PROJECT_ROOT / "datasets",
        preproc_dir=PREPROC_DIR,
        models_dir=MODELS_DIR,
        manifest_name="combined_dataset_split.json"
    )
}

"""
Unified Master Pipeline Engine across all datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import DATASET_CONFIGS, CHECKPOINTS_DIR, BENCHMARKS_DIR
from .dataset_utils import create_dataset_split_manifest, extract_dataset_windows_from_manifest
from .representation_utils import generate_spectrogram_rep, generate_projections_rep, generate_pointnet_rep
from .models import Radar4DCNN, ResNet18Spectrogram, ResNet18Projections, PointNetFallDetector
from .trainer import train_and_evaluate

def run_dataset_split(dataset_key: str):
    """Executes Step 00 zero-touch train/test splitting."""
    manifest_path = create_dataset_split_manifest(dataset_key)
    print(f"Step 00 Dataset Split Manifest created: {manifest_path}")
    return manifest_path

def run_dataset_explorer(dataset_key: str):
    """Executes Step 01 exploratory summary for training files."""
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"Step 01 Explorer summary for dataset: {cfg.name}")
    return cfg

def run_data_preparation(dataset_key: str):
    """Executes Step 02 window extraction, N=32 resampling, and saves train/val/test tensors."""
    cfg = DATASET_CONFIGS[dataset_key]
    res = extract_dataset_windows_from_manifest(dataset_key)
    X_tr, y_tr, df_tr_meta = res[0:3]
    X_va, y_va, df_va_meta = res[3:6]
    X_te, y_te, df_te_meta = res[6:9]

    cfg.preproc_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.preproc_dir / f"X_train_{dataset_key}.npy", X_tr)
    np.save(cfg.preproc_dir / f"y_train_{dataset_key}.npy", y_tr)
    np.save(cfg.preproc_dir / f"X_val_{dataset_key}.npy", X_va)
    np.save(cfg.preproc_dir / f"y_val_{dataset_key}.npy", y_va)
    np.save(cfg.preproc_dir / f"X_test_{dataset_key}.npy", X_te)
    np.save(cfg.preproc_dir / f"y_test_{dataset_key}.npy", y_te)

    # Legacy backward compatibility aliases
    tag_alias = "mmfall" if dataset_key == "mmfall" else ("ti" if dataset_key == "mmwave" else "combined")
    np.save(cfg.preproc_dir / f"X_{tag_alias}_clean_balanced.npy", X_tr)
    np.save(cfg.preproc_dir / f"y_{tag_alias}_clean_balanced.npy", y_tr)

    df_tr_meta.to_csv(cfg.preproc_dir / f"{tag_alias}_train_metadata.csv", index=False)
    df_va_meta.to_csv(cfg.preproc_dir / f"{tag_alias}_val_metadata.csv", index=False)
    df_te_meta.to_csv(cfg.preproc_dir / f"{tag_alias}_test_metadata.csv", index=False)

    print(f"Step 02 Preprocessing Complete for '{cfg.name}':")
    print(f"  Train Tensors: {X_tr.shape} (Labels: {np.bincount(y_tr)})")
    print(f"  Val Tensors:   {X_va.shape} (Labels: {np.bincount(y_va) if len(y_va) > 0 else []})")
    print(f"  Test Tensors:  {X_te.shape} (Labels: {np.bincount(y_te) if len(y_te) > 0 else []})")
    return X_tr, y_tr, X_va, y_va, X_te, y_te

def run_model_training(dataset_key: str, model_name: str = "cnn", epochs: int = 15, lr: float = 1e-3, batch_size: int = 32, variant: str = "raw"):
    """Executes model training & evaluation for CNN or representation benchmarks."""
    cfg = DATASET_CONFIGS[dataset_key]
    tag_alias = "mmfall" if dataset_key == "mmfall" else ("ti" if dataset_key == "mmwave" else "combined")

    X_tr_raw = np.load(cfg.preproc_dir / f"X_train_{dataset_key}.npy")
    y_tr = np.load(cfg.preproc_dir / f"y_train_{dataset_key}.npy")
    X_va_raw = np.load(cfg.preproc_dir / f"X_val_{dataset_key}.npy")
    y_va = np.load(cfg.preproc_dir / f"y_val_{dataset_key}.npy")
    X_te_raw = np.load(cfg.preproc_dir / f"X_test_{dataset_key}.npy")
    y_te = np.load(cfg.preproc_dir / f"y_test_{dataset_key}.npy")

    ckpt_path = CHECKPOINTS_DIR / f"ckpt_{model_name}_{tag_alias}_best.pth"
    res_json = BENCHMARKS_DIR / f"benchmark_results_{tag_alias}.json"

    if model_name == "cnn":
        X_tr = torch.tensor(X_tr_raw, dtype=torch.float32).permute(0, 3, 1, 2).numpy()
        X_va = torch.tensor(X_va_raw, dtype=torch.float32).permute(0, 3, 1, 2).numpy()
        X_te = torch.tensor(X_te_raw, dtype=torch.float32).permute(0, 3, 1, 2).numpy()
        model = Radar4DCNN(in_channels=4, num_classes=2)
        rep_key = "Radar4DCNN_Baseline"
    elif model_name == "rep1":
        X_tr = generate_spectrogram_rep(X_tr_raw, variant=variant)
        X_va = generate_spectrogram_rep(X_va_raw, variant=variant)
        X_te = generate_spectrogram_rep(X_te_raw, variant=variant)
        model = ResNet18Spectrogram(in_channels=3, num_classes=2)
        rep_key = f"Representation_1_Spectrogram_{variant}"
    elif model_name == "rep2":
        X_tr = generate_projections_rep(X_tr_raw, variant=variant)
        X_va = generate_projections_rep(X_va_raw, variant=variant)
        X_te = generate_projections_rep(X_te_raw, variant=variant)
        model = ResNet18Projections(in_channels=3, num_classes=2)
        rep_key = f"Representation_2_Projections_{variant}"
    elif model_name == "rep3":
        X_tr = generate_pointnet_rep(X_tr_raw, variant=variant)
        X_va = generate_pointnet_rep(X_va_raw, variant=variant)
        X_te = generate_pointnet_rep(X_te_raw, variant=variant)
        model = PointNetFallDetector(in_channels=5, num_classes=2)
        rep_key = f"Representation_3_PointNet_{variant}"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    print(f"\n--- Training {rep_key} (variant='{variant}') on {cfg.name} (epochs={epochs}, lr={lr}, batch_size={batch_size}) ---")
    return train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te, ckpt_path, res_json, rep_key, epochs=epochs, lr=lr, batch_size=batch_size)

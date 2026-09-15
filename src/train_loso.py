"""
Leave-One-Subject-Out (LOSO) Cross-Validation Training Script for TI mmWave Radar Fall Detection.
Executes 3-fold LOSO cross-validation across Representation 1 (Doppler-Time Map),
Representation 2 (2D Orthogonal Projections), and Representation 3 (Native 3D Point Sets).
Logs out-of-fold Accuracy, Precision, Recall, and Macro F1-score.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.dataset_parser import get_loso_splits, load_recording_df, extract_label, DATA_DIR
from src.transforms import (
    extract_raw_windows_from_df,
    FoldScaler,
    transform_rep1_doppler_time,
    transform_rep2_orthogonal_projections,
    transform_rep3_pointnet
)
from src.models import ResNet18Adaptor, PointNetPlusPlusFallDetector

CHECKPOINTS_DIR = Path("models/checkpoints")
BENCHMARKS_DIR = Path("models/benchmarks")

def extract_dataset_for_files(file_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a list of CSV recording files and extracts sliding windows.
    Returns:
       X_raw: (N, 10, 32, 4) array of target-relative windows
       y_raw: (N,) array of binary labels
    """
    windows_list = []
    labels_list = []
    print(f"Extracting windows from {len(file_list)} recording files...", flush=True)
    for idx, fpath in enumerate(file_list):
        lbl = extract_label(fpath)
        df_clean = load_recording_df(fpath)
        win_arrs = extract_raw_windows_from_df(df_clean, window_size_frames=10, stride=5)
        for win in win_arrs:
            windows_list.append(win)
            labels_list.append(lbl)

    if len(windows_list) == 0:
        return np.zeros((0, 10, 32, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(windows_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)

def train_one_fold(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    epochs: int = 25,
    lr: float = 5e-4,
    batch_size: int = 32,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains model on single fold training set with PyTorch WeightedRandomSampler mini-batch balance.
    Evaluates on Validation fold and Test fold.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_y_t = torch.tensor(y_tr, dtype=torch.long)
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), train_y_t)
    va_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))

    # Balanced mini-batch sampling
    class_counts = torch.bincount(train_y_t)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_y_t]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        r_loss = 0.0
        corr = 0
        tot = 0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * bx.size(0)
            corr += (out.argmax(dim=1) == by).sum().item()
            tot += by.size(0)

        if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
            print(f"  Epoch [{ep+1:02d}/{epochs:02d}] Train Loss: {r_loss/max(tot,1):.4f}, Train Acc: {corr/max(tot,1)*100:.2f}%", flush=True)

    # Evaluation on Validation fold & Test fold
    model.eval()
    val_probs, test_probs = [], []
    with torch.no_grad():
        for bx, by in va_loader:
            bx = bx.to(device)
            out = model(bx)
            prob = torch.softmax(out, dim=1)[:, 1]
            val_probs.extend(prob.cpu().numpy())

        for bx, by in te_loader:
            bx = bx.to(device)
            out = model(bx)
            prob = torch.softmax(out, dim=1)[:, 1]
            test_probs.extend(prob.cpu().numpy())

    return np.array(val_probs, dtype=np.float32), y_va, np.array(test_probs, dtype=np.float32), y_te


def run_loso_cross_validation(
    rep_key: str = "rep1",
    epochs: int = 25,
    lr: float = 5e-4,
    batch_size: int = 32
) -> Dict[str, any]:
    """
    Executes 3-fold Leave-One-Subject-Out (LOSO) cross-validation across Train, Validation, and Test sets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = get_loso_splits()

    oof_val_targets, oof_val_probs = [], []
    oof_test_targets, oof_test_probs = [], []

    print(f"\n=======================================================")
    print(f"Starting 3-Fold Train/Val/Test LOSO Cross-Validation for Representation: {rep_key}")
    print(f"=======================================================")

    for fold_info in splits:
        fold_idx = fold_info["fold"]
        train_subjs = fold_info.get("train_subjects", [fold_info.get("train_subject", "Unknown")])
        train_subjs_str = ", ".join(train_subjs) if isinstance(train_subjs, list) else str(train_subjs)
        test_subj = fold_info.get("test_subject", "Unknown")
        print(f"\n--- Fold {fold_idx} | Train/Val Pool: {train_subjs_str} | Held-Out Test: {test_subj} ---")

        # 1. Extract raw target-relative windows
        X_tr_raw, y_tr = extract_dataset_for_files(fold_info["train_files"])
        X_va_raw, y_va = extract_dataset_for_files(fold_info["val_files"])
        X_te_raw, y_te = extract_dataset_for_files(fold_info["test_files"])
        print(f"Train windows: {len(X_tr_raw)} | Val windows: {len(X_va_raw)} | Test windows: {len(X_te_raw)}")

        # 2. Fit scaler strictly on training fold
        scaler = FoldScaler()
        scaler.fit(X_tr_raw)
        X_tr_scaled = scaler.transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        X_te_scaled = scaler.transform(X_te_raw)

        # 3. Transform into specific representation
        key_clean = str(rep_key).lower().strip()
        if key_clean in ("rep1_doppler", "rep1"): # Doppler-Time Map (ResNet-18)
            X_tr = transform_rep1_doppler_time(X_tr_scaled)
            X_va = transform_rep1_doppler_time(X_va_scaled)
            X_te = transform_rep1_doppler_time(X_te_scaled)
            model = ResNet18Adaptor(in_channels=3, num_classes=2)
        elif key_clean in ("rep2_projections", "rep2"): # 2D Orthogonal Projections (ResNet-18)
            X_tr = transform_rep2_orthogonal_projections(X_tr_scaled)
            X_va = transform_rep2_orthogonal_projections(X_va_scaled)
            X_te = transform_rep2_orthogonal_projections(X_te_scaled)
            model = ResNet18Adaptor(in_channels=3, num_classes=2)
        elif key_clean in ("rep3_pointset", "rep3"): # Native 3D Point Tensor (PointNet++)
            X_tr = transform_rep3_pointnet(X_tr_scaled)
            X_va = transform_rep3_pointnet(X_va_scaled)
            X_te = transform_rep3_pointnet(X_te_scaled)
            model = PointNetPlusPlusFallDetector(in_channels=5, num_classes=2)
        else:
            raise ValueError(f"Unknown rep_key '{rep_key}'")

        # 4. Train model on training fold and evaluate on Val and Test folds
        val_probs, val_targets, test_probs, test_targets = train_one_fold(
            model, X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=epochs, lr=lr, batch_size=batch_size, device=device
        )

        oof_val_probs.extend(val_probs)
        oof_val_targets.extend(val_targets)
        oof_test_probs.extend(test_probs)
        oof_test_targets.extend(test_targets)

        # Calculate fold metrics
        f_val_preds = (val_probs >= 0.50).astype(int)
        f_acc = accuracy_score(val_targets, f_val_preds)
        f_f1 = f1_score(val_targets, f_val_preds, average="macro", zero_division=0)
        print(f"Fold {fold_idx} Val Result -> Val Accuracy: {f_acc*100:.2f}%, Val Macro F1: {f_f1*100:.2f}%")

    # Helper for calculating performance metrics dict
    def compute_metrics(targets, probs):
        targets_arr = np.array(targets)
        probs_arr = np.array(probs)
        preds_arr = (probs_arr >= 0.50).astype(int)

        acc = float(accuracy_score(targets_arr, preds_arr))
        prec = float(precision_score(targets_arr, preds_arr, zero_division=0))
        rec = float(recall_score(targets_arr, preds_arr, zero_division=0))
        macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
        auc = float(roc_auc_score(targets_arr, probs_arr)) if len(np.unique(targets_arr)) > 1 else 0.5

        cm = confusion_matrix(targets_arr, preds_arr)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "accuracy": acc, "precision": prec, "recall": rec, "macro_f1": macro_f1,
            "roc_auc": auc, "fpr": fpr, "confusion_matrix": cm.tolist()
        }

    val_res = compute_metrics(oof_val_targets, oof_val_probs)
    test_res = compute_metrics(oof_test_targets, oof_test_probs)

    print(f"\n--- Aggregated Validation Set Results for {rep_key.upper()} ---")
    print(f"Val Acc: {val_res['accuracy']*100:.2f}% | Val Rec: {val_res['recall']*100:.2f}% | Val Prec: {val_res['precision']*100:.2f}% | Val Macro F1: {val_res['macro_f1']*100:.2f}% | Val ROC-AUC: {val_res['roc_auc']:.4f}")

    print(f"\n--- Aggregated Test Set Results for {rep_key.upper()} ---")
    print(f"Test Acc: {test_res['accuracy']*100:.2f}% | Test Rec: {test_res['recall']*100:.2f}% | Test Prec: {test_res['precision']*100:.2f}% | Test Macro F1: {test_res['macro_f1']*100:.2f}% | Test ROC-AUC: {test_res['roc_auc']:.4f}")

    results = {
        "representation": rep_key,
        "accuracy": val_res["accuracy"],
        "precision": val_res["precision"],
        "recall": val_res["recall"],
        "macro_f1": val_res["macro_f1"],
        "roc_auc": val_res["roc_auc"],
        "fpr": val_res["fpr"],
        "confusion_matrix": val_res["confusion_matrix"],
        "val_metrics": val_res,
        "test_metrics": test_res
    }
    return results



if __name__ == "__main__":
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for rep in ["rep1", "rep2", "rep3"]:
        res = run_loso_cross_validation(rep_key=rep, epochs=20, lr=5e-4, batch_size=32)
        all_res[rep] = res

    out_path = BENCHMARKS_DIR / "loso_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=4)
    print(f"\nSaved LOSO benchmark results to {out_path}")

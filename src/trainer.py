"""
Unified PyTorch Model Training & Evaluation Engine.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, confusion_matrix

def train_and_evaluate(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, checkpoint_path: Path,
                       results_json_path: Path = None, rep_key: str = None, epochs: int = 15, lr: float = 1e-3, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    # Class-weighted loss to handle training imbalance without discarding data
    class_counts = np.bincount(y_train)
    c0_cnt = max(class_counts[0], 1) if len(class_counts) > 0 else 1
    c1_cnt = max(class_counts[1], 1) if len(class_counts) > 1 else 1
    class_weights = torch.tensor([len(y_train) / (2.0 * c0_cnt), len(y_train) / (2.0 * c1_cnt)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_val_f2 = -1.0
    best_threshold = 0.50
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        r_loss, corr, tot = 0.0, 0, 0
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

        train_loss = r_loss / max(tot, 1)
        train_acc = corr / max(tot, 1)

        model.eval()
        v_loss, v_probs, v_targets = 0.0, [], []
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                prob = torch.softmax(out, dim=1)[:, 1]
                v_loss += loss.item() * bx.size(0)
                v_probs.extend(prob.cpu().numpy())
                v_targets.extend(by.cpu().numpy())

        v_tot = max(len(v_targets), 1)
        val_loss = v_loss / v_tot
        v_probs_arr = np.array(v_probs)
        v_targets_arr = np.array(v_targets)

        # Threshold sweep on natural validation set (0.10 to 0.90) to optimize Fall F2-Score (Recall heavily weighted)
        ep_best_f2 = -1.0
        ep_best_thresh = 0.50
        if len(v_targets_arr) > 0 and len(np.unique(v_targets_arr)) > 1:
            for thresh in np.arange(0.10, 0.90, 0.02):
                preds_th = (v_probs_arr >= thresh).astype(int)
                f2_th = fbeta_score(v_targets_arr, preds_th, beta=2.0, zero_division=0)
                if f2_th > ep_best_f2:
                    ep_best_f2 = f2_th
                    ep_best_thresh = thresh
        else:
            ep_best_f2 = 0.0

        val_preds_opt = (v_probs_arr >= ep_best_thresh).astype(int) if len(v_probs_arr) > 0 else []
        val_acc = accuracy_score(v_targets_arr, val_preds_opt) if len(v_targets_arr) > 0 else 0.0
        val_rec = recall_score(v_targets_arr, val_preds_opt, zero_division=0) if len(v_targets_arr) > 0 else 0.0
        val_f1 = f1_score(v_targets_arr, val_preds_opt, zero_division=0) if len(v_targets_arr) > 0 else 0.0
        val_f2 = ep_best_f2

        if val_f2 > best_val_f2 or (abs(val_f2 - best_val_f2) < 1e-4 and val_loss < best_val_loss):
            best_val_f2 = val_f2
            best_val_loss = val_loss
            best_threshold = ep_best_thresh
            torch.save(model.state_dict(), checkpoint_path)
            saved_mark = " [BEST SAVED]"
        else:
            saved_mark = ""

        if (ep + 1) % 5 == 0 or ep == 0 or (ep + 1) == epochs:
            print(f"Epoch [{ep+1:02d}/{epochs:02d}] Tr Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F2: {val_f2*100:.2f}%, Rec: {val_rec*100:.2f}% (Thresh: {ep_best_thresh:.2f}){saved_mark}")

    # Evaluate Best Checkpoint on Test Set using Frozen Optimal Threshold
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    probs_list, targets_list = [], []
    with torch.no_grad():
        for bx, by in te_loader:
            bx = bx.to(device)
            out = model(bx)
            prob = torch.softmax(out, dim=1)[:, 1]
            probs_list.extend(prob.cpu().numpy())
            targets_list.extend(by.numpy())

    targets_arr = np.array(targets_list)
    probs_arr = np.array(probs_list)
    preds_arr = (probs_arr >= best_threshold).astype(int)

    acc = float(accuracy_score(targets_arr, preds_arr))
    rec = float(recall_score(targets_arr, preds_arr, zero_division=0))
    prec = float(precision_score(targets_arr, preds_arr, zero_division=0))
    f1 = float(f1_score(targets_arr, preds_arr, zero_division=0))
    auc = float(roc_auc_score(targets_arr, probs_arr)) if len(np.unique(targets_arr)) > 1 else 0.5
    cm = confusion_matrix(targets_arr, preds_arr)
    fpr = float(cm[0, 1] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0

    print(f"\n--- Test Results: {rep_key or 'Radar4DCNN'} (Frozen Val Decision Threshold: {best_threshold:.2f}) ---")
    print(f"Test Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | Recall: {rec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1*100:.2f}%")

    if results_json_path and rep_key:
        import datetime
        now = datetime.datetime.now()
        ts_clean = now.strftime("%Y%m%d_%H%M%S")
        ts_display = now.strftime("%Y-%m-%d %H:%M:%S")

        run_id = f"benchmark_{rep_key}_ep{epochs}_lr{lr}_bs{batch_size}_th{best_threshold:.2f}_{ts_clean}"
        single_json_path = results_json_path.parent / f"{run_id}.json"
        single_png_path = results_json_path.parent / f"{run_id}.png"

        run_data = {
            "timestamp": ts_display,
            "rep_key": rep_key,
            "run_id": run_id,
            "hyperparameters": {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "optimal_threshold": float(best_threshold)
            },
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1_score": f1,
            "roc_auc": auc,
            "fpr": fpr,
            "optimal_threshold": float(best_threshold),
            "confusion_matrix": cm.tolist(),
            "json_path": str(single_json_path),
            "image_path": str(single_png_path)
        }

        # Save single timestamped benchmark run JSON
        with open(single_json_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=4)
        print(f"Saved individual run benchmark JSON to: {single_json_path}")

        # Update master dataset benchmark JSON
        bm_res = {}
        if results_json_path.exists():
            try:
                bm_res = json.load(open(results_json_path))
            except Exception:
                bm_res = {}
        bm_res[rep_key] = run_data
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(bm_res, f, indent=4)
        print(f"Saved master benchmark summary to: {results_json_path}")

    return acc, auc, cm

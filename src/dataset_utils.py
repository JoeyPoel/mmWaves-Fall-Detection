"""
Data cleaning, coordinate shift, N=32 uniform resampling, zero-touch splitting, and window extraction utilities.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATASET_CONFIGS, DatasetConfig

def uniform_resample_frame(frame_points: np.ndarray, target_n: int = 32, rng=None) -> np.ndarray:
    """
    Uniform random sampling with replacement (M < target_n) or without replacement (M >= target_n).
    Preserves exact measured coordinates [delta_x, delta_y, z, v].
    """
    if rng is None:
        rng = np.random.default_rng(42)
    M = len(frame_points)
    if M == 0:
        return np.zeros((target_n, frame_points.shape[1] if frame_points.ndim > 1 else 4), dtype=np.float32)
    if M < target_n:
        idx = rng.choice(M, size=target_n, replace=True)
    else:
        idx = rng.choice(M, size=target_n, replace=False)
    return frame_points[idx].astype(np.float32)

class MMfallDataCleaner:
    def __init__(self, height=1.80, tilt_angle_deg=10.0, min_snr=100, max_v=3.0):
        self.height = height
        self.min_snr = min_snr
        self.max_v = max_v
        tilt_rad = np.deg2rad(tilt_angle_deg)
        self.rot_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt_rad), np.sin(tilt_rad)],
            [0.0, -np.sin(tilt_rad), np.cos(tilt_rad)]
        ])

    def clean_and_parse_file(self, npy_filepath: str) -> pd.DataFrame:
        arr = np.load(npy_filepath, allow_pickle=True)
        records = []
        for frame_idx, frame_data in enumerate(arr):
            if frame_data is None or len(frame_data) == 0:
                continue
            p0 = frame_data[0]
            cx, cy, cz = p0[3], p0[4], p0[5]
            rot_c = np.matmul(self.rot_matrix, np.array([cx, cy, cz]))
            cx_room, cy_room, cz_room = rot_c[0], rot_c[1], rot_c[2] + self.height

            for p_idx, point in enumerate(frame_data):
                if len(point) < 15:
                    continue
                r, az, el, v, snr, noise = point[9], point[10], point[11], point[12], point[13], point[14]
                if snr < self.min_snr or abs(v) > self.max_v:
                    continue

                xs = r * np.cos(el) * np.sin(az)
                ys = r * np.cos(el) * np.cos(az)
                zs = r * np.sin(el)
                rot = np.matmul(self.rot_matrix, np.array([xs, ys, zs]))
                x_room, y_room, z_room = rot[0], rot[1], rot[2] + self.height

                if not (-2.0 <= x_room <= 2.0 and 0.0 <= y_room <= 6.0 and -0.5 <= z_room <= 2.2):
                    continue

                rcs = 40.0 * np.log10(max(r, 1e-3)) + 0.1 * snr + 0.1 * noise
                records.append({
                    "frame": frame_idx, "DetObj#": p_idx,
                    "x": x_room, "y": y_room, "z": z_room,
                    "centroid_z": cz_room, "centroid_x": cx_room, "centroid_y": cy_room,
                    "v": v, "snr": snr, "range": r, "rcs": rcs
                })
        df = pd.DataFrame(records)
        if len(df) > 0:
            df["time_s"] = (df["frame"] / 10.0).round(2)
        return df

def safe_group_split(files, labels, groups, test_size=0.20, random_state=42):
    unique_groups = list(set(groups))
    if len(unique_groups) > 1:
        from sklearn.model_selection import GroupShuffleSplit
        for seed in range(random_state, random_state + 100):
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            try:
                train_idx, test_idx = next(gss.split(files, labels, groups))
                if len(train_idx) > 0 and len(test_idx) > 0:
                    y_tr = [labels[i] for i in train_idx]
                    y_te = [labels[i] for i in test_idx]
                    if len(set(y_tr)) > 1 and len(set(y_te)) > 1:
                        tr_files = [files[i] for i in train_idx]
                        te_files = [files[i] for i in test_idx]
                        return tr_files, te_files, y_tr, y_te
            except Exception:
                pass
    tr_files, te_files, y_tr, y_te = train_test_split(files, labels, test_size=test_size, random_state=random_state, stratify=labels)
    return tr_files, te_files, y_tr, y_te

def create_dataset_split_manifest(dataset_key: str) -> Path:
    cfg = DATASET_CONFIGS[dataset_key]
    cfg.preproc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.preproc_dir / cfg.manifest_name

    if dataset_key == "mmfall":
        data_dir = cfg.raw_data_dir
        ds_files = sorted([str(p.resolve()) for p in (list((data_dir / "DS1").glob("*.npy")) + list((data_dir / "DS2").glob("*.npy")))
                           if p.stem not in ["normal_test_data", "normal_train_data", "DS2"]])
        categories = [1 if ("fall" in Path(p).stem.lower() or any(k in Path(p).stem for k in ["_bf_", "_ff_", "_lf_", "_rf_", "_sf_"])) else 0 for p in ds_files]
        # Group by recording trial / session prefix
        groups = [f"{Path(p).parent.name}_{Path(p).stem.split('_')[1] if '_' in Path(p).stem else Path(p).stem}" for p in ds_files]
        
        tr_val_files, te_files, y_tr_val, y_te = safe_group_split(ds_files, categories, groups, test_size=0.20, random_state=42)
        tr_val_groups = [groups[ds_files.index(f)] for f in tr_val_files]
        tr_files, val_files, y_tr, y_val = safe_group_split(tr_val_files, y_tr_val, tr_val_groups, test_size=0.20, random_state=42)
        
        manifest = {
            "dataset": dataset_key,
            "train_files": tr_files, "val_files": val_files, "test_files": te_files,
            "train_categories": y_tr, "val_categories": y_val, "test_categories": y_te
        }

    elif dataset_key == "mmwave":
        data_dir = cfg.raw_data_dir
        fall_files = [str(p.resolve()) for p in (data_dir / "Fall").glob("*.csv")]
        not_files = [str(p.resolve()) for p in (data_dir / "Not").glob("*.csv")]
        all_csvs = sorted(fall_files + not_files)
        labels = [1 if Path(p).parent.name.lower() == "fall" else 0 for p in all_csvs]
        # Participant subject grouping (e.g. Areeb, Raffay, Towsif)
        groups = [Path(p).stem.split('_')[0] for p in all_csvs]
        
        tr_val_files, te_files, y_tr_val, y_te = safe_group_split(all_csvs, labels, groups, test_size=0.20, random_state=42)
        tr_val_groups = [groups[all_csvs.index(f)] for f in tr_val_files]
        tr_files, val_files, y_tr, y_val = safe_group_split(tr_val_files, y_tr_val, tr_val_groups, test_size=0.20, random_state=42)
        
        manifest = {
            "dataset": dataset_key,
            "train_files": tr_files, "val_files": val_files, "test_files": te_files,
            "train_categories": y_tr, "val_categories": y_val, "test_categories": y_te
        }

    elif dataset_key == "combined":
        mf_manifest = json.load(open(cfg.preproc_dir / DATASET_CONFIGS["mmfall"].manifest_name))
        ti_manifest = json.load(open(cfg.preproc_dir / DATASET_CONFIGS["mmwave"].manifest_name))
        manifest = {
            "dataset": dataset_key,
            "train_files": mf_manifest["train_files"] + ti_manifest["train_files"],
            "val_files": mf_manifest["val_files"] + ti_manifest["val_files"],
            "test_files": mf_manifest["test_files"] + ti_manifest["test_files"],
            "train_categories": mf_manifest["train_categories"] + ti_manifest["train_categories"],
            "val_categories": mf_manifest["val_categories"] + ti_manifest["val_categories"],
            "test_categories": mf_manifest["test_categories"] + ti_manifest["test_categories"]
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    return manifest_path

def extract_dataset_windows_from_manifest(dataset_key: str):
    cfg = DATASET_CONFIGS[dataset_key]
    manifest_path = cfg.preproc_dir / cfg.manifest_name
    if not manifest_path.exists():
        create_dataset_split_manifest(dataset_key)
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if dataset_key == "combined":
        res_mf = extract_dataset_windows_from_manifest("mmfall")
        X_tr_mf, y_tr_mf, df_tr_mf = res_mf[0:3]
        X_va_mf, y_va_mf, df_va_mf = res_mf[3:6]
        X_te_mf, y_te_mf, df_te_mf = res_mf[6:9]

        res_ti = extract_dataset_windows_from_manifest("mmwave")
        X_tr_ti, y_tr_ti, df_tr_ti = res_ti[0:3]
        X_va_ti, y_va_ti, df_va_ti = res_ti[3:6]
        X_te_ti, y_te_ti, df_te_ti = res_ti[6:9]

        X_tr_raw = np.concatenate([X_tr_mf, X_tr_ti], axis=0)
        y_tr_raw = np.concatenate([y_tr_mf, y_tr_ti], axis=0)
        X_va_raw = np.concatenate([X_va_mf, X_va_ti], axis=0)
        y_va_raw = np.concatenate([y_va_mf, y_va_ti], axis=0)
        X_te_raw = np.concatenate([X_te_mf, X_te_ti], axis=0)
        y_te_raw = np.concatenate([y_te_mf, y_te_ti], axis=0)

        np.random.seed(42)
        p_tr = np.random.permutation(len(X_tr_raw))
        p_va = np.random.permutation(len(X_va_raw))
        p_te = np.random.permutation(len(X_te_raw))
        return (
            X_tr_raw[p_tr], y_tr_raw[p_tr], pd.concat([df_tr_mf, df_tr_ti]).iloc[p_tr].reset_index(drop=True),
            X_va_raw[p_va], y_va_raw[p_va], pd.concat([df_va_mf, df_va_ti]).iloc[p_va].reset_index(drop=True),
            X_te_raw[p_te], y_te_raw[p_te], pd.concat([df_te_mf, df_te_ti]).iloc[p_te].reset_index(drop=True)
        )

    cleaner = MMfallDataCleaner() if dataset_key == "mmfall" else None
    rng = np.random.default_rng(42)

    def process_file_list(file_list, is_train=True):
        windows, labels, meta = [], [], []
        for file_str in file_list:
            fpath = Path(file_str)
            if not fpath.exists():
                continue
            if dataset_key == "mmfall":
                df_clean = cleaner.clean_and_parse_file(str(fpath))
            else:
                df_raw = pd.read_csv(fpath)
                df_clean = df_raw[(df_raw["x"].between(-2.0, 2.0)) & 
                                  (df_raw["y"].between(0.0, 6.0)) & 
                                  (df_raw["z"].between(-0.5, 2.2)) & 
                                  (df_raw["v"].abs() <= 3.0) & 
                                  (df_raw["snr"] >= 100)]
            if len(df_clean) == 0:
                continue

            csv_p = fpath.with_suffix(".csv")
            gt_markers = []
            if dataset_key == "mmfall" and csv_p.exists():
                with open(csv_p, "r") as f:
                    gt_markers = [int(line.strip()) for line in f if line.strip().isdigit()]

            all_frames = np.sort(df_clean["frame"].unique())
            max_frame = all_frames.max()

            for start_f in range(0, max_frame - 9, cfg.stride):
                end_f = start_f + 9
                df_win = df_clean[(df_clean["frame"] >= start_f) & (df_clean["frame"] <= end_f)]
                if df_win["frame"].nunique() < cfg.min_frames:
                    continue

                if dataset_key == "mmfall":
                    is_gt_fall = any(start_f - 10 <= gm <= end_f + 10 for gm in gt_markers)
                    cz_start = df_win[df_win["frame"] == df_win["frame"].min()]["centroid_z"].values
                    cz_end = df_win[df_win["frame"] == df_win["frame"].max()]["centroid_z"].values
                    h_drop = (cz_start[0] - cz_end[0]) if len(cz_start) > 0 and len(cz_end) > 0 else 0.0
                    is_fall_file = "fall" in fpath.stem.lower() or any(k in fpath.stem for k in ["_bf_", "_ff_", "_lf_", "_rf_", "_sf_"])
                    label = 1 if (is_gt_fall or (is_fall_file and h_drop > 0.50)) else 0
                    ref_x0 = df_win["centroid_x"].iloc[0]
                    ref_y0 = df_win["centroid_y"].iloc[0]
                else:
                    label = 1 if fpath.parent.name.lower() == "fall" else 0
                    ref_x0 = df_win["x"].iloc[0]
                    ref_y0 = df_win["y"].iloc[0]

                win_frames = []
                for f_i in range(start_f, end_f + 1):
                    df_f = df_win[df_win["frame"] == f_i]
                    if len(df_f) > 0:
                        delta_x = df_f["x"].values - ref_x0
                        delta_y = df_f["y"].values - ref_y0
                        z_vals = df_f["z"].values
                        v_vals = df_f["v"].values
                        pts = np.column_stack([delta_x, delta_y, z_vals, v_vals])
                    else:
                        pts = np.zeros((0, 4))
                    pts_ov = uniform_resample_frame(pts, target_n=cfg.target_n, rng=rng)
                    win_frames.append(pts_ov)

                windows.append(np.array(win_frames, dtype=np.float32))
                labels.append(label)
                meta.append({
                    "file": fpath.stem, "category": "Fall" if label == 1 else "ADL",
                    "start_frame": start_f, "end_frame": end_f, "label": label
                })

        X_arr = np.array(windows, dtype=np.float32)
        y_arr = np.array(labels, dtype=np.int64)
        df_meta = pd.DataFrame(meta)

        if is_train and len(y_arr) > 0:
            fall_idx = np.where(y_arr == 1)[0]
            adl_idx = np.where(y_arr == 0)[0]
            if len(fall_idx) > 0 and len(adl_idx) > 0:
                n_sample = min(len(fall_idx), len(adl_idx))
                np.random.seed(42)
                sampled_fall = np.random.choice(fall_idx, size=n_sample, replace=False)
                sampled_adl = np.random.choice(adl_idx, size=n_sample, replace=False)
                balanced_idx = np.sort(np.concatenate([sampled_fall, sampled_adl]))
                return X_arr[balanced_idx], y_arr[balanced_idx], df_meta.iloc[balanced_idx].reset_index(drop=True)
        return X_arr, y_arr, df_meta

    X_train, y_train, df_train_meta = process_file_list(manifest["train_files"], is_train=True)
    X_val, y_val, df_val_meta = process_file_list(manifest.get("val_files", []), is_train=False)
    X_test, y_test, df_test_meta = process_file_list(manifest["test_files"], is_train=False)
    return X_train, y_train, df_train_meta, X_val, y_val, df_val_meta, X_test, y_test, df_test_meta

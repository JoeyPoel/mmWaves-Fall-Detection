import json
from pathlib import Path

Path("notebooks").mkdir(parents=True, exist_ok=True)

# Remove legacy/old notebook naming variants if they exist
for old_file in ["00_data_inspection_split.ipynb", "01_exploration_and_centering.ipynb", "02_preprocessing_transforms.ipynb"]:
    old_p = Path("notebooks") / old_file
    if old_p.exists():
        old_p.unlink()
        print(f"Removed old variant notebook: {old_p}")

# ==============================================================================
# 00_train_val_test_split.ipynb
# ==============================================================================
nb00 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 00: Dataset Integrity & 3-Way Train/Val/Test Partitioning\n",
                "\n",
                "Ingests the **TI IWR6843 mmWave Radar Dataset** (102 CSV files: 51 Falls, 51 ADLs across subjects **Areeb**, **Raffay**, **Towsif**) and partitions them into **Train**, **Validation**, and **Test** sets with zero cross-subject data leakage.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext autoreload\n",
                "%autoreload 2\n",
                "import sys\n",
                "from pathlib import Path\n",
                "root_dir = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n",
                "if str(root_dir) not in sys.path: sys.path.insert(0, str(root_dir))\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.dataset_parser import get_all_recording_files, get_loso_splits, extract_subject, extract_label\n",
                "from src.config import DATASET_CONFIGS\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "print(f\"Dataset: {DATASET_CONFIGS['ti'].name}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Ingestion Audit (102 CSV Files)\n",
                "all_files = get_all_recording_files()\n",
                "records = []\n",
                "for f in all_files:\n",
                "    subj = extract_subject(f)\n",
                "    lbl = extract_label(f)\n",
                "    df_raw = pd.read_csv(f)\n",
                "    records.append({\n",
                "        \"file\": f.name,\n",
                "        \"subject\": subj,\n",
                "        \"label\": lbl,\n",
                "        \"class_name\": \"Fall\" if lbl == 1 else \"Non-Fall (ADL)\",\n",
                "        \"total_frames\": df_raw[\"frame\"].nunique(),\n",
                "        \"total_points\": len(df_raw)\n",
                "    })\n",
                "df_meta = pd.DataFrame(records)\n",
                "print(f\"Discovered {len(df_meta)} CSV recording files (51 Falls vs 51 ADLs).\")\n",
                "df_meta.head(10)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. 3-Way Train / Validation / Test Partitioning per Fold\n",
                "loso_splits = get_loso_splits()\n",
                "print(\"=== 3-Fold LOSO Train / Validation / Test Partitioning Scheme ===\")\n",
                "for fold_info in loso_splits:\n",
                "    f_idx = fold_info[\"fold\"]\n",
                "    tr_subs = \", \".join(fold_info[\"train_subjects\"])\n",
                "    te_sub = fold_info[\"test_subject\"]\n",
                "    tr_f = len(fold_info['train_files'])\n",
                "    va_f = len(fold_info['val_files'])\n",
                "    te_f = len(fold_info['test_files'])\n",
                "    print(f\"Fold {f_idx} [Held-Out Test Subject: {te_sub:6s}]: \"\n",
                "          f\"Train = {tr_f} files (27 Falls, 27 ADLs) | \"\n",
                "          f\"Val = {va_f} files (7 Falls, 7 ADLs) | \"\n",
                "          f\"Test = {te_f} files (17 Falls, 17 ADLs)\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Visual Breakdown of 3-Way LOSO Partitioning (54 Train / 14 Val / 34 Test)\n",
                "fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)\n",
                "fold_labels = [f\"Fold {f['fold']}\\n(Test: {f['test_subject']})\" for f in loso_splits]\n",
                "tr_counts = [len(f[\"train_files\"]) for f in loso_splits]\n",
                "va_counts = [len(f[\"val_files\"]) for f in loso_splits]\n",
                "te_counts = [len(f[\"test_files\"]) for f in loso_splits]\n",
                "\n",
                "x_idx = np.arange(len(fold_labels))\n",
                "ax.bar(x_idx - 0.25, tr_counts, width=0.25, label=\"Train Set (54 files: 27 Fall/27 ADL)\", color=\"#3498db\")\n",
                "ax.bar(x_idx, va_counts, width=0.25, label=\"Val Set (14 files: 7 Fall/7 ADL)\", color=\"#f39c12\")\n",
                "ax.bar(x_idx + 0.25, te_counts, width=0.25, label=\"Held-Out Test Set (34 files: 17 Fall/17 ADL)\", color=\"#e74c3c\")\n",
                "ax.set_xticks(x_idx)\n",
                "ax.set_xticklabels(fold_labels)\n",
                "ax.set_title(\"3-Fold LOSO 3-Way Partitioning (54 Train / 14 Val / 34 Test)\", fontweight=\"bold\")\n",
                "ax.set_ylabel(\"Number of CSV Recording Files\")\n",
                "ax.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }

    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open("notebooks/00_train_val_test_split.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb00, f, indent=2)


# ==============================================================================
# 01_dataset_explorer.ipynb
# ==============================================================================
nb01 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 01: Dataset Explorer & 3 Representation Visualizations\n",
                "\n",
                "Explores raw mmWave radar point cloud motion trajectories and visually renders sample data across **all 3 representations** side-by-side:\n",
                "1. **Representation 1 (`rep1_doppler`)**: Micro-Doppler Velocity-Time Spectrogram Map.\n",
                "2. **Representation 2 (`rep2_projections`)**: 2D Orthogonal Spatial Occupancy Projections (Elevation, Ground, Doppler-Height).\n",
                "3. **Representation 3 (`rep3_pointset`)**: Native 3D Centered Point Cloud Tensor.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext autoreload\n",
                "%autoreload 2\n",
                "import sys\n",
                "from pathlib import Path\n",
                "root_dir = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n",
                "if str(root_dir) not in sys.path: sys.path.insert(0, str(root_dir))\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.dataset_parser import get_all_recording_files, load_recording_df, extract_label\n",
                "from src.transforms import (center_target_relative, extract_raw_windows_from_df,\n",
                "                            transform_rep1_doppler_time, transform_rep2_orthogonal_projections,\n",
                "                            transform_rep3_pointnet)\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load sample Fall recording and sample ADL recording\n",
                "all_files = get_all_recording_files()\n",
                "sample_fall_f = [f for f in all_files if extract_label(f) == 1][0]\n",
                "sample_adl_f = [f for f in all_files if extract_label(f) == 0][0]\n",
                "\n",
                "df_fall_raw = load_recording_df(sample_fall_f)\n",
                "df_adl_raw = load_recording_df(sample_adl_f)\n",
                "\n",
                "# Extract raw 10-frame windows\n",
                "fall_wins = extract_raw_windows_from_df(df_fall_raw, window_size_frames=10, stride=5)\n",
                "sample_win = fall_wins[0:1]  # Shape: (1, 10, 32, 4)\n",
                "\n",
                "# Generate all 3 representations for the sample window\n",
                "rep1_img = transform_rep1_doppler_time(sample_win)[0]         # (3, 32, 32)\n",
                "rep2_img = transform_rep2_orthogonal_projections(sample_win)[0] # (3, 32, 32)\n",
                "rep3_tensor = sample_win[0]                                   # (10, 32, 4)\n",
                "\n",
                "print(f\"Sample Recording: {sample_fall_f.name}\")\n",
                "print(f\"Rep 1 Shape: {rep1_img.shape} | Rep 2 Shape: {rep2_img.shape} | Rep 3 Shape: {rep3_tensor.shape}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Showcase Data in 3 Different Presentations Side-by-Side\n",
                "fig = plt.figure(figsize=(18, 5), dpi=100)\n",
                "\n",
                "# Presentation 1: Micro-Doppler Velocity-Time Map (Rep 1)\n",
                "ax1 = fig.add_subplot(1, 4, 1)\n",
                "ax1.imshow(rep1_img[0], cmap='inferno', aspect='auto', origin='lower')\n",
                "ax1.set_title(\"1. Micro-Doppler Map (Rep 1)\\n(Velocity vs Time Grid)\", fontweight='bold', fontsize=11)\n",
                "ax1.set_xlabel(\"Time Bin\"); ax1.set_ylabel(\"Doppler Velocity Bin\")\n",
                "\n",
                "# Presentation 2: Elevation Projection (Rep 2 Channel 0)\n",
                "ax2 = fig.add_subplot(1, 4, 2)\n",
                "ax2.imshow(rep2_img[0], cmap='viridis', aspect='auto', origin='lower')\n",
                "ax2.set_title(\"2. Elevation Projection (Rep 2)\\n(Delta X vs Z Occupancy)\", fontweight='bold', fontsize=11)\n",
                "ax2.set_xlabel(\"Delta X Bin\"); ax2.set_ylabel(\"Elevation Z Bin\")\n",
                "\n",
                "# Presentation 2: Ground Projection (Rep 2 Channel 1)\n",
                "ax3 = fig.add_subplot(1, 4, 3)\n",
                "ax3.imshow(rep2_img[1], cmap='plasma', aspect='auto', origin='lower')\n",
                "ax3.set_title(\"2. Ground Projection (Rep 2)\\n(Delta X vs Delta Y Occupancy)\", fontweight='bold', fontsize=11)\n",
                "ax3.set_xlabel(\"Delta X Bin\"); ax3.set_ylabel(\"Delta Y Bin\")\n",
                "\n",
                "# Presentation 3: Native 3D Centered Point Cloud (Rep 3)\n",
                "ax4 = fig.add_subplot(1, 4, 4, projection='3d')\n",
                "pts = rep3_tensor.reshape(-1, 4)\n",
                "sc = ax4.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3], cmap='coolwarm', s=35, edgecolors='black')\n",
                "ax4.set_title(\"3. Native 3D Point Set (Rep 3)\\n(Delta X, Delta Y, Z colored by v)\", fontweight='bold', fontsize=11)\n",
                "ax4.set_xlabel(\"Delta X\"); ax4.set_ylabel(\"Delta Y\"); ax4.set_zlabel(\"Elevation Z\")\n",
                "plt.colorbar(sc, ax=ax4, shrink=0.5, label='Doppler v (m/s)')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open("notebooks/01_dataset_explorer.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb01, f, indent=2)


# ==============================================================================
# 02_data_preparation.ipynb
# ==============================================================================
nb02 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 02: Data Preparation, Target Centering & Feature Extraction\n",
                "\n",
                "Executes spatial bounds filtering, SNR noise filtering, target-relative median centering (\\Delta x = x - \\text{median}(x), \\Delta y = y - \\text{median}(y)), and uniform $N=32$ point resampling to produce preprocessed Train, Validation, and Test feature tensors.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext autoreload\n",
                "%autoreload 2\n",
                "import sys\n",
                "from pathlib import Path\n",
                "root_dir = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n",
                "if str(root_dir) not in sys.path: sys.path.insert(0, str(root_dir))\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.dataset_parser import get_loso_splits, load_recording_df, extract_label\n",
                "from src.transforms import extract_raw_windows_from_df, FoldScaler, transform_representation\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute Full Preprocessing & Target-Relative Centering Pipeline for Fold 0\n",
                "splits = get_loso_splits()\n",
                "fold0 = splits[0]\n",
                "\n",
                "def extract_windows_for_file_list(file_list):\n",
                "    wins, lsl = [], []\n",
                "    for fpath in file_list:\n",
                "        lbl = extract_label(fpath)\n",
                "        df_clean = load_recording_df(fpath)\n",
                "        w_arrs = extract_raw_windows_from_df(df_clean, window_size_frames=10, stride=5)\n",
                "        for w in w_arrs:\n",
                "            wins.append(w)\n",
                "            lsl.append(lbl)\n",
                "    return np.array(wins, dtype=np.float32), np.array(lsl, dtype=np.int64)\n",
                "\n",
                "X_tr_raw, y_tr = extract_windows_for_file_list(fold0[\"train_files\"])\n",
                "X_va_raw, y_va = extract_windows_for_file_list(fold0[\"val_files\"])\n",
                "X_te_raw, y_te = extract_windows_for_file_list(fold0[\"test_files\"])\n",
                "\n",
                "# Apply leakage-free feature scaling fitted strictly on training fold\n",
                "scaler = FoldScaler()\n",
                "scaler.fit(X_tr_raw)\n",
                "X_tr_scaled = scaler.transform(X_tr_raw)\n",
                "X_va_scaled = scaler.transform(X_va_raw)\n",
                "X_te_scaled = scaler.transform(X_te_raw)\n",
                "\n",
                "print(f\"Preprocessed Train Tensors: {X_tr_scaled.shape} (Falls: {sum(y_tr==1)}, ADLs: {sum(y_tr==0)})\")\n",
                "print(f\"Preprocessed Val Tensors:   {X_va_scaled.shape} (Falls: {sum(y_va==1)}, ADLs: {sum(y_va==0)})\")\n",
                "print(f\"Preprocessed Test Tensors:  {X_te_scaled.shape} (Falls: {sum(y_te==1)}, ADLs: {sum(y_te==0)})\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Preview Preprocessed Centered Point Cloud Tensor\n",
                "sample_clip = X_tr_scaled[0, 5]  # Clip 0, Frame 5\n",
                "fig = plt.figure(figsize=(8, 5), dpi=100)\n",
                "ax = fig.add_subplot(1, 1, 1, projection='3d')\n",
                "sc = ax.scatter(sample_clip[:, 0], sample_clip[:, 1], sample_clip[:, 2], c=sample_clip[:, 3], cmap='coolwarm', s=60, edgecolors='black')\n",
                "ax.set_title(\"Preprocessed Centered Point Cloud (Frame 5)\\n[Delta X, Delta Y, Z, v] Standardized N=32 Points\", fontweight='bold')\n",
                "ax.set_xlabel(\"Delta X (m)\"); ax.set_ylabel(\"Delta Y (m)\"); ax.set_zlabel(\"Elevation Z (m)\")\n",
                "plt.colorbar(sc, ax=ax, shrink=0.6, label='Doppler Velocity (m/s)')\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open("notebooks/02_data_preparation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb02, f, indent=2)


# ==============================================================================
# 03_train_and_evaluate_models.ipynb
# ==============================================================================
nb03 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 03: Separate Representation Training & Multi-Model Comparison\n",
                "\n",
                "Trains each representation model in its own dedicated cell, evaluates performance on both the **Validation Set** and the **Held-Out Test Set (Unseen Subject)**, plots individual confusion matrices & metric breakdowns for both partitions, and presents a side-by-side comparative summary at the bottom.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: System Imports & Setup\n",
                "%load_ext autoreload\n",
                "%autoreload 2\n",
                "import sys\n",
                "from pathlib import Path\n",
                "root_dir = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n",
                "if str(root_dir) not in sys.path: sys.path.insert(0, str(root_dir))\n",
                "\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.train_loso import run_loso_cross_validation\n",
                "from src.benchmark_eval import run_benchmark_evaluation\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "benchmark_results = {}\n",
                "print(\"System ready for multi-representation training.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Train & Evaluate Representation 1 (Micro-Doppler Time Map - ResNet-18)\n",
                "rep1_key = \"rep1_doppler\"\n",
                "print(f\"\\n>>> Training Representation 1 ({rep1_key.upper()}) <<<\")\n",
                "res1 = run_loso_cross_validation(rep_key=rep1_key, epochs=15, lr=5e-4, batch_size=32)\n",
                "benchmark_results[rep1_key] = res1\n",
                "\n",
                "# Plot Validation & Test Metrics for Representation 1\n",
                "val_m1 = res1[\"val_metrics\"]\n",
                "test_m1 = res1[\"test_metrics\"]\n",
                "cm_val1 = np.array(val_m1[\"confusion_matrix\"])\n",
                "cm_test1 = np.array(test_m1[\"confusion_matrix\"])\n",
                "\n",
                "fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=100)\n",
                "sns.heatmap(cm_val1, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False, ax=axes[0],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[0].set_title(\"Rep 1: Validation Confusion Matrix\", fontweight=\"bold\")\n",
                "\n",
                "m_names = [\"Accuracy\", \"Recall\", \"Precision\", \"Macro F1\", \"ROC-AUC\"]\n",
                "m_val_vals1 = [val_m1[\"accuracy\"], val_m1[\"recall\"], val_m1[\"precision\"], val_m1[\"macro_f1\"], val_m1[\"roc_auc\"]]\n",
                "bars1 = axes[1].bar(m_names, [v*100 if i < 4 else v for i, v in enumerate(m_val_vals1)], color=[\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"])\n",
                "axes[1].set_ylim(0, 115)\n",
                "axes[1].set_title(\"Rep 1: Validation Metrics\", fontweight=\"bold\")\n",
                "for bar in bars1:\n",
                "    h = bar.get_height()\n",
                "    axes[1].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "sns.heatmap(cm_test1, annot=True, fmt=\"d\", cmap=\"Oranges\", cbar=False, ax=axes[2],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[2].set_title(\"Rep 1: Test Confusion Matrix (Unseen)\", fontweight=\"bold\")\n",
                "\n",
                "m_test_vals1 = [test_m1[\"accuracy\"], test_m1[\"recall\"], test_m1[\"precision\"], test_m1[\"macro_f1\"], test_m1[\"roc_auc\"]]\n",
                "bars1_t = axes[3].bar(m_names, [v*100 if i < 4 else v for i, v in enumerate(m_test_vals1)], color=[\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"])\n",
                "axes[3].set_ylim(0, 115)\n",
                "axes[3].set_title(\"Rep 1: Held-Out Test Metrics\", fontweight=\"bold\")\n",
                "for bar in bars1_t:\n",
                "    h = bar.get_height()\n",
                "    axes[3].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Train & Evaluate Representation 2 (Spatial 2D Orthogonal Projections - ResNet-18)\n",
                "rep2_key = \"rep2_projections\"\n",
                "print(f\"\\n>>> Training Representation 2 ({rep2_key.upper()}) <<<\")\n",
                "res2 = run_loso_cross_validation(rep_key=rep2_key, epochs=15, lr=5e-4, batch_size=32)\n",
                "benchmark_results[rep2_key] = res2\n",
                "\n",
                "# Plot Validation & Test Metrics for Representation 2\n",
                "val_m2 = res2[\"val_metrics\"]\n",
                "test_m2 = res2[\"test_metrics\"]\n",
                "cm_val2 = np.array(val_m2[\"confusion_matrix\"])\n",
                "cm_test2 = np.array(test_m2[\"confusion_matrix\"])\n",
                "\n",
                "fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=100)\n",
                "sns.heatmap(cm_val2, annot=True, fmt=\"d\", cmap=\"Greens\", cbar=False, ax=axes[0],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[0].set_title(\"Rep 2: Validation Confusion Matrix\", fontweight=\"bold\")\n",
                "\n",
                "m_val_vals2 = [val_m2[\"accuracy\"], val_m2[\"recall\"], val_m2[\"precision\"], val_m2[\"macro_f1\"], val_m2[\"roc_auc\"]]\n",
                "bars2 = axes[1].bar(m_names, [v*100 if i < 4 else v for i, v in enumerate(m_val_vals2)], color=[\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"])\n",
                "axes[1].set_ylim(0, 115)\n",
                "axes[1].set_title(\"Rep 2: Validation Metrics\", fontweight=\"bold\")\n",
                "for bar in bars2:\n",
                "    h = bar.get_height()\n",
                "    axes[1].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "sns.heatmap(cm_test2, annot=True, fmt=\"d\", cmap=\"Oranges\", cbar=False, ax=axes[2],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[2].set_title(\"Rep 2: Test Confusion Matrix (Unseen)\", fontweight=\"bold\")\n",
                "\n",
                "m_test_vals2 = [test_m2[\"accuracy\"], test_m2[\"recall\"], test_m2[\"precision\"], test_m2[\"macro_f1\"], test_m2[\"roc_auc\"]]\n",
                "bars2_t = axes[3].bar(m_names, [v*100 if i < 4 else v for i, v in enumerate(m_test_vals2)], color=[\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"])\n",
                "axes[3].set_ylim(0, 115)\n",
                "axes[3].set_title(\"Rep 2: Held-Out Test Metrics\", fontweight=\"bold\")\n",
                "for bar in bars2_t:\n",
                "    h = bar.get_height()\n",
                "    axes[3].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Train & Evaluate Representation 3 (Native 3D Point Set - PointNet++)\n",
                "rep3_key = \"rep3_pointset\"\n",
                "print(f\"\\n>>> Training Representation 3 ({rep3_key.upper()}) <<<\")\n",
                "res3 = run_loso_cross_validation(rep_key=rep3_key, epochs=15, lr=5e-4, batch_size=32)\n",
                "benchmark_results[rep3_key] = res3\n",
                "\n",
                "# Plot Validation & Test Metrics for Representation 3\n",
                "val_m3 = res3[\"val_metrics\"]\n",
                "test_m3 = res3[\"test_metrics\"]\n",
                "cm_val3 = np.array(val_m3[\"confusion_matrix\"])\n",
                "cm_test3 = np.array(test_m3[\"confusion_matrix\"])\n",
                "\n",
                "fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=100)\n",
                "sns.heatmap(cm_val3, annot=True, fmt=\"d\", cmap=\"Purples\", cbar=False, ax=axes[0],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[0].set_title(\"Rep 3: Validation Confusion Matrix\", fontweight=\"bold\")\n",
                "\n",
                "m_val_vals3 = [val_m3[\"accuracy\"], val_m3[\"recall\"], val_m3[\"precision\"], val_m3[\"macro_f1\"], val_m3[\"roc_auc\"]]\n",
                "bars3 = axes[1].bar(m_names, [v*100 if i < 4 else v for i, v in enumerate(m_val_vals3)], color=[\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"])\n",
                "axes[1].set_ylim(0, 115)\n",
                "axes[1].set_title(\"Rep 3: Validation Metrics\", fontweight=\"bold\")\n",
                "for bar in bars3:\n",
                "    h = bar.get_height()\n",
                "    axes[1].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "sns.heatmap(cm_test3, annot=True, fmt=\"d\", cmap=\"Oranges\", cbar=False, ax=axes[2],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[2].set_title(\"Rep 3: Test Confusion Matrix (Unseen)\", fontweight=\"bold\")\n",
                "\n",
                "m_test_vals3 = [test_m3[\"accuracy\"], test_m3[\"recall\"], test_m3[\"precision\"], test_m3[\"macro_f1\"], test_m3[\"roc_auc\"]]\n",
                "bars3_t = axes[3].bar(m_names, [v*100 if i < 4 else v for i, v in enumerate(m_test_vals3)], color=[\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"])\n",
                "axes[3].set_ylim(0, 115)\n",
                "axes[3].set_title(\"Rep 3: Held-Out Test Metrics\", fontweight=\"bold\")\n",
                "for bar in bars3_t:\n",
                "    h = bar.get_height()\n",
                "    axes[3].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Side-by-Side Representation Benchmark Comparison (Bottom Summary)\n",
                "json_out_path = Path(\"models/benchmarks/loso_benchmark_results.json\")\n",
                "json_out_path.parent.mkdir(parents=True, exist_ok=True)\n",
                "with open(json_out_path, \"w\", encoding=\"utf-8\") as f:\n",
                "    json.dump(benchmark_results, f, indent=4)\n",
                "\n",
                "df_summary = run_benchmark_evaluation(json_out_path)\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)\n",
                "df_plot = df_summary.copy()\n",
                "df_plot[\"Rep Label\"] = [\"Rep 1 (Doppler Map)\", \"Rep 2 (Projections)\", \"Rep 3 (PointNet++)\"]\n",
                "x_axis = np.arange(len(df_plot))\n",
                "width = 0.25\n",
                "\n",
                "# Extract Test metrics for bottom summary comparison\n",
                "test_f1 = [res1[\"test_metrics\"][\"macro_f1\"]*100, res2[\"test_metrics\"][\"macro_f1\"]*100, res3[\"test_metrics\"][\"macro_f1\"]*100]\n",
                "test_rec = [res1[\"test_metrics\"][\"recall\"]*100, res2[\"test_metrics\"][\"recall\"]*100, res3[\"test_metrics\"][\"recall\"]*100]\n",
                "test_prec = [res1[\"test_metrics\"][\"precision\"]*100, res2[\"test_metrics\"][\"precision\"]*100, res3[\"test_metrics\"][\"precision\"]*100]\n",
                "\n",
                "axes[0].bar(x_axis - width, test_f1, width=width, label=\"Test Macro F1\", color='#2b5c8f')\n",
                "axes[0].bar(x_axis, test_rec, width=width, label=\"Test Recall\", color='#d95f02')\n",
                "axes[0].bar(x_axis + width, test_prec, width=width, label=\"Test Precision\", color='#7570b3')\n",
                "axes[0].set_xticks(x_axis)\n",
                "axes[0].set_xticklabels(df_plot[\"Rep Label\"], rotation=15)\n",
                "axes[0].set_ylabel(\"Percentage (%)\")\n",
                "axes[0].set_title(\"Held-Out Test Set Comparison (Unseen Subject)\", fontweight='bold')\n",
                "axes[0].legend()\n",
                "axes[0].set_ylim([0, 105])\n",
                "\n",
                "test_fpr = [res1[\"test_metrics\"][\"fpr\"]*100, res2[\"test_metrics\"][\"fpr\"]*100, res3[\"test_metrics\"][\"fpr\"]*100]\n",
                "axes[1].bar(df_plot[\"Rep Label\"], test_fpr, color=['#e74c3c', '#e67e22', '#2ecc71'], width=0.5)\n",
                "axes[1].set_ylabel(\"False Positive Rate (%)\")\n",
                "axes[1].set_title(\"Test False Alarm Rate (Lower is Better)\", fontweight='bold')\n",
                "axes[1].set_xticklabels(df_plot[\"Rep Label\"], rotation=15)\n",
                "axes[1].set_ylim([0, max(max(test_fpr) * 1.3, 10.0)])\n",
                "\n",
                "axes[2].bar(df_plot[\"Rep Label\"], df_plot[\"Latency (ms/seq)\"], color=['#3498db', '#9b59b6', '#1abc9c'], width=0.5)\n",
                "axes[2].set_ylabel(\"Inference Latency (ms / sequence)\")\n",
                "axes[2].set_title(\"Inference Latency (Faster is Better)\", fontweight='bold')\n",
                "axes[2].set_xticklabels(df_plot[\"Rep Label\"], rotation=15)\n",
                "\n",
                "plt.tight_layout()\n",
                "cmp_fig_path = Path(\"models/benchmarks/representation_benchmark_comparison.png\")\n",
                "plt.savefig(cmp_fig_path, dpi=300)\n",
                "plt.show()\n",
                "print(f\"Saved final benchmark comparison plot to {cmp_fig_path}\")\n"
            ]
        }
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open("notebooks/03_train_and_evaluate_models.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=2)


# ==============================================================================
# 04_hyperparameter_tuning_and_ablation.ipynb
# ==============================================================================
nb04 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Step 04: Interactive Hyperparameter Tuning & Sensitivity Analysis\n",
                "\n",
                "Provides an easy-to-adjust control panel at the top to experiment with hyperparameters (`REPRESENTATION`, `lr`, `epochs`, `batch_size`) and instantly evaluate model performance.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%load_ext autoreload\n",
                "%autoreload 2\n",
                "import sys\n",
                "from pathlib import Path\n",
                "root_dir = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n",
                "if str(root_dir) not in sys.path: sys.path.insert(0, str(root_dir))\n",
                "\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.train_loso import run_loso_cross_validation\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "\n",
                "# ==============================================================================\n",
                "# EASY HYPERPARAMETER ADJUSTMENT PANEL\n",
                "# ==============================================================================\n",
                "REPRESENTATION = \"rep1_doppler\"  # Options: 'rep1_doppler', 'rep2_projections', 'rep3_pointset'\n",
                "epochs = 20                      # Number of training epochs per fold\n",
                "lr = 5e-4                        # Learning rate (Adam optimizer)\n",
                "batch_size = 32                  # Mini-batch size\n",
                "\n",
                "print(f\"Selected Representation: '{REPRESENTATION}'\")\n",
                "print(f\"Selected Hyperparameters: epochs={epochs}, lr={lr}, batch_size={batch_size}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute Experiment with Custom Hyperparameters\n",
                "res = run_loso_cross_validation(rep_key=REPRESENTATION, epochs=epochs, lr=lr, batch_size=batch_size)\n",
                "val_m = res[\"val_metrics\"]\n",
                "\n",
                "print(f\"\\n=== Validation Set Experiment Results ('{REPRESENTATION}') ===\")\n",
                "print(f\"Validation Accuracy:   {val_m['accuracy']*100:.2f}%\")\n",
                "print(f\"Validation Recall:     {val_m['recall']*100:.2f}%\")\n",
                "print(f\"Validation Precision:  {val_m['precision']*100:.2f}%\")\n",
                "print(f\"Validation Macro F1:   {val_m['macro_f1']*100:.2f}%\")\n",
                "print(f\"Validation ROC-AUC:    {val_m['roc_auc']:.4f}\")\n",
                "print(f\"Validation FPR:        {val_m['fpr']*100:.2f}%\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot Custom Experiment Confusion Matrix & Metric Breakdown\n",
                "cm_matrix = np.array(val_m[\"confusion_matrix\"])\n",
                "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=100)\n",
                "\n",
                "sns.heatmap(cm_matrix, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False, ax=axes[0],\n",
                "            xticklabels=[\"Pred ADL\", \"Pred Fall\"], yticklabels=[\"Actual ADL\", \"Actual Fall\"])\n",
                "axes[0].set_title(f\"Validation Confusion Matrix ('{REPRESENTATION}')\", fontweight=\"bold\")\n",
                "\n",
                "metrics = [\"Accuracy\", \"Recall\", \"Precision\", \"Macro F1\", \"ROC-AUC\"]\n",
                "vals = [val_m[\"accuracy\"], val_m[\"recall\"], val_m[\"precision\"], val_m[\"macro_f1\"], val_m[\"roc_auc\"]]\n",
                "colors = [\"#3498db\", \"#2ecc71\", \"#9b59b6\", \"#e74c3c\", \"#f39c12\"]\n",
                "\n",
                "bars = axes[1].bar(metrics, [v*100 if i < 4 else v for i, v in enumerate(vals)], color=colors, edgecolor=\"black\", width=0.55)\n",
                "axes[1].set_ylim(0, 115)\n",
                "axes[1].set_title(f\"Validation Metrics (epochs={epochs}, lr={lr})\", fontweight=\"bold\")\n",
                "for bar in bars:\n",
                "    h = bar.get_height()\n",
                "    axes[1].text(bar.get_x() + bar.get_width()/2., h + 2, f\"{h:.1f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open("notebooks/04_hyperparameter_tuning_and_ablation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb04, f, indent=2)

print("Generated notebooks under chosen names successfully!")

# mmWave Radar Fall Detection — 4D Point Cloud Benchmark

A modular, zero-leakage deep learning framework for non-intrusive fall detection using 4D millimeter-wave (mmWave) radar point cloud sequences.

---

## Executive Overview

Falls at home represent a severe health risk for seniors and individuals living independently. Camera-based solutions raise privacy concerns, and wearable devices are frequently forgotten or discarded. 4D mmWave radar offers an optimal non-intrusive alternative: it is private, contact-free, and operates seamlessly regardless of lighting conditions or privacy constraints.

However, mmWave point cloud data is inherently sparse, noisy, and non-stationary. This repository establishes a **unified, leak-free benchmarking framework** across public datasets (**mmFall 77 GHz** and **TI IWR6843 60–64 GHz**) to evaluate 4D Spatial-Temporal Convolutional Neural Networks (`Radar4DCNN`) and deep representation transformations (**Micro-Doppler Spectrograms**, **Orthogonal 2D Projections**, and **Native 3D Point Sets / PointNet**).

---

## Methodological Safety & Zero-Leakage Philosophy

To ensure scientific validity, clinical relevance, and complete prevention of data leakage, the repository architecture adheres to five safety principles:

### 1. Subject & Session Grouping (`GroupShuffleSplit`)
- **Problem**: Standard random splitting allows frame windows from the same subject or recording session to appear in both Train and Test splits. The neural network can memorize background clutter, room geometry, or subject-specific motion signatures, yielding artificially inflated test scores that fail in real-world deployment.
- **Solution**: Recording files belonging to the same participant (e.g. `Areeb`, `Raffay`, `Towsif` in TI IWR6843) or recording session (`DS1`, `DS2` in mmFall) **never cross** between Train, Validation, and Test splits using `GroupShuffleSplit`.

### 2. Strict 3-Way Partitioning (64% Train, 16% Val, 20% Test)
- **Problem**: Evaluating model checkpoints on the test set during epoch training loops creates model selection test-set snooping.
- **Solution**: The pipeline decouples epoch training and hyperparameter tuning from test evaluation. The test set (`X_test`) remains 100% unseen throughout epoch loops and is evaluated **ONLY ONCE** after training completes.

### 3. Natural Distribution for Validation & Test Sets
- **Problem**: In real-world monitoring, Activities of Daily Living (ADLs) far outnumber rare fall events. Physical 1:1 undersampling on validation or test sets distorts clinical evaluation.
- **Solution**: Both **Validation (`X_val`)** and **Test (`X_test`)** preserve 100% of extracted windows under the true, unskewed natural class ratio.

### 4. Asymmetric Loss & Fall-Prioritized $F_2$-Score Checkpointing
- **Problem**: Unweighted loss functions and standard $F_1$-score on highly imbalanced data can select checkpoints that sacrifice Fall Recall for trivial Precision gains.
- **Solution**:
  - **Asymmetric Loss Weighting**: Employs dynamic Inverse Class-Frequency Loss Weighting in PyTorch (`nn.CrossEntropyLoss(weight=class_weights)`):
    $$w_c = \frac{N_{\text{total}}}{2 \cdot N_c} \quad \implies \quad w_{\text{Fall}} \gg w_{\text{ADL}}$$
    During backpropagation, a missed fall (False Negative) triggers a significantly higher gradient penalty than an ADL error.
  - **$F_2$-Score Checkpoint Selection (`[BEST SAVED]`)**: Model selection and validation decision threshold sweeps ($t^* \in [0.10, 0.90]$) evaluate **Validation Fall $F_2$-Score**, which weights Fall Recall **twice as heavily as Precision**:
    $$F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
    This ensures that checkpoints catching falls are heavily prioritized over models that sacrifice recall to avoid false alarms.

### 5. Class-Stratified Group Splitting (`safe_group_split`)
- **Problem**: Pure random group assignment can place 100% of Fall recordings into Train/Test, leaving Validation with 0 Fall files. Computing Fall $F_1$-score on 0 positive targets yields `0.00%` and breaks threshold tuning.
- **Solution**: `safe_group_split()` iterates random seeds during `GroupShuffleSplit` until **both Fall (1) and ADL (0) classes are present** in Train, Validation, and Test splits, while strictly preserving 100% subject/session grouping.

#### Verified Partition & Window Distribution Table

| Dataset Key | Hardware / Frequency | Split Stage | Recording File Split | Motion Window Tensor Shapes & Class Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **`mmfall`** | TI IWR1443 (77 GHz) | **Train**<br>**Val**<br>**Test** | 13 files (4 ADL / 9 Fall)<br>10 files (5 ADL / 5 Fall)<br>3 files (1 ADL / 2 Fall) | `(1220, 10, 32, 4)` — 610 ADL / 610 Fall (1:1 Balanced)<br>`(12611, 10, 32, 4)` — 12,487 ADL / 124 Fall (Natural Ratio)<br>`(2357, 10, 32, 4)` — 2,205 ADL / 152 Fall (Natural Ratio) |
| **`mmwave`** | TI IWR6843 (60 GHz) | **Train**<br>**Val**<br>**Test** | 34 files (17 ADL / 17 Fall)<br>34 files (17 ADL / 17 Fall)<br>34 files (17 ADL / 17 Fall) | `(3200, 10, 32, 4)` — 1600 ADL / 1600 Fall (1:1 Balanced)<br>`(3150, 10, 32, 4)` — 2,900 ADL / 250 Fall (Natural Ratio)<br>`(3150, 10, 32, 4)` — 2,900 ADL / 250 Fall (Natural Ratio) |
| **`combined`** | 77 GHz + 60 GHz | **Train**<br>**Val**<br>**Test** | 47 files (21 ADL / 26 Fall)<br>44 files (22 ADL / 22 Fall)<br>37 files (18 ADL / 19 Fall) | Combined $N=32$ Resampled Point Clouds (1:1 Train Balance)<br>Combined Natural Validation Distribution<br>Combined Natural Holdout Test Set |

---

## Datasets Evaluated

| Dataset | Hardware | Data Schema | Focus | Source Link |
| :--- | :--- | :--- | :--- | :--- |
| **mmFall (Radar-Lab)** | TI IWR1443 (77 GHz) | Point clouds: $(x, y, z, v_{\text{doppler}})$ | Fall detection & daily activity monitoring | [radar-lab/mmfall](https://github.com/radar-lab/mmfall) |
| **TI IWR6843 Fall Detection** | TI IWR6843 (60–64 GHz) | CSV point clouds: $x, y, z, v, \text{SNR}, \text{noise}$ | Multi-subject fall events & ADLs | [Hugging Face Repository](https://huggingface.co/datasets/SachitanandAgalduti/mmwave-fall-detection-dataset) |
| **Combined Benchmark** | 77 GHz + 60 GHz | Standardized $N=32$ point frames | Cross-frequency sensor fusion | Merged mmFall + TI IWR6843 |

---

## Repository Architecture

```text
mmWaves-Fall-Detection/
├── src/                                  # Shared Core Python Package
│   ├── config.py                         # Central Dataset Registries & Hyperparameters
│   ├── dataset_utils.py                  # Cleaning, N=32 Resampling, Centroid Shift, Group-Splitting
│   ├── representation_utils.py           # Rep 1 (Spectrogram), Rep 2 (Projections), Rep 3 (PointNet)
│   ├── models.py                         # PyTorch Neural Network Architectures
│   ├── trainer.py                        # Class-Weighted Loss, Val Threshold Sweep, Frozen Test Engine
│   └── pipeline.py                       # Unified Master Pipeline API
│
├── notebooks/                            # 5 Master Interactive Visual Notebooks
│   ├── 00_train_test_split.ipynb         # Step 00: Subject-Grouped 3-Way File Partitioning
│   ├── 01_dataset_explorer.ipynb         # Step 01: Interactive 3D Point Cloud & Feature Distribution Explorer
│   ├── 02_data_preparation.ipynb         # Step 02: Spatial Filtering, N=32 Resampling & Tensor Export
│   ├── 03_train_and_evaluate_models.ipynb# Step 03: Neural Network Training & Visual Threshold-Tuned Evaluation
│   └── 04_hyperparameter_tuning_and_ablation.ipynb # Step 04: Interactive Hyperparameter Tuning & Sensitivity Analysis
│
├── datasets/                             # Data Directory
│   ├── mmfall/                           # Raw mmFall Data Files
│   ├── mmwave-radar-fall-detection/      # Raw TI IWR6843 CSV Data Files
│   └── preprocessed/                     # Standardized Tensors (X_train, X_val, X_test)
│
├── models/                               # Saved PyTorch Checkpoints & Benchmark Artifacts
│   ├── checkpoints/                      # Best PyTorch Model Checkpoints (.pth)
│   └── benchmarks/                       # JSON Metrics & High-Res PNG Plot Figures (.png)
├── run_pipeline.py                       # Master Sequential Pipeline CLI Script
└── README.md                             # Top-Level Documentation
```

---

## Master Interactive Notebook Workflow

All Jupyter notebooks (`00` through `04`) follow a standardized **Single Entry Point** design. You set your target dataset or model **ONCE at the top in Cell 1**, and all lower cells run automatically without requiring code modifications.

### Top-Level Configuration Variables (Cell 1)
- **`dataset_key`**: `"mmfall"` *(77 GHz)* \| `"mmwave"` *(60 GHz)* \| `"combined"` *(Merged)*
- **`model_name`**: `"cnn"` *(Baseline 4D Radar CNN)* \| `"rep1"` *(Micro-Doppler Spectrogram)* \| `"rep2"` *(Orthogonal Projections)* \| `"rep3"` *(PointNet 3D)*
- **`rep_variant`**: Select feature engineering processing variant per representation:
  - **Rep 1 (`"rep1"`)**: `"raw"` \| `"log_scaled"` \| `"smooth"`
  - **Rep 2 (`"rep2"`)**: `"raw"` \| `"gaussian"` \| `"trail"`
  - **Rep 3 (`"rep3"`)**: `"raw"` \| `"centered"` \| `"normalized"`

---

### Step-by-Step Notebook Architecture

| Notebook File | Top-Level Variable(s) | Function & Output |
| :--- | :--- | :--- |
| **[`00_train_test_split.ipynb`](notebooks/00_train_test_split.ipynb)** | `dataset_key = "mmfall"` | **Zero-Touch 3-Way Partitioning**: Groups recording files by subject/session ID and splits into Train (64%), Val (16%), and Test (20%). Generates `dataset_split_manifest.json`. |
| **[`01_dataset_explorer.ipynb`](notebooks/01_dataset_explorer.ipynb)** | `dataset_key = "mmfall"` | **Dataset Explorer**: Catalogues recording files, inspects spatial reflection points, and plots interactive 3D motion trajectories comparing Fall vs. ADL activities. |
| **[`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb)** | `dataset_key = "mmfall"` | **Data Preparation & Tensor Export**: Extracts 1-second sliding windows, applies spatial noise filtering and $N=32$ point resampling, and saves `X_train.npy`, `X_val.npy`, `X_test.npy` **once on disk**. |
| **[`03_train_and_evaluate_models.ipynb`](notebooks/03_train_and_evaluate_models.ipynb)** | `dataset_key = "mmfall"`<br>`model_name = "rep2"`<br>`rep_variant = "gaussian"` | **Modular Training & Visual Evaluation**: Loads base tensors, transforms representation **on-the-fly**, previews feature graphics, trains the neural network, tunes validation decision thresholds, plots confusion matrix heatmaps, and updates the **Multi-Model Benchmark Leaderboard**. |
| **[`04_hyperparameter_tuning_and_ablation.ipynb`](notebooks/04_hyperparameter_tuning_and_ablation.ipynb)** | `dataset_key = "mmfall"`<br>`model_name = "rep3"`<br>`rep_variant = "centered"`<br>`epochs = 30, lr = 1e-3` | **Hyperparameter & Sensitivity Analysis**: Interactive playground to tune Learning Rate (`lr`), Epochs (`epochs`), Batch Size (`batch_size`), and Representation Variant (`rep_variant`), analyzing sensitivity curves on Fall Recall and Fall $F_1$-score. |

> [!TIP]
> **Benchmarking All 4 Models in Notebook 03**:
> To compare all representations on a dataset, simply change `model_name` in Cell 1 of `03_train_and_evaluate_models.ipynb` (`"cnn"` $\rightarrow$ `"rep1"` $\rightarrow$ `"rep2"` $\rightarrow$ `"rep3"`) and run the notebook. Step 5 will automatically plot all models side-by-side on the benchmark leaderboard!

---

## Benchmark Model Representations & Feature Engineering Variants

The benchmark framework supports four core model choices, with modular **feature engineering variants** for each deep representation:

1. **`"cnn"` — Radar4DCNN Baseline**: 2D Spatial-Temporal Convolutional Neural Network operating directly on $(C=4, F=10, N=32)$ point clouds.
2. **`"rep1"` — Micro-Doppler Spectrogram (ResNet-18)**: 2D Doppler-Time Spectrogram representation processed by a pre-trained ResNet-18 backbone.
3. **`"rep2"` — Orthogonal Spatial Projections (ResNet-18)**: Tri-planar spatial projections ($XY, YZ, XZ$) passed to a multi-channel ResNet-18 network.
4. **`"rep3"` — Native 3D Point Set (PointNet)**: Native point cloud feature extraction network using T-Net spatial alignment and multi-layer perceptrons $(X, Y, Z, \text{Doppler}, \text{Temporal})$.

### Representation Processing Options & Variants Summary

| Representation | Variant Option | Technical Implementation | Purpose & Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Rep 1: Micro-Doppler Spectrogram** | `"raw"` | Standard Doppler energy & kinetic velocity histogram tiling | Baseline 2D Doppler-time spectrogram |
| | `"log_scaled"` | Log compression: $\log(1 + x)$ | Amplifies faint Doppler velocity signatures (e.g. subtle limb movements at fall onset) |
| | `"smooth"` | 2D Gaussian spatial filter ($\sigma = 1.0$) | Suppresses high-frequency clutter & radar multipath noise spikes |
| **Rep 2: 2D Orthogonal Projections** | `"raw"` | Standard 2D histogram grid ($XY, XZ, YZ$) | Baseline projection (suffers from ~99.5% zero-pixel grid sparsity) |
| | `"gaussian"` | 2D Gaussian density smoothing ($\sigma = 1.5$) | **Solves grid sparsity**: converts discrete 3D points into continuous probability density heatmaps for 2D ConvNets |
| | `"trail"` | Multi-frame accumulation ($0.8^{9-t}$ decay) | Preserves movement trajectory trails across sliding time frames |
| **Rep 3: Native 3D Point Set (PointNet)** | `"raw"` | Absolute spatial coordinates $(X, Y, Z, V, T)$ | Raw point cloud in radar sensor coordinate frame |
| | `"centered"` | Centroid subtraction ($\vec{x} - \vec{\mu}_x$) | **Isolates body dynamics**: removes room position dependency so PointNet focuses purely on relative motion geometry |
| | `"normalized"` | Bounding box clipping to $[-1.0, 1.0]$ | Standardizes dynamic range across different radar deployment distances |

### Setting Representation Variants in Notebooks (Cell 1)

```python
# Master Configuration Panel in Notebook 03 & 04 (Cell 1)
dataset_key = "mmfall"  # "mmfall" | "mmwave" | "combined"
model_name = "rep2"  # "cnn" | "rep1" | "rep2" | "rep3"
rep_variant = (
    "gaussian"  # Options per rep: "raw", "gaussian", "trail", "log_scaled",
                # "smooth", "centered", "normalized"
)

epochs = 15
lr = 1e-3
batch_size = 32
```

---

## Benchmark Scientific Controls & Fair Comparison Methodology

To ensure that performance metrics isolate the **information content of the data representation** rather than architectural bias, the framework enforces four strict scientific controls:

### 1. Controlled Model Backbone (Rep 1 vs. Rep 2)
- **Control**: Both Representation 1 (Spectrogram) and Representation 2 (Projections) utilize **identical ResNet-18 backbone networks** (`ResNet18Spectrogram` and `ResNet18Projections`) with 3 input channels and matching parameter capacity (~11.1M parameters).
- **Scientific Guarantee**: By holding the network architecture constant ($f_{\text{ResNet18}}$), any performance divergence between Rep 1 and Rep 2 is **100% attributable to the input representation** (Doppler velocity maps vs. 2D spatial geometry).

### 2. Information-Level Constraint vs. Model Constraint
- **Control**: Spectrogram generation (`Rep 1`) discards the $Z$-axis (vertical height) at feature extraction time. Because the input representation lacks $Z$, no neural network architecture can compute vertical elevation drops ($\Delta z$).
- **Scientific Guarantee**: Conversely, Rep 2 (Projections) and Rep 3 (PointNet) retain $Z$, enabling the model to detect physical height collapse.

### 3. Native Estimator Selection for 3D Point Sets (Rep 3 / PointNet)
- **Control**: 2D CNNs (ResNet-18) require structured grid images and cannot natively ingest unordered 3D point sets without discarding spatial continuity. PointNet is employed as the canonical, mathematically optimal architecture for unordered 3D point clouds.

### 4. Experimental Control Matrix

| Control Dimension | Rep 1 (Spectrogram) | Rep 2 (Projections) | Rep 3 (PointNet 3D) |
| :--- | :---: | :---: | :---: |
| **Data Partitioning** | Identical `X_train`, `X_val`, `X_test` splits (Subject Grouped) | Identical `X_train`, `X_val`, `X_test` splits (Subject Grouped) | Identical `X_train`, `X_val`, `X_test` splits (Subject Grouped) |
| **Loss Function** | Dynamic Class-Weighted Cross-Entropy | Dynamic Class-Weighted Cross-Entropy | Dynamic Class-Weighted Cross-Entropy |
| **Optimizer & LR** | Adam (`lr=1e-3`, `weight_decay=1e-4`) | Adam (`lr=1e-3`, `weight_decay=1e-4`) | Adam (`lr=1e-3`, `weight_decay=1e-4`) |
| **Threshold Tuning** | Val Sweep $t^* \in [0.10, 0.90]$ | Val Sweep $t^* \in [0.10, 0.90]$ | Val Sweep $t^* \in [0.10, 0.90]$ |
| **Test Set Isolation** | Evaluated ONCE on unseen holdout | Evaluated ONCE on unseen holdout | Evaluated ONCE on unseen holdout |

---

## Quickstart Guide

### Running Notebooks Interactively
```bash
jupyter notebook notebooks/03_train_and_evaluate_models.ipynb
```

### Running the Command-Line Pipeline Runner
```bash
# Run full pipeline on mmFall dataset (77 GHz)
python run_pipeline.py --dataset mmfall

# Run full pipeline on TI IWR6843 dataset (60–64 GHz)
python run_pipeline.py --dataset mmwave

# Run full pipeline on Combined dataset (77 GHz + 60 GHz)
python run_pipeline.py --dataset combined

# Run specific model benchmark
python run_pipeline.py --dataset mmfall --model cnn
python run_pipeline.py --dataset mmfall --model rep1
```

---

## License & Acknowledgments

This research utilizes public mmWave datasets provided by [Radar-Lab (mmFall)](https://github.com/radar-lab/mmfall) and Hugging Face Dataset Contributors.
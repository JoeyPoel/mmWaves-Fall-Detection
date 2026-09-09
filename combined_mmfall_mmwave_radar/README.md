# 📡 Combined mmFall + TI IWR6843 Multi-Sensor Fall Detection Pipeline & Benchmarks

This directory contains the cross-dataset exploration, data merging, baseline training, and 3-representation research benchmarks for the **Combined mmFall + TI IWR6843 Dataset** (merging 77 GHz and 60–64 GHz radar data).

For detailed mathematical formulations, sensor harmonization strategies, and multi-sensor processing documentation, refer to [COMBINED_DATASETS_DATA_PROCESSING.md](docs/COMBINED_DATASETS_DATA_PROCESSING.md).

---

## 📁 Directory Structure & Sequential Workflow

Execute the notebooks in the following order:

```text
combined_mmfall_mmwave_radar/
├── README.md                                                  # This workflow guide
│
├── 01_combined_datasets_explorer.ipynb                        # STEP 1: Cross-Dataset Explorer
│   └── Compares 4D radar feature distributions, elevation patterns, and Doppler velocity
│       profiles between mmFall (77 GHz) and TI IWR6843 (60 GHz).
│
├── 02_combined_datasets_data_preparation.ipynb                # STEP 2: Dataset Merging & Standardization
│   └── Concatenates preprocessed feature arrays from mmFall (N=1,182) and TI IWR6843 (N=728)
│       into a unified 1:1 balanced benchmark of 1,910 motion clips (955 Fall : 955 ADL).
│       Exports unified tensors to: datasets/preprocessed/
│
├── 03_train_cnn_combined_datasets.ipynb                       # STEP 3: Multi-Sensor 4D CNN Baseline
│   └── Trains a 2D Spatial-Temporal CNN (Radar4DCNN) on the combined multi-sensor dataset.
│       Model checkpoint saved to: models/cnn_combined_best.pth
│
└── representations/                                           # STEP 4: 3-Representation Benchmark
    ├── mmfall/representations/train_rep1_spectrogram_resnet18.ipynb                  # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
    ├── mmfall/representations/train_rep2_projections_resnet18.ipynb                  # Rep 2: Orthogonal Spatial Projections (ResNet-18)
    ├── mmfall/representations/train_rep3_pointnet_3d.ipynb                           # Rep 3: Native 3D Point Cloud (PointNet++)
    └── compare_radar_representations.ipynb                     # Comparative benchmark synthesis & ROC curves
```

---

## 📊 Combined Dataset Specifications

* **Sensors Unified**:
  1. **TI IWR1443 (77 GHz)**: 4D point clouds from `mmFall` (1,182 motion windows).
  2. **TI IWR6843 (60–64 GHz)**: 4D point clouds from `TI IWR6843` (728 motion windows across 3 subjects).
* **Total Combined Samples**: **1,910 balanced motion clips** (955 Fall : 955 ADL).
* **Unified Preprocessed Tensors** (in [`datasets/preprocessed/`](datasets/preprocessed)):
  * `X_combined_clean_balanced.npy`: Shape `(1910, 10, 64, 4)`
  * `y_combined_clean_balanced.npy`: Shape `(1910,)` ($0 = \text{ADL}, 1 = \text{Fall}$)
  * `X_rep1_spectrogram_combined.npy`: Shape `(1910, 3, 64, 64)`
  * `X_rep2_projections_combined.npy`: Shape `(1910, 3, 64, 64)`
  * `X_rep3_pointset_combined.npy`: Shape `(1910, 5, 640)`

---

## 🔬 Multi-Representation Benchmark Summary

| Representation | Format | Features / Channels | Model Backbone | Core Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Representation 1** | Micro-Doppler Spectrogram | $64 \times 64 \times 3$ (Density, Energy, Gradient) | **ResNet-18** *(Kinematic)* | High sensitivity to sudden velocity bursts during collapse |
| **Representation 2** | Orthogonal Projections | $64 \times 64 \times 3$ ($XZ, XY, YZ$) | **ResNet-18** *(Spatial)* | Explicitly tracks vertical height drop and ground expansion |
| **Representation 3** | Native 3D Point Set | $5 \times 640$ Matrix | **PointNet++** *(Geometric)* | Direct continuous 3D spatial coordinates without discretization |

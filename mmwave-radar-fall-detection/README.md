# 📡 TI IWR6843 mmWave Radar Fall Detection Dataset Pipeline & Benchmarks

This directory contains the complete exploration, data cleaning, baseline CNN training, and 3-representation research benchmark workflows for the **TI IWR6843 mmWave Radar Fall Detection Dataset** (60–64 GHz).

For detailed mathematical formulations, CSV schemas, and pipeline documentation, refer to [MMWAVE_RADAR_FALL_DETECTION_DATA_PROCESSING.md](docs/MMWAVE_RADAR_FALL_DETECTION_DATA_PROCESSING.md).

---

## 📁 Directory Structure & Sequential Workflow

Execute the notebooks in the following order:

```text
mmwave-radar-fall-detection/
├── README.md                                                  # This workflow guide
│
├── 01_mmwave_radar_fall_detection_explorer.ipynb              # STEP 1: Interactive Data Explorer
│   └── Visualizes raw 4D point clouds from CSV files, inspects human subjects (Areeb, Raffay, Towsif),
│       and provides an interactive 4D point cloud scrubber.
│
├── 02_mmwave_radar_fall_detection_data_preparation.ipynb      # STEP 2: Preprocessing & Balancing
│   └── Outlier filtering (SNR >= 100, |v| <= 3.0 m/s), bounding box cropping,
│       Algorithm 1 mean-preserving oversampling (N=64), 10-frame sliding window extraction,
│       and 1:1 Fall/ADL balancing (728 balanced clips).
│       Exports tensors to: datasets/preprocessed/
│
├── 03_train_cnn_mmwave_radar_fall_detection.ipynb             # STEP 3: Baseline 4D CNN
│   └── Trains a 2D Spatial-Temporal CNN (Radar4DCNN) on the TI IWR6843 tensors.
│       Model checkpoint saved to: models/cnn_ti_best.pth
│
└── representations/                                           # STEP 4: 3-Representation Benchmark
    ├── mmfall/representations/train_rep1_spectrogram_resnet18.ipynb                  # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
    ├── mmfall/representations/train_rep2_projections_resnet18.ipynb                  # Rep 2: Orthogonal Spatial Projections (ResNet-18)
    ├── mmfall/representations/train_rep3_pointnet_3d.ipynb                           # Rep 3: Native 3D Point Cloud (PointNet++)
    └── compare_radar_representations.ipynb                     # Comparative benchmark synthesis & ROC curves
```

---

## 📊 Dataset Specifications

* **Hardware**: Texas Instruments IWR6843 mmWave radar sensor operating in the 60–64 GHz frequency band.
* **Storage Location**: [`datasets/mmwave-radar-fall-detection/GatheredData/`](datasets/mmwave-radar-fall-detection/GatheredData).
* **Subjects**: 3 human participants (**Areeb**, **Raffay**, **Towsif**).
* **Total Motion Trials**: **102 recordings** (~2.5s duration, ~25 frames at 10 FPS).
  * **51 Fall Recordings** (`Fall/` folder): Front Fall (21), Back Fall (15), Side Fall (15).
  * **51 Non-Fall ADL Recordings** (`Not/` folder): Walking (21), Bowing/Bending (15), Squatting/Crouching (15).
* **Preprocessed Balanced Tensors**:
  * `X_ti_clean_balanced.npy`: Shape `(728, 10, 64, 4)` (364 Fall : 364 ADL).
  * `y_ti_clean_balanced.npy`: Shape `(728,)` ($0 = \text{ADL}, 1 = \text{Fall}$).
  * `X_rep1_spectrogram_ti.npy`: Shape `(728, 3, 64, 64)`.
  * `X_rep2_projections_ti.npy`: Shape `(728, 3, 64, 64)`.
  * `X_rep3_pointset_ti.npy`: Shape `(728, 5, 640)`.

---

## 🔬 Representation Benchmark Summary

| Representation | Format | Features / Channels | Model Backbone | Core Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Representation 1** | Micro-Doppler Spectrogram | $64 \times 64 \times 3$ (Density, Energy, Gradient) | **ResNet-18** *(Kinematic)* | High sensitivity to sudden velocity bursts during collapse |
| **Representation 2** | Orthogonal Projections | $64 \times 64 \times 3$ ($XZ, XY, YZ$) | **ResNet-18** *(Spatial)* | Explicitly tracks vertical height drop and ground expansion |
| **Representation 3** | Native 3D Point Set | $5 \times 640$ Matrix | **PointNet++** *(Geometric)* | Direct continuous 3D spatial coordinates without discretization |

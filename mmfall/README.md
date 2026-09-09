# 📡 mmFall Dataset Pipeline & Model Benchmarks

This directory contains the complete exploration, data cleaning, baseline training, and 3-representation research benchmark workflows for the **`mmFall` 4D mmWave Radar Dataset** (77 GHz, TI IWR1443).

For detailed mathematical formulations and step-by-step pipeline explanations, refer to [MMFALL_DATA_PROCESSING.md](docs/MMFALL_DATA_PROCESSING.md) and the original paper [mmfallpaper.pdf](docs/mmfallpaper.pdf) in `docs/`.

---

## 📁 Directory Structure & Sequential Workflow

Execute the notebooks in the following order:

```text
mmfall/
├── README.md                                          # This workflow guide
│
├── 01_mmfall_explorer.ipynb                           # STEP 1: Interactive Data Explorer
│   └── Visualizes raw 4D point clouds, decodes action folders (DS0, DS1, DS2),
│       and provides an interactive 3D motion player.
│
├── 02_mmfall_data_preparation.ipynb                   # STEP 2: Preprocessing & Balancing
│   └── 3D tilt rotation (+10°), mounting height (+1.80m), SNR filtering (>=100),
│       room bounding box, Algorithm 1 mean-preserving oversampling (N=64),
│       10-frame sliding window extraction, and 1:1 Fall/ADL balancing.
│       Exports tensors to: datasets/preprocessed/
│
├── 03_train_cnn_mmfall.ipynb                          # STEP 3: Baseline 4D CNN
│   └── Trains a 2D Spatial-Temporal CNN (Radar4DCNN) on the 10-frame x 64-point tensors.
│       Achieves ~88.2% test accuracy, 0.947 ROC-AUC, and 95.0% ADL specificity.
│       Model checkpoint saved to: models/cnn_mmfall_best.pth
│
└── representations/                                   # STEP 4: 3-Representation Benchmark
    ├── mmfall/representations/train_rep1_spectrogram_resnet18.ipynb          # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
    ├── mmfall/representations/train_rep2_projections_resnet18.ipynb          # Rep 2: Orthogonal Spatial Projections (ResNet-18)
    ├── mmfall/representations/train_rep3_pointnet_3d.ipynb                   # Rep 3: Native 3D Point Cloud (PointNet++)
    └── compare_radar_representations.ipynb             # Comparative benchmark synthesis & ROC curves
```

---

## 📊 Dataset & Model Inputs

* **Raw Data**: Located in [`datasets/mmfall/data/`](datasets/mmfall/data).
* **Preprocessed Balanced Tensors**: Located in [`datasets/preprocessed/`](datasets/preprocessed):
  * `X_mmfall_clean_balanced.npy`: Shape `(1182, 10, 64, 4)`
  * `y_mmfall_clean_balanced.npy`: Shape `(1182,)` ($0 = \text{ADL}, 1 = \text{Fall}$)
  * `mmfall_cleaned_metadata.csv`: Motion clip metadata & statistics

---

## 🔬 Representation Benchmark Summary

| Representation | Format | Features / Channels | Model Backbone | Core Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Representation 1** | Micro-Doppler Spectrogram | $64 \times 64 \times 3$ (Density, Energy, Gradient) | **ResNet-18** *(Kinematic)* | High sensitivity to sudden velocity bursts during collapse |
| **Representation 2** | Orthogonal Projections | $64 \times 64 \times 3$ ($XZ, XY, YZ$) | **ResNet-18** *(Spatial)* | Explicitly tracks vertical height drop and ground expansion |
| **Representation 3** | Native 3D Point Set | $5 \times 640$ Matrix | **PointNet++** *(Geometric)* | Direct continuous 3D spatial coordinates without discretization |

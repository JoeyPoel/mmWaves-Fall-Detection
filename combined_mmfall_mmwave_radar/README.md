# Combined Multi-Sensor Dataset Pipeline

This directory contains the cross-dataset exploration, data preparation, baseline model training, and representation benchmark notebooks for the combined **mmFall (77 GHz)** and **TI IWR6843 (60–64 GHz)** datasets.

For detailed data processing documentation, see [COMBINED_DATASETS_DATA_PROCESSING.md](docs/COMBINED_DATASETS_DATA_PROCESSING.md).

---

## Notebook Overview

Execute the notebooks in the following order:

```text
combined_mmfall_mmwave_radar/
├── README.md                                         # Pipeline guide
├── 01_combined_datasets_explorer.ipynb               # Step 1: Cross-dataset feature explorer
├── 02_combined_datasets_data_preparation.ipynb       # Step 2: Combined dataset preparation (1,910 clips)
├── 03_train_cnn_combined_datasets.ipynb              # Step 3: Baseline 2D Spatial-Temporal CNN
└── representations/                                  # Step 4: Data Representation Benchmarks
    ├── train_rep1_spectrogram_resnet18.ipynb         # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
    ├── train_rep2_projections_resnet18.ipynb         # Rep 2: Orthogonal Spatial Projections (ResNet-18)
    ├── train_rep3_pointnet_3d.ipynb                  # Rep 3: Native 3D Point Set (PointNet)
    └── compare_radar_representations.ipynb            # Representation Comparison
```

---

## Preprocessed Arrays

* **Location**: `datasets/preprocessed/`
* **Arrays**:
  * `X_combined_clean_balanced.npy`: Base features (1,910 clips x 10 frames x 32 points x 4 features)
  * `y_combined_clean_balanced.npy`: Binary labels (0 = ADL, 1 = Fall)
  * `X_rep1_spectrogram_combined.npy`: Micro-Doppler Spectrograms (1,910 x 3 x 64 x 64)
  * `X_rep2_projections_combined.npy`: Orthogonal Projections (1,910 x 3 x 64 x 64)
  * `X_rep3_pointset_combined.npy`: Native 3D Point Sets (1,910 x 5 x 320)

---

## Representation Benchmarks

| Representation | Format | Dimensions | Backbone | Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Representation 1** | Micro-Doppler Spectrogram | 64 x 64 x 3 | ResNet-18 | Velocity profiles and frequency shifts |
| **Representation 2** | Orthogonal Projections | 64 x 64 x 3 | ResNet-18 | Spatial trajectory projections (XZ, XY, YZ) |
| **Representation 3** | Native 3D Point Set | 5 x 320 | PointNet | Continuous 3D point cloud coordinates |

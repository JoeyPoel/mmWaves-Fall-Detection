# mmFall Dataset Pipeline

This directory contains the exploration, data preparation, baseline model training, and representation benchmark notebooks for the **mmFall** dataset (77 GHz, TI IWR1443).

For detailed data processing documentation, see [MMFALL_DATA_PROCESSING.md](docs/MMFALL_DATA_PROCESSING.md) and [mmfallpaper.pdf](docs/mmfallpaper.pdf).

---

## Notebook Overview

Execute the notebooks in the following order:

```text
mmfall/
├── README.md                                         # Pipeline guide
├── 01_mmfall_explorer.ipynb                          # Step 1: Interactive Data Explorer
├── 02_mmfall_data_preparation.ipynb                  # Step 2: Preprocessing and Balancing
├── 03_train_cnn_mmfall.ipynb                         # Step 3: Baseline 2D Spatial-Temporal CNN
└── representations/                                  # Step 4: Data Representation Benchmarks
    ├── train_rep1_spectrogram_resnet18.ipynb         # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
    ├── train_rep2_projections_resnet18.ipynb         # Rep 2: Orthogonal Spatial Projections (ResNet-18)
    ├── train_rep3_pointnet_3d.ipynb                  # Rep 3: Native 3D Point Set (PointNet)
    └── compare_radar_representations.ipynb            # Representation Comparison
```

---

## Datasets and Preprocessed Arrays

* **Raw Data**: Located in `datasets/mmfall/data/`
* **Preprocessed Arrays**: Located in `datasets/preprocessed/`:
  * `X_mmfall_clean_balanced.npy`: Base features (1,182 clips x 10 frames x 32 points x 4 features)
  * `y_mmfall_clean_balanced.npy`: Binary labels (0 = ADL, 1 = Fall)
  * `mmfall_cleaned_metadata.csv`: Clip metadata

---

## Representation Benchmarks

| Representation | Format | Dimensions | Backbone | Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Representation 1** | Micro-Doppler Spectrogram | 64 x 64 x 3 | ResNet-18 | Velocity profiles and frequency shifts |
| **Representation 2** | Orthogonal Projections | 64 x 64 x 3 | ResNet-18 | Spatial trajectory projections (XZ, XY, YZ) |
| **Representation 3** | Native 3D Point Set | 5 x 320 | PointNet | Continuous 3D point cloud coordinates |

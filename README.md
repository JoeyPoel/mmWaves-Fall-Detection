# mmWave Radar Fall Detection

This repository contains data processing and deep learning code for non-intrusive fall detection using 4D millimeter-wave (mmWave) radar point clouds.

Falls at home are a major health risk for the elderly, but current monitoring solutions are often unpopular because cameras feel invasive and wearable devices are frequently not worn. mmWave radar is a great alternative because it is private and contact-free, but radar data can be messy and hard to interpret. This research compares three different ways to structure this radar data to see which is most effective at detecting falls. We test these methods using standard datasets and neural network models, focusing on how well each format avoids false alarms during fast movements and how it handles obstacles. The goal is to find an accurate and reliable data format to help keep seniors safe in their homes.

---

## Datasets Evaluated

We surveyed several public mmWave radar datasets:

| Dataset | Hardware | Data Schema | Focus | Source Link |
| :--- | :--- | :--- | :--- | :--- |
| **mmFall (Radar-Lab)** | TI IWR1443 (77 GHz) | Sequenced frames of (x, y, z) points with Doppler velocity | Fall detection and activity monitoring | [radar-lab/mmfall](https://github.com/radar-lab/mmfall) |
| **mmWave Fall Detection** | TI IWR6843 (60–64 GHz) | CSV format: x, y, z, velocity, SNR, noise | Fall events and daily activities | [Hugging Face Repository](https://huggingface.co/datasets/SachitanandAgalduti/mmwave-fall-detection-dataset) |
| **MiliPoint** | 60/77 GHz FMCW | Temporal sequences of 3D point coordinates | Human activity recognition | [yizzfz/milipoint](https://github.com/yizzfz/milipoint) |
| **mm-Pose** | TI IWR1443 (77 GHz) | Point clouds synchronized with 3D joint locations | Pose estimation | [sensor-research/mm-Pose](https://github.com/KylinC/mPose3D) |
| **mmBody** | Multi-sensor mmWave | Point clouds mapped to SMPL mesh vertices | 3D body reconstruction | [Chen3110/mmBody](https://github.com/Chen3110/mmBody) |

### Dataset Focus

We focus primarily on **mmFall** and **TI IWR6843**:
- **mmFall**: Provides 77 GHz recordings of fall events alongside routine activities of daily living (ADLs) like walking, sitting, bending, and lying down.
- **TI IWR6843**: Provides 60–64 GHz CSV point clouds collected across multiple subjects under varied room settings.

---

## Repository Structure

```text
mmWaves-Fall-Detection/
├── README.md                                 # Top-level project overview
├── docs/                                     # Data processing documentation
│   ├── MMFALL_DATA_PROCESSING.md             # mmFall processing details
│   ├── MMWAVE_RADAR_FALL_DETECTION_DATA_PROCESSING.md # TI IWR6843 processing details
│   └── COMBINED_DATASETS_DATA_PROCESSING.md  # Multi-sensor processing details
│
├── mmfall/                                   # mmFall Dataset Pipeline (77 GHz)
│   ├── README.md                             # Pipeline guide
│   ├── 01_mmfall_explorer.ipynb              # 3D data inspection
│   ├── 02_mmfall_data_preparation.ipynb      # Filtering, uniform sampling, and balancing
│   ├── 03_train_cnn_mmfall.ipynb             # 2D Spatial-Temporal CNN training
│   └── representations/                      # Data representation benchmarks
│       ├── train_rep1_spectrogram_resnet18.ipynb # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
│       ├── train_rep2_projections_resnet18.ipynb # Rep 2: Orthogonal Projections (ResNet-18)
│       ├── train_rep3_pointnet_3d.ipynb          # Rep 3: Native 3D Point Set (PointNet)
│       └── compare_radar_representations.ipynb    # Comparative evaluation
│
├── mmwave-radar-fall-detection/              # TI IWR6843 Dataset Pipeline (60–64 GHz)
│   ├── README.md                             # Pipeline guide
│   ├── 01_mmwave_radar_fall_detection_explorer.ipynb        # CSV data inspector
│   ├── 02_mmwave_radar_fall_detection_data_preparation.ipynb # Filtering, uniform sampling, and balancing
│   ├── 03_train_cnn_mmwave_radar_fall_detection.ipynb       # 2D Spatial-Temporal CNN training
│   └── representations/                                     # Data representation benchmarks
│       ├── train_rep1_spectrogram_resnet18.ipynb # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
│       ├── train_rep2_projections_resnet18.ipynb # Rep 2: Orthogonal Projections (ResNet-18)
│       ├── train_rep3_pointnet_3d.ipynb          # Rep 3: Native 3D Point Set (PointNet)
│       └── compare_radar_representations.ipynb    # Comparative evaluation
│
├── combined_mmfall_mmwave_radar/             # Combined Dataset Pipeline (77 GHz + 60 GHz)
│   ├── README.md                             # Pipeline guide
│   ├── 01_combined_datasets_explorer.ipynb   # Feature distribution comparisons
│   ├── 02_combined_datasets_data_preparation.ipynb # Dataset merging and standardization
│   ├── 03_train_cnn_combined_datasets.ipynb  # Combined CNN model training
│   └── representations/                      # Data representation benchmarks
│       ├── train_rep1_spectrogram_resnet18.ipynb # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
│       ├── train_rep2_projections_resnet18.ipynb # Rep 2: Orthogonal Projections (ResNet-18)
│       ├── train_rep3_pointnet_3d.ipynb          # Rep 3: Native 3D Point Set (PointNet)
│       └── compare_radar_representations.ipynb    # Comparative evaluation
│
├── datasets/                                 # Data Directory
│   └── preprocessed/                         # Balanced numpy feature arrays
│
└── models/                                   # Model Checkpoints
```

---

## Workflow Steps & Automated Pipeline Runner

Each dataset folder follows a four-step pipeline:

1. **`01_*_explorer.ipynb`**: Inspects raw point clouds, frame rates, and label metadata.
2. **`02_*_data_preparation.ipynb`**: Filters spatial noise, resamples frame points to N=32 using uniform random sampling, extracts 10-frame sliding windows, and creates a 1:1 balanced dataset.
3. **`03_*_train_cnn.ipynb`**: Trains a baseline 2D Spatial-Temporal CNN (`Radar4DCNN`).
4. **`representations/`**: Trains ResNet-18 models on Micro-Doppler Spectrograms (Rep 1) and Orthogonal Projections (Rep 2), and PointNet models on Native 3D Point Sets (Rep 3).

### Automated Sequential Execution

To execute all notebooks in strict sequential order for a dataset—halting immediately if any step fails—use the command-line pipeline runner:

```bash
# Run mmFall pipeline (77 GHz)
python run_pipeline.py --dataset mmfall

# Run TI IWR6843 pipeline (60–64 GHz)
python run_pipeline.py --dataset mmwave

# Run Combined dataset pipeline (77 GHz + 60 GHz)
python run_pipeline.py --dataset combined

# Run all dataset pipelines sequentially
python run_pipeline.py --dataset all

# Optionally skip exploratory notebooks
python run_pipeline.py --dataset mmfall --skip-explorer
```

Alternatively, run from within any dataset folder:
```bash
python mmfall/run_pipeline.py
python mmwave-radar-fall-detection/run_pipeline.py
python combined_mmfall_mmwave_radar/run_pipeline.py
```

---

## Baseline Performance Summary (`Radar4DCNN`)

| Dataset Pipeline | Radar Frequency | Balanced Clips | Accuracy | ROC-AUC | ADL Specificity | Model Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **[`mmfall/`](mmfall)** | 77 GHz | 1,182 clips | 88.19% | 0.9470 | 95.00% | [`models/cnn_mmfall_best.pth`](models/cnn_mmfall_best.pth) |
| **[`mmwave-radar-fall-detection/`](mmwave-radar-fall-detection)** | 60–64 GHz | 728 clips | 99.32% | 1.0000 | 100.00% | [`models/cnn_ti_best.pth`](models/cnn_ti_best.pth) |
| **[`combined_mmfall_mmwave_radar/`](combined_mmfall_mmwave_radar)** | 77 GHz + 60 GHz | 1,910 clips | 93.98% | 0.9867 | 95.29% | [`models/cnn_combined_best.pth`](models/cnn_combined_best.pth) |
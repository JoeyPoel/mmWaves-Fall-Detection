# 📡 mmWave Radar Fall Detection

A deep learning and data processing repository for **privacy-preserving, non-intrusive fall detection** using **4D Millimeter-Wave (mmWave) Radar Point Clouds**.

---

## 🎯 Project Overview & Objectives

Falls represent a critical risk for elderly populations. Traditional optical camera-based systems raise privacy concerns, while wearable sensors require active user compliance. **mmWave Radar** offers a high-precision, privacy-friendly alternative by emitting radio frequency pulses to capture 4D point clouds $(x, y, z, v_{\text{doppler}})$ of human motion.

In this repository, we:
1. **Explore & Benchmark mmWave Datasets**: Compare multiple open-source mmWave radar datasets for fall detection and human activity recognition.
2. **Focus on the `mmFall` Dataset**: Process, filter, and balance 4D point cloud sequences from the `mmFall` dataset, which offers the most comprehensive fall and non-fall (ADL) motion recordings.
3. **Preprocess & Clean Radar Data**: Implement noise filtering, spatial bounding box cropping, velocity normalization, frame-level point oversampling ($N=64$), and class balancing.
4. **Train Spatial-Temporal CNNs**: Build and train a PyTorch **2D Spatial-Temporal Convolutional Neural Network (`Radar4DCNN`)** achieving **~88.2% test accuracy** and **0.947 ROC-AUC**.

---

## 📊 Dataset Comparison & Primary Focus

We surveyed and evaluated several key mmWave radar datasets across the research community:

| Dataset | Sensor / Hardware | Data Schema | Primary Tasks | Repository Link |
| :--- | :--- | :--- | :--- | :--- |
| **mmFall (Radar-Lab)** ⭐ *(Primary Focus)* | 4D mmWave Radar (77 GHz) | Sequenced frames of $(x, y, z)$ spatial points with Doppler velocity | Fall detection & trajectory tracking | [GitHub: radar-lab/mmfall](https://github.com/radar-lab/mmfall) |
| **mmWave Fall Detection Dataset** | TI IWR6843 (60–64 GHz) | CSV columns: `[x, y, z, velocity, snr, noise]` | Fall detection vs. ADLs | [Hugging Face Repository](https://huggingface.co/datasets/SachitanandAgalduti/mmwave-fall-detection-dataset) |
| **MiliPoint** | 60/77 GHz mmWave FMCW | Temporal sequences of $(x, y, z)$ coordinates across indoor scenes | Human activity recognition & posture classification | [GitHub: yizzfz/milipoint](https://github.com/yizzfz/milipoint) |
| **mm-Pose** | TI IWR1443 (77 GHz) | $(x, y, z)$ point clouds synchronized with 3D joint coordinate ground truth | Human pose estimation & tracking | [GitHub: sensor-research/mm-Pose](https://github.com/sensor-research/mm-Pose) |
| **mmBody** | Multi-sensor mmWave Transceivers | Large-scale $(x, y, z)$ point cloud frames mapped to SMPL mesh vertices | 3D human body mesh and skeleton reconstruction | [GitHub: zeng-x/mmBody](https://github.com/zeng-x/mmBody) |

### 💡 Why `mmFall`?
We primarily focus on **`mmFall`** because:
- **Comprehensive Coverage**: It contains extensive recordings specifically capturing fall events alongside realistic Activities of Daily Living (ADLs) such as walking, sitting, bending, and lying down.
- **Rich 4D Point Clouds**: Each frame provides 3D spatial positions $(x, y, z)$, Doppler velocity, and signal energy, allowing for detailed tracking of rapid height drops and velocity spikes during fall events.
- **Realistic Motion Variance**: Includes subjects performing falls across different directions and speeds, serving as a robust benchmark for real-world deployment.

---

## 📁 Repository Structure (3 Parallel Dataset Pipelines)

```text
mmWaves-Fall-Detection/
├── README.md                                  # Top-level project overview & roadmap
│
├── docs/                                      # Documentation & Research References
│   ├── MMFALL_DATA_PROCESSING.md              # Detailed mathematical processing guide
│   └── mmfallpaper.pdf                        # Original mmFall research publication
│
├── mmfall/                                    # 📡 1. mmFall Dataset Pipeline (77 GHz, TI IWR1443)
│   ├── README.md                              # mmFall sequential workflow guide
│   ├── 01_mmfall_explorer.ipynb               # 1. Interactive 3D data explorer & action decoder
│   ├── 02_mmfall_data_preparation.ipynb       # 2. Outlier filtering, oversampling, & balancing
│   ├── 03_train_cnn_mmfall.ipynb              # 3. Spatial-Temporal 2D CNN baseline (88.2% acc)
│   └── representations/                       # 4. Multi-Representation Research Benchmark
│       ├── train_rep1_spectrogram_resnet18.ipynb  # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
│       ├── train_rep2_projections_resnet18.ipynb  # Rep 2: Orthogonal Projections (ResNet-18)
│       ├── train_rep3_pointnet_3d.ipynb           # Rep 3: Native 3D Point Set (PointNet++)
│       └── compare_radar_representations.ipynb     # Comparative benchmark synthesis & ROC curves
│
├── mmwave-radar-fall-detection/               # 📡 2. TI IWR6843 Dataset Pipeline (60–64 GHz CSV Data)
│   ├── README.md                              # TI IWR6843 sequential workflow guide
│   ├── 01_mmwave_radar_fall_detection_explorer.ipynb         # 1. Interactive CSV reader & 4D scrubber
│   ├── 02_mmwave_radar_fall_detection_data_preparation.ipynb # 2. Outlier filtering, oversampling, & balancing
│   ├── 03_train_cnn_mmwave_radar_fall_detection.ipynb        # 3. Spatial-Temporal 2D CNN baseline
│   └── representations/                                      # 4. Multi-Representation Benchmark
│       ├── train_rep1_spectrogram_resnet18.ipynb  # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
│       ├── train_rep2_projections_resnet18.ipynb  # Rep 2: Orthogonal Projections (ResNet-18)
│       ├── train_rep3_pointnet_3d.ipynb           # Rep 3: Native 3D Point Set (PointNet++)
│       └── compare_radar_representations.ipynb     # Comparative benchmark synthesis & ROC curves
│
├── combined_mmfall_mmwave_radar/              # 📡 3. Combined Multi-Sensor Benchmark (77 GHz + 60 GHz)
│   ├── README.md                              # Combined dataset workflow guide
│   ├── 01_combined_datasets_explorer.ipynb    # 1. Cross-dataset feature distribution comparisons
│   ├── 02_combined_datasets_data_preparation.ipynb # 2. Unified multi-sensor dataset merging (1,910 clips)
│   ├── 03_train_cnn_combined_datasets.ipynb   # 3. Multi-Sensor 4D CNN baseline training
│   └── representations/                       # 4. Multi-Representation Benchmark
│       ├── train_rep1_spectrogram_resnet18.ipynb  # Rep 1: Micro-Doppler Spectrogram (ResNet-18)
│       ├── train_rep2_projections_resnet18.ipynb  # Rep 2: Orthogonal Projections (ResNet-18)
│       ├── train_rep3_pointnet_3d.ipynb           # Rep 3: Native 3D Point Set (PointNet++)
│       └── compare_radar_representations.ipynb     # Comparative benchmark synthesis & ROC curves
│
├── datasets/                                  # Data Storage (Raw & Preprocessed)
│   ├── mmfall/                                # Raw mmFall dataset (DS0, DS1, DS2)
│   ├── mmwave-radar-fall-detection/          # Raw TI IWR6843 GatheredData (3 subjects)
│   ├── milipoint/                             # MiliPoint HAR dataset
│   ├── mmBody/                                # mmBody 3D mesh dataset
│   └── preprocessed/                          # Cleaned, 1:1 balanced feature tensors
│       ├── X_mmfall_clean_balanced.npy        # Shape: (1182, 10, 64, 4)
│       ├── y_mmfall_clean_balanced.npy        # Binary labels (0 = ADL, 1 = Fall)
│       ├── X_ti_clean_balanced.npy            # Shape: (728, 10, 64, 4)
│       ├── y_ti_clean_balanced.npy            # Binary labels (0 = ADL, 1 = Fall)
│       ├── X_combined_clean_balanced.npy      # Shape: (1910, 10, 64, 4)
│       ├── y_combined_clean_balanced.npy      # Binary labels (0 = ADL, 1 = Fall)
│       ├── X_rep1_spectrogram_*.npy           # 64x64x3 Micro-Doppler Spectrograms
│       ├── X_rep2_projections_*.npy           # 64x64x3 Orthogonal Projections
│       └── X_rep3_pointset_*.npy              # 5x640 Native 3D Point Sets
│
└── models/                                    # Saved Checkpoints & Evaluation Metrics
    ├── cnn_mmfall_best.pth                    # Best PyTorch 4D CNN weights (mmFall)
    ├── resnet18_rep1_spectrogram.pth          # Best ResNet-18 Spectrogram weights (mmFall)
    ├── cnn_ti_best.pth                        # Best PyTorch 4D CNN weights (TI IWR6843)
    └── cnn_combined_best.pth                  # Best PyTorch 4D CNN weights (Combined)
```

---

## 🚀 Workflows & Getting Started

Each of the three dataset folders follows an identical **4-step sequence**:

1. **`01_*_explorer.ipynb`**: Visualizes raw 4D point clouds, understands action codes and subjects, and scrubs frame-by-frame through individual trials.
2. **`02_*_data_preparation.ipynb`**: Filters spatial and SNR noise, applies room geometry adjustments, oversamples point clouds to $N=64$ using Algorithm 1, extracts 10-frame sliding windows, and guarantees an exact 1:1 balanced dataset.
3. **`03_*_train_cnn.ipynb`**: Trains a PyTorch 2D Spatial-Temporal `Radar4DCNN` baseline model.
4. **`representations/`**: Trains and evaluates ResNet-18 Kinematic, ResNet-18 Spatial, and PointNet++ Geometric models across the three representation formats, comparing ROC-AUC curves in `compare_radar_representations.ipynb`.

---

## 📈 Baseline Model Performance Summary (`Radar4DCNN`)

| Dataset Pipeline | Radar Frequency | Balanced Clips | Accuracy | ROC-AUC | ADL Specificity | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **[`mmfall/`](file:///c:/Users/joeyw/GitProjects/mmWaves-Fall-Detection/mmfall)** | 77 GHz | 1,182 clips | **88.19%** | **0.9470** | **95.00%** | [`models/cnn_mmfall_best.pth`](file:///c:/Users/joeyw/GitProjects/mmWaves-Fall-Detection/models/cnn_mmfall_best.pth) |
| **[`mmwave-radar-fall-detection/`](file:///c:/Users/joeyw/GitProjects/mmWaves-Fall-Detection/mmwave-radar-fall-detection)** | 60–64 GHz | 728 clips | **99.32%** | **1.0000** | **100.00%** | [`models/cnn_ti_best.pth`](file:///c:/Users/joeyw/GitProjects/mmWaves-Fall-Detection/models/cnn_ti_best.pth) |
| **[`combined_mmfall_mmwave_radar/`](file:///c:/Users/joeyw/GitProjects/mmWaves-Fall-Detection/combined_mmfall_mmwave_radar)** | 77 GHz + 60 GHz | 1,910 clips | **93.98%** | **0.9867** | **95.29%** | [`models/cnn_combined_best.pth`](file:///c:/Users/joeyw/GitProjects/mmWaves-Fall-Detection/models/cnn_combined_best.pth) |
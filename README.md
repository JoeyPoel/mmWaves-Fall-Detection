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

## 📁 Repository Structure & Workflows

```text
mmWaves-Fall-Detection/
├── mmfall_explorer.ipynb                      # Interactive visual explorer & action decoder for mmFall
├── mmwave_radar_fall_detection_explorer.ipynb # Interactive reader for TI IWR6843 CSV fall dataset
├── epa.ipynb / eda_mmfall.ipynb               # Exploratory Pattern Analysis & 4D radar feature distributions
├── mmfall_data_preparation.ipynb              # Data cleaning, outlier filtering, & 1:1 balance pipeline
├── train_cnn_mmfall.ipynb                     # PyTorch Spatial-Temporal CNN training & evaluation
├── datasets/
│   ├── mmfall/                                # Raw mmFall dataset directory (Action folders 1-32)
│   └── preprocessed/                          # Cleaned, balanced feature tensors (X, y, metadata)
│       ├── X_mmfall_clean_balanced.npy        # Shape: (1182, 10, 64, 4)
│       ├── y_mmfall_clean_balanced.npy        # Binary labels (0 = ADL, 1 = Fall)
│       └── mmfall_cleaned_metadata.csv        # Motion clip metadata & statistics
└── models/
    └── cnn_mmfall_best.pth                    # Saved PyTorch CNN model checkpoint
```

---

## 🚀 Key Notebooks & Getting Started

1. **`mmfall_explorer.ipynb`**:
   - Decodes folder structures, action codes, and subject IDs.
   - Interactive 3D point cloud visualizer with interactive frame sliders.
2. **`mmfall_data_preparation.ipynb`**:
   - Filters spatial noise, crops bounding boxes ($x, y \in [-3, 3]\text{m}$, $z \in [-1, 2.5]\text{m}$).
   - Resamples frames to uniform point counts ($N=64$) and extracts 10-frame sliding windows ($1.0\text{s}$ clip length).
   - Exports preprocessed `.npy` feature tensors.
3. **`train_cnn_mmfall.ipynb`**:
   - Loads preprocessed tensors, splits into Train (80%) and Test (20%).
   - Trains `Radar4DCNN` model over 25 epochs.
   - Generates Loss/Accuracy curves, Confusion Matrix, ROC-AUC curve (0.947), and single-clip inference.

---

## 📈 Model Architecture & Results (`Radar4DCNN`)

- **Input Tensor**: `(Batch_Size, 4_Channels, 10_Frames, 64_Points)` where 4 channels represent $[x, y, z, v_{\text{doppler}}]$.
- **Architecture**: 3 2D Conv Blocks with Batch Normalization, ReLU, MaxPool2d / AdaptiveAvgPool2d, Dropout ($p=0.4$), and Dense Layers.
- **Test Performance**:
  - **Accuracy**: `88.19%`
  - **ROC-AUC**: `0.9470`
  - **ADL Specificity**: `95.0%` (113/119 ADL test clips correctly identified)
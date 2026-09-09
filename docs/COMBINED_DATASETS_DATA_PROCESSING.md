# 📡 Combined mmFall + TI IWR6843 Dataset: Harmonization & Processing Guide

This document provides an exhaustive, mathematically rigorous guide detailing how two heterogeneous millimeter-wave (mmWave) radar datasets—**`mmFall`** (77 GHz, TI IWR1443) and **`TI IWR6843`** (60–64 GHz)—are harmonized, standardized, merged, and transformed into a unified multi-sensor fall detection benchmark across this repository.

---

## 📑 Table of Contents
1. [Research Motivation: Cross-Sensor & Cross-Frequency Benchmark](#1-research-motivation-cross-sensor--cross-frequency-benchmark)
2. [Comparative Sensor & Hardware Specifications](#2-comparative-sensor--hardware-specifications)
3. [Multi-Sensor Data Harmonization Strategy](#3-multi-sensor-data-harmonization-strategy)
4. [Dataset Merging & Random Permutation Protocol](#4-dataset-merging--random-permutation-protocol)
5. [Combined Tensor Schema & Output Artifacts](#5-combined-tensor-schema--output-artifacts)
6. [Unified Downstream Model Representations (Rep 1, 2, 3)](#6-unified-downstream-model-representations-rep-1-2-3)
7. [Empirical Multi-Sensor Model Performance](#7-empirical-multi-sensor-model-performance)
8. [Cross-Domain Generalization Research Protocol](#8-cross-domain-generalization-research-protocol)

---

## 1. Research Motivation: Cross-Sensor & Cross-Frequency Benchmark

Most published mmWave fall detection research evaluates algorithms on a single sensor in a single laboratory setting. This introduces severe domain-overfitting risks:
* **Carrier Frequency Differences**: 77 GHz vs. 60 GHz signals exhibit different wavelength penetrations, reflection coefficients, and Doppler sensitivities ($f_d = \frac{2v}{\lambda}$).
* **Hardware & Antenna Geometries**: Differences in chirp configurations, antenna beam patterns, and onboard CFAR detection thresholds produce varying point cloud densities.
* **Subject Variance**: Different actors, movement cadences, and body morphologies.

By unifying the **`mmFall`** and **`TI IWR6843`** datasets, this repository creates a **cross-frequency, multi-sensor benchmark of 1,910 motion clips** to evaluate true model generalization across diverse radar platforms.

---

## 2. Comparative Sensor & Hardware Specifications

| Parameter | mmFall Dataset | TI IWR6843 Dataset | Unified Benchmark |
| :--- | :--- | :--- | :--- |
| **Radar Chipset** | TI IWR1443 | TI IWR6843 | Heterogeneous Multi-Sensor |
| **Carrier Frequency** | $77.0\text{ GHz}$ ($\lambda \approx 3.9\text{ mm}$) | $60.0 - 64.0\text{ GHz}$ ($\lambda \approx 4.8\text{ mm}$) | Multi-Band (60–77 GHz) |
| **Raw Storage Format** | Binary NumPy Arrays (`.npy`) | Tabular CSV Files (`.csv`) | Standardized Tensor (`.npy`) |
| **Coordinate Space** | Polar Range-Azimuth-Elevation | Pre-calculated Sensor Cartesian | Room-Aligned Cartesian |
| **Mounting Orientation** | $H = 1.80\text{ m}$, $+10.0^\circ$ downward tilt | Standard horizontal room mount | Standardized Ground Relative |
| **Temporal Sampling** | $10\text{ Hz}$ ($100\text{ ms}$/frame) | $10\text{ Hz}$ ($100\text{ ms}$/frame) | **$10\text{ Hz}$ Synced** |
| **Clean Balanced Clips** | **1,182 clips** (591 Fall : 591 ADL) | **728 clips** (364 Fall : 364 ADL) | **1,910 clips** (955 Fall : 955 ADL) |

---

## 3. Multi-Sensor Data Harmonization Strategy

Implemented in [02_combined_datasets_data_preparation.ipynb](combined_mmfall_mmwave_radar/02_combined_datasets_data_preparation.ipynb):

To make the datasets interchangeable and combinable, three harmonization steps are enforced:

### 3.1 Spatial Frame of Reference Alignment
* **mmFall**: Raw polar coordinates $(r, \theta, \phi)$ are rotated by $+10^\circ$ pitch and elevated by $+1.80\text{ m}$ to establish room coordinates $(X, Y, Z)$.
* **TI IWR6843**: Raw Cartesian coordinates $(x, y, z)$ are bounded by the room box $X \in [-2, 2]\text{ m}$, $Y \in [0, 6]\text{ m}$, $Z \in [-0.5, 2.2]\text{ m}$.
* **Position Invariance**: Both datasets apply window-level mean centering to horizontal axes:
  $$\Delta X = X - \bar{X}_{\text{window}}, \quad \Delta Y = Y - \bar{Y}_{\text{window}}, \quad Z = Z$$
  Absolute vertical elevation $Z$ is preserved identically across both sensors.

### 3.2 Dynamic Point Cloud Standardization (Algorithm 1)
Radar point density per frame differs between the two sensors:
* `mmFall`: $\sim 5 - 20$ points per frame.
* `TI IWR6843`: $\sim 6 - 15$ points per frame.

Both datasets are resampled to an exact dimension of **$N = 64$ points per frame** using the mean-preserving transformation:
$$\mathbf{p}_i' = \sqrt{\frac{N}{M}} \cdot (\mathbf{p}_i - \hat{\mu}) + \hat{\mu}, \quad i = 1, \dots, M$$
with centroid padding for remaining positions.

### 3.3 Temporal Clip Harmonization
* **Duration**: Exactly 10 frames ($1.0\text{ second}$ of motion at $10\text{ FPS}$).
* **Feature Schema**: Every point vector carries 4 identical channels: $[\Delta X, \Delta Y, Z, v_{\text{doppler}}]$.

---

## 4. Dataset Merging & Random Permutation Protocol

Simple sequential concatenation ($[\text{mmFall}, \text{TI}]$) would cause mini-batches during SGD training to alternate between homogeneous blocks of sensor data, inducing catastrophic gradient oscillations.

The merging pipeline implements:
1. **Direct Axis-0 Concatenation**:
   $$\mathbf{X}_{\text{combined}} = \begin{bmatrix} \mathbf{X}_{\text{mmfall}} \\ \mathbf{X}_{\text{ti}} \end{bmatrix} \in \mathbb{R}^{1910 \times 10 \times 64 \times 4}$$
   $$\mathbf{y}_{\text{combined}} = \begin{bmatrix} \mathbf{y}_{\text{mmfall}} \\ \mathbf{y}_{\text{ti}} \end{bmatrix} \in \{0, 1\}^{1910}$$

2. **Deterministic Pseudo-Random Permutation**:
   Using a fixed random seed (`seed = 42`):
   $$\pi = \text{Permutation}(1910)$$
   $$\mathbf{X}_{\text{combined}} \leftarrow \mathbf{X}_{\text{combined}}[\pi], \quad \mathbf{y}_{\text{combined}} \leftarrow \mathbf{y}_{\text{combined}}[\pi]$$
   This guarantees identical train/test splits across runs while thoroughly interleaving 77 GHz and 60 GHz recordings across mini-batches.

---

## 5. Combined Tensor Schema & Output Artifacts

All unified tensors are exported to [`datasets/preprocessed/`](datasets/preprocessed):

| Tensor Filename | Shape | Data Type | Description |
| :--- | :--- | :---: | :--- |
| [`X_combined_clean_balanced.npy`](datasets/preprocessed/X_combined_clean_balanced.npy) | `(1910, 10, 64, 4)` | `float32` | Standardized 4D point cloud sequences $[\Delta x, \Delta y, z, v]$ |
| [`y_combined_clean_balanced.npy`](datasets/preprocessed/y_combined_clean_balanced.npy) | `(1910,)` | `int64` | Binary ground-truth labels (**955 Fall : 955 ADL**) |
| [`X_rep1_spectrogram_combined.npy`](datasets/preprocessed/X_rep1_spectrogram_combined.npy) | `(1910, 3, 64, 64)` | `float32` | Unified Micro-Doppler Spectrograms |
| [`X_rep2_projections_combined.npy`](datasets/preprocessed/X_rep2_projections_combined.npy) | `(1910, 3, 64, 64)` | `float32` | Unified Dual Orthogonal 2D Spatial Projections |
| [`X_rep3_pointset_combined.npy`](datasets/preprocessed/X_rep3_pointset_combined.npy) | `(1910, 5, 640)` | `float32` | Unified Unordered 3D Point Matrices $[\Delta x, \Delta y, z, v, t]$ |
| `y_rep*_combined.npy` | `(1910,)` | `int64` | Identical ground-truth label vectors for all representations |

---

## 6. Unified Downstream Model Representations (Rep 1, 2, 3)

The combined dataset evaluates the three core radar data representation paradigms on a truly diverse multi-sensor benchmark:

```text
                           X_combined_clean_balanced.npy (1910, 10, 64, 4)
                                                 │
                ┌────────────────────────────────┼────────────────────────────────┐
                ▼                                ▼                                ▼
          Representation 1                 Representation 2                 Representation 3
      Micro-Doppler Spectrogram        Orthogonal Projections           Native 3D Point Set
         (64x64 x 3 Channels)             (64x64 x 3 Channels)               (5 x 640 Matrix)
                │                                │                                │
                ▼                                ▼                                ▼
            ResNet-18                        ResNet-18                        PointNet++
           (Kinematic)                       (Spatial)                        (Geometric)
```

| Dimension | Representation 1 (Kinematic) | Representation 2 (Spatial) | Representation 3 (Geometric) |
| :--- | :--- | :--- | :--- |
| **Channels** | Density, Energy, $\Delta v / \Delta t$ | $XZ$ Elevation, $XY$ Ground, Velocity | $\Delta x, \Delta y, z, v, t_{\text{norm}}$ |
| **Model** | PyTorch ResNet-18 | PyTorch ResNet-18 | PyTorch PointNet / PointNet++ |
| **Input Shape** | `(Batch, 3, 64, 64)` | `(Batch, 3, 64, 64)` | `(Batch, 5, 640)` |
| **Core Hypothesis** | Doppler bursts generalize across carrier frequency shifts | Spatial height collapse ($Z$) is frequency-invariant | Geometry-based networks avoid quantization artifacts |

---

## 7. Empirical Multi-Sensor Model Performance

Trained using `Radar4DCNN` on a stratified 20% test split (382 test clips: 191 Fall / 191 ADL):

* **Test Accuracy**: **`93.98%`** (359 / 382 correct)
* **Test ROC-AUC**: **`0.9867`**
* **ADL Specificity (TNR)**: **`95.29%`** (182 / 191 ADLs correctly identified)
* **Fall Sensitivity (TPR)**: **`92.67%`** (177 / 191 Falls correctly detected)
* **Model Checkpoint**: [`models/cnn_combined_best.pth`](models/cnn_combined_best.pth)

### Confusion Matrix on Multi-Sensor Test Set:
$$\begin{bmatrix} \text{TN} = 182 & \text{FP} = 9 \\ \text{FN} = 14 & \text{TP} = 177 \end{bmatrix}$$

---

## 8. Cross-Domain Generalization Research Protocol

The existence of both individual dataset pipelines ([`mmfall/`](mmfall), [`mmwave-radar-fall-detection/`](mmwave-radar-fall-detection)) and the unified pipeline ([`combined_mmfall_mmwave_radar/`](combined_mmfall_mmwave_radar)) enables three key research experiments:

1. **In-Domain Benchmark**: Train and test exclusively on the same sensor (establishes upper-bound performance).
2. **Joint Multi-Sensor Training**: Train on `X_combined` to develop a single sensor-agnostic fall detection backbone.
3. **Zero-Shot Cross-Sensor Transfer**:
   * Train on `mmFall` (77 GHz) $\rightarrow$ Test zero-shot on `TI IWR6843` (60 GHz).
   * Train on `TI IWR6843` (60 GHz) $\rightarrow$ Test zero-shot on `mmFall` (77 GHz).
   * Directly quantifies which data representation (Spectrogram vs. Projections vs. PointNet) exhibits superior cross-frequency transferability.

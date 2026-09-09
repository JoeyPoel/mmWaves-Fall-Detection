# 📡 mmFall Radar Dataset: Interpretation & Processing Guide

This document provides an exhaustive, end-to-end technical reference explaining how raw 4D millimeter-wave (mmWave) radar data from the **`mmFall`** dataset is interpreted, filtered, labeled, and converted into training tensors across this repository.

> **📖 Companion Processing Guides**:
> * For the 60–64 GHz TI IWR6843 dataset, see [MMWAVE_RADAR_FALL_DETECTION_DATA_PROCESSING.md](docs/MMWAVE_RADAR_FALL_DETECTION_DATA_PROCESSING.md).
> * For the unified multi-sensor benchmark, see [COMBINED_DATASETS_DATA_PROCESSING.md](docs/COMBINED_DATASETS_DATA_PROCESSING.md).

---

## 📑 Table of Contents
1. [Sensor Hardware & Raw Data Schema](#1-sensor-hardware--raw-data-schema)
2. [Dataset Hierarchy & File Taxonomy](#2-dataset-hierarchy--file-taxonomy)
3. [Outlier Filtering & 3D Room Coordinate Transformation](#3-outlier-filtering--3d-room-coordinate-transformation)
4. [Algorithm 1: Mean-Preserving Oversampling ($N=64$)](#4-algorithm-1-mean-preserving-oversampling-n64)
5. [Temporal Windowing & Ground-Truth Labeling Protocol](#5-temporal-windowing--ground-truth-labeling-protocol)
6. [Class Balancing & Audit Metadata](#6-class-balancing--audit-metadata)
7. [Downstream Model Representations (Rep 1, 2, 3)](#7-downstream-model-representations-rep-1-2-3)

---

## 1. Sensor Hardware & Raw Data Schema

The raw radar data is captured using a **Texas Instruments (TI) IWR1443 mmWave 77 GHz Radar Sensor** coupled with a **DCA1000 EVM** real-time data capture board.

### Physical Installation
* **Mounting Height**: $H_{\text{sensor}} = 1.80\text{ m}$ above the floor.
* **Pitch Angle**: Tilted downward by $+10.0^\circ$ toward the room floor.
* **Frame Rate**: $10\text{ Hz}$ ($100\text{ ms}$ per radar frame).

### Raw Array Structure (`.npy`)
Each recording file contains a sequence of temporal frames. Because radar point clouds are sparse and variable, each frame contains an arbitrary number of detected reflection points $M_t$:
```text
Raw Recording (.npy)
  ├── Frame 0: Array of shape (M_0, 15)
  ├── Frame 1: Array of shape (M_1, 15)
  └── Frame T: Array of shape (M_T, 15)
```

Each point vector consists of **15 attributes**:
| Index | Attribute | Symbol / Unit | Description |
| :---: | :--- | :---: | :--- |
| `0, 1, 2` | Sensor Cartesian | $x_s, y_s, z_s$ [m] | Raw relative coordinates in sensor frame |
| `3, 4, 5` | Body Centroid | $c_x, c_y, c_z$ [m] | Estimated center-of-mass coordinates for the human cluster |
| `9` | Target Range | $r$ [m] | Radial distance from radar antenna |
| `10` | Azimuth Angle | $\theta_{\text{az}}$ [rad] | Horizontal angle of arrival |
| `11` | Elevation Angle | $\phi_{\text{el}}$ [rad] | Vertical angle of arrival |
| `12` | Doppler Velocity | $v$ [m/s] | Radial Doppler velocity (negative = moving away, positive = moving toward) |
| `13` | Signal-to-Noise | $\text{SNR}$ [dB / code] | Confidence & reflection intensity |
| `14` | Noise Metric | $\sigma_{\text{noise}}$ | Ambient RF noise floor |

---

## 2. Dataset Hierarchy & File Taxonomy

The raw files are located under [`datasets/mmfall/data/`](datasets/mmfall/data):

```text
datasets/mmfall/data/
├── DS0/                               # 2 hours continuous unannotated normal ADL (No falls)
├── DS1/                               # Demo recordings: 4 falls and 4 normal activities
│   ├── DS1_4falls.npy                 # Forward, backward, left, right falls
│   └── DS1_4normal.npy                # Sit on floor, crouch, bend, jump
└── DS2/                               # 35 Benchmark trial files (Falls + ADLs)
    ├── DS2_bf_01.npy / .csv           # Backward Fall trials
    ├── DS2_ff_01.npy / .csv           # Forward Fall trials
    ├── DS2_lf_01.npy / .csv           # Left Fall trials
    ├── DS2_rf_01.npy / .csv           # Right Fall trials
    ├── DS2_sf_01.npy ... 05.npy       # Syncope / Slow Fall trials
    ├── DS2_b_01.npy                   # Bending ADL trials
    ├── DS2_c_01.npy ... 05.npy        # Crouching ADL trials
    └── DS2_j_01.npy ... 03.npy        # Jumping ADL trials
```

### Action Code Glossary
* **`_bf_` (Backward Fall - Class 1)**: Loss of balance backwards onto the floor.
* **`_ff_` (Forward Fall - Class 1)**: Loss of balance forwards onto the floor.
* **`_lf_` (Left Fall - Class 1)**: Fall directly onto the left side.
* **`_rf_` (Right Fall - Class 1)**: Fall directly onto the right side.
* **`_sf_` (Syncope / Slow Fall - Class 1 / Edge Case)**: Fainting or slow vertical collapse.
* **`_b_` (Bending - Class 0 ADL)**: Reaching down to pick up an object and returning upright.
* **`_c_` (Crouching - Class 0 ADL)**: Squatting low to the ground and standing back up.
* **`_j_` (Jumping - Class 0 ADL)**: Jumping vertically on the spot.

### Ground-Truth Timestamp Files (`.csv`)
In `DS2`, every fall recording `.npy` is paired with a `.csv` file (e.g., [`DS2_bf_01.csv`](datasets/mmfall/data/DS2/DS2_bf_01.csv)). These CSV files contain the **exact camera-verified frame indices** where the fall impact occurs (e.g., frames `193, 414, 649, 986, 1333`).

---

## 3. Outlier Filtering & 3D Room Coordinate Transformation

Implemented in `MMfallDataCleaner` in [mmfall_data_preparation.ipynb](mmfall/02_mmfall_data_preparation.ipynb):

### 3.1 Polar to Sensor Cartesian
Using range $r$, azimuth $\theta_{\text{az}}$, and elevation $\phi_{\text{el}}$:
$$x_s = r \cdot \cos(\phi_{\text{el}}) \cdot \sin(\theta_{\text{az}})$$
$$y_s = r \cdot \cos(\phi_{\text{el}}) \cdot \cos(\theta_{\text{az}})$$
$$z_s = r \cdot \sin(\phi_{\text{el}})$$

### 3.2 3D Room Coordinate Tilt Rotation & Height Correction
To account for the downward tilt ($\alpha = +10.0^\circ$) and height ($H = 1.80\text{ m}$):
$$R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & \sin\alpha \\ 0 & -\sin\alpha & \cos\alpha \end{bmatrix}$$

$$\begin{bmatrix} X_{\text{room}} \\ Y_{\text{room}} \\ Z_{\text{room}} \end{bmatrix} = R_x(10^\circ) \begin{bmatrix} x_s \\ y_s \\ z_s \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 1.80 \end{bmatrix}$$

The same transformation is applied to the cluster centroid coordinates $(c_x, c_y, c_z) \rightarrow (c_{x,\text{room}}, c_{y,\text{room}}, c_{z,\text{room}})$.

### 3.3 Outlier & Artifact Rejection Rules
Points are discarded if they fail any of three physical sanity checks:
1. **SNR Quality Filter**: $\text{SNR} \ge 100$ (eliminates diffuse multipath and phantom reflections).
2. **Velocity Clamping**: $|v| \le 3.0\text{ m/s}$ (rejects non-human velocity artifacts).
3. **Physical Room Bounding Box**:
   * $X_{\text{room}} \in [-2.0, 2.0]\text{ m}$ (lateral room walls)
   * $Y_{\text{room}} \in [0.2, 5.5]\text{ m}$ (sensor minimum distance to back wall)
   * $Z_{\text{room}} \in [-0.2, 2.2]\text{ m}$ (floor to ceiling)

### 3.4 Radar Cross Section (RCS) Proxy
$$\text{RCS} = 40 \log_{10}(\max(r, 10^{-3})) + 0.1 \cdot \text{SNR} + 0.1 \cdot \sigma_{\text{noise}}$$

---

## 4. Algorithm 1: Mean-Preserving Oversampling ($N=64$)

Because the number of valid points $M_t$ fluctuates from frame to frame, neural networks require a standardized point count. Simple zero-padding or random duplication distorts spatial moments.

The pipeline implements **Algorithm 1** from the mmFall paper:
* **Target Count**: Fixed $N = 64$ points per frame.
* **If $M \ge N$**: The frame is truncated to the first $N$ points.
* **If $M < N$**: Points are rescaled and padded using the empirical centroid $\hat{\mu} = \frac{1}{M}\sum_{i=1}^M \mathbf{p}_i$:
  $$\mathbf{p}_i' = \sqrt{\frac{N}{M}} \cdot \mathbf{p}_i + \left(1 - \sqrt{\frac{N}{M}}\right) \hat{\mu}, \quad i = 1, \dots, M$$
  The remaining $(N - M)$ slots are filled with exact copies of the centroid $\hat{\mu}$.

> [!NOTE]
> This transformation mathematically preserves the exact sample mean $\hat{\mu}$ and covariance matrix $\hat{\Sigma}$ of the original radar reflections.

---

## 5. Temporal Windowing & Ground-Truth Labeling Protocol

Continuous recording sequences are segmented into **10-frame sliding windows** ($1.0\text{ second}$) with a stride of 3 frames ($\sim 0.3\text{ seconds}$).

### 5.1 Position-Invariant Reference Shift
To ensure the model is invariant to where the person stands in the room, spatial coordinates are centered on the initial frame's centroid $(c_x^{(1)}, c_y^{(1)})$:
$$\Delta X = X_{\text{room}} - c_x^{(1)}, \quad \Delta Y = Y_{\text{room}} - c_y^{(1)}, \quad Z = Z_{\text{room}}$$
*Absolute vertical height $Z$ is kept unshifted*, as distance to the floor is the fundamental geometric indicator of posture.

### 5.2 Label Assignment Logic
Every window (`start_f` to `end_f = start_f + 9`) is assigned a binary label $y \in \{0, 1\}$ using:

```python
# 1. Ground truth timestamp overlap
is_gt_fall = any(start_f - 10 <= gm <= end_f + 10 for gm in gt_markers)

# 2. Vertical centroid drop calculation
h_drop = cz_start - cz_end

# 3. Source file context gate
is_fall_file = any(
    code in fpath.stem.lower()
    for code in ["fall", "_bf_", "_ff_", "_lf_", "_rf_", "_sf_"]
)

# 4. Final binary classification
is_fall_window = 1 if (is_gt_fall or (is_fall_file and h_drop > 0.50)) else 0
```

### Why Crouching is NOT Mislabeled as a Fall
* In dedicated crouching (`_c_`), bending (`_b_`), and jumping (`_j_`) files, **`is_fall_file` is `False`** and **`is_gt_fall` is `False`**.
* Even if a person crouches down by $0.85\text{ m}$, the window is **guaranteed to remain Class 0 (ADL)**.
* In the audit metadata, **100% of the 97 crouching windows from `DS2_c_*` are labeled `0` (ADL)**.

---

## 6. Class Balancing & Audit Metadata

In natural motion, normal activities generate far more windows than brief fall events. An unconstrained dataset would heavily bias models toward predicting ADLs.

### 1:1 Balancing
* Total Fall windows extracted: **591 windows** (Class 1).
* Non-Fall ADL windows sampled: **591 windows** (Class 0), randomly subsampled from normal activities and reset intervals.
* **Final Dataset Size**: Exactly **1,182 balanced clips**.

### Preprocessed Artifacts
Saved in [`datasets/preprocessed/`](datasets/preprocessed):
1. [`X_mmfall_clean_balanced.npy`](datasets/preprocessed/X_mmfall_clean_balanced.npy): Shape `(1182, 10, 64, 4)` containing $[\Delta x, \Delta y, z, v]$.
2. [`y_mmfall_clean_balanced.npy`](datasets/preprocessed/y_mmfall_clean_balanced.npy): Shape `(1182,)` binary labels ($0 = \text{ADL}, 1 = \text{Fall}$).
3. [`mmfall_cleaned_metadata.csv`](datasets/preprocessed/mmfall_cleaned_metadata.csv): Per-window audit log containing `file`, `start_frame`, `end_frame`, `label`, and `height_drop`.

---

## 7. Downstream Model Representations (Rep 1, 2, 3)

The preprocessed balanced clips are mapped into three distinct deep learning representations to benchmark kinematic, spatial, and geometric paradigms:

```text
                         X_mmfall_clean_balanced.npy (1182, 10, 64, 4)
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

| Metric / Dimension | Representation 1 (Kinematic) | Representation 2 (Spatial) | Representation 3 (Geometric) |
| :--- | :--- | :--- | :--- |
| **Tensor File** | [`X_rep1_spectrogram.npy`](datasets/preprocessed/X_rep1_spectrogram.npy) | [`X_rep2_projections.npy`](datasets/preprocessed/X_rep2_projections.npy) | [`X_rep3_pointset.npy`](datasets/preprocessed/X_rep3_pointset.npy) |
| **Tensor Shape** | `(1182, 3, 64, 64)` | `(1182, 3, 64, 64)` | `(1182, 5, 640)` |
| **Label Vector** | [`y_rep1_spectrogram.npy`](datasets/preprocessed/y_rep1_spectrogram.npy) | [`y_rep2_projections.npy`](datasets/preprocessed/y_rep2_projections.npy) | [`y_rep3_pointset.npy`](datasets/preprocessed/y_rep3_pointset.npy) |
| **Channels / Feats** | Velocity Density, Energy, $\Delta v / \Delta t$ | $XZ$ (Side view), $XY$ (Top view), $YZ$ | $x, y, z, v, \text{time\_norm}$ |
| **Model Backbone** | [ResNet-18 (Rep 1)](mmfall/representations/train_rep1_spectrogram_resnet18.ipynb) | [ResNet-18 (Rep 2)](mmfall/representations/train_rep2_projections_resnet18.ipynb) | [PointNet++ (Rep 3)](mmfall/representations/train_rep3_pointnet_3d.ipynb) |
| **Core Advantage** | Captures Doppler velocity acceleration spikes | Tracks vertical posture drop & ground spread | Operates directly on raw point clouds without grid binning |

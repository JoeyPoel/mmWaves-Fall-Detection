# TI IWR6843 mmWave Radar Fall Detection — 4D Point Cloud Benchmark

A modular, zero-leakage deep learning framework for non-intrusive fall detection using 4D millimeter-wave (mmWave) radar point cloud sequences captured by the **Texas Instruments (TI) IWR6843 (60–64 GHz)** radar sensor.

---

## Executive Overview

Falls at home represent a critical health risk for seniors and individuals living independently. Camera-based systems introduce privacy concerns, and wearables are frequently forgotten or discarded. 4D mmWave radar provides an optimal non-intrusive alternative: it is private, contact-free, and operates reliably regardless of room lighting or privacy constraints.

However, mmWave point cloud data is inherently sparse, noisy, and non-stationary. This repository establishes a **unified, zero-leakage Leave-One-Subject-Out (LOSO) benchmarking framework** on the **TI IWR6843 mmWave Radar Fall Detection Dataset (102 CSV files)** to evaluate spatial-temporal representations (**Micro-Doppler Velocity-Time Spectrograms**, **Orthogonal 2D Spatial Projections**, and **Native 3D Point Tensors / PointNet++**).

---

## Dataset Overview & Partitioning Integrity

The benchmark operates exclusively on the **TI IWR6843 mmWave Radar Fall Detection Dataset**:

| Metric / Dimension | Specification Details |
| :--- | :--- |
| **Hardware / Frequency** | TI IWR6843 Evaluation Module (60–64 GHz) |
| **Total Recording CSV Files** | **102 CSV files** (51 Falls vs. 51 Non-Falls / ADLs) |
| **Participant Subjects** | **3 Subjects**: `Areeb` (34 files), `Raffay` (34 files), `Towsif` (34 files) |
| **Class Balance** | Exactly 17 Falls & 17 ADLs per subject (1:1 class ratio) |
| **Raw Point Schema** | `[frame, DetObj#, x, y, z, v, snr, noise]` |
| **Cross-Validation Scheme** | **3-Fold Leave-One-Subject-Out (LOSO)** |

### 3-Fold Leave-One-Subject-Out (LOSO) Protocol with 3-Way Partitioning

To ensure 100% room and subject generalization without data leakage:
- **Test Set (34 files / 33.3%)**: One complete subject held out exclusively for final evaluation (17 Falls, 17 ADLs).
- **Training Set (54 files / 52.9%)**: Partitioned from the remaining 68 training-pool files with exact 1:1 class balance (27 Falls, 27 ADLs).
- **Validation Set (14 files / 13.7%)**: Partitioned from the 68 training-pool files for hyperparameter selection and epoch early stopping (7 Falls, 7 ADLs).

- **Fold 0**: Held-out Test Subject = `Areeb` (34 test files) | Train/Val Pool = `Raffay`, `Towsif` (54 train, 14 val)
- **Fold 1**: Held-out Test Subject = `Raffay` (34 test files) | Train/Val Pool = `Areeb`, `Towsif` (54 train, 14 val)
- **Fold 2**: Held-out Test Subject = `Towsif` (34 test files) | Train/Val Pool = `Areeb`, `Raffay` (54 train, 14 val)

> [!IMPORTANT]
> **Zero Data Leakage Guarantee**: Entire CSV recordings stay intact within their assigned subject partition. Frame-level random shuffling is strictly forbidden.

---

## Room & Layout Invariance via Target-Relative Centering

To eliminate dependency on room dimensions and sensor mounting locations, every frame undergoes **target-relative spatial centering**:

$$\Delta x = x - \text{median}(x), \quad \Delta y = y - \text{median}(y)$$

1. **Absolute Coordinates Stripped**: Raw absolute $X$ and $Y$ spatial coordinates are completely removed from features, preventing the neural network from memorizing static room boundaries.
2. **Elevation Retained**: Vertical height $z$ (or relative vertical displacement $\Delta z = z - \text{centroid}_z$) is preserved to track height drops during a fall.
3. **Point Resampling ($N=32$)**: Reflection points per frame are standardized to $N=32$ points using uniform sampling/oversampling.

---

## Understanding Point Resampling ($N=32$) & Doppler Speed ($v$)

### How $N=32$ Point Resampling Works
mmWave radar CFAR detection yields a variable number of reflection points per frame (ranging from 5 to 60+ points depending on subject posture, distance, and SNR). To feed these point clouds into fixed-shape neural network architectures:
- **Oversampling (Points $< 32$)**: Frames with fewer than 32 points duplicate existing detections via uniform replacement.
- **Subsampling (Points $> 32$)**: Frames with more than 32 points are uniformly sampled to exactly 32 points, prioritizing high SNR reflections.
- A sliding window of 10 temporal frames produces a normalized sequence of $10 \times 32 = 320$ points per window.

### Role of Doppler Speed ($v$)
Each reflection point includes a radial Doppler velocity $v \text{ (m/s)}$, capturing the instantaneous speed of that point toward ($v > 0$) or away from ($v < 0$) the sensor. During a fall event, body segments (head/torso) accelerate rapidly, creating high Doppler velocity peaks ($|v| > 1.5 \text{ m/s}$ to $3.0 \text{ m/s}$), whereas normal activities (ADLs) exhibit lower, steady velocities.

### Impact of $N=32$ and Speed $v$ across Data Representations

1. **`rep1_doppler` (Micro-Doppler Velocity-Time Spectrogram $\to$ ResNet-18)**
   - The 32 Doppler velocities per frame are binned into 32 velocity channels across 10 temporal frames.
   - Creates a $32 \times 32$ grid mapping velocity vs. time. $N=32$ ensures consistent grid resolution across all frames.
   - Channel 0: Doppler intensity (point density / SNR). Channel 1: Doppler velocity spread. Channel 2: Frame-to-frame velocity acceleration ($\Delta v$).

2. **`rep2_projections` (2D Orthogonal Spatial Projections $\to$ ResNet-18)**
   - Projects the 320 window points into 3 orthogonal planes binned into $32 \times 32$ occupancy grids:
     - **Plane 1 ($\Delta x - z$)**: Elevation profile tracking height drop.
     - **Plane 2 ($\Delta x - \Delta y$)**: Ground plane profile tracking torso expansion upon impact.
     - **Plane 3 ($z - v$)**: **Height vs. Speed Profile**, directly coupling vertical height $z$ with Doppler falling speed $v$. High downward speed at low elevation forms a distinct fall signature.

3. **`rep3_pointset` (Native 3D Point Tensor $\to$ PointNet++)**
   - Directly feeds the continuous 320-point tensor of shape `(5, 320)` where each point carries features `[delta_x, delta_y, z, v, delta_z]`.
   - $N=32$ guarantees a fixed 320-point input tensor for PointNet++ Set Abstraction (SA) modules. Doppler speed $v$ acts as a continuous feature channel, allowing MLPs to learn spatial-kinematic correlations.

---

## Dynamic Representation Pipelines

The pipeline supports three distinct data representations, selectable via top-level configuration toggle:

```python
REPRESENTATION = "rep1_doppler"  # Options: 'rep1_doppler', 'rep2_projections', 'rep3_pointset'
```

| Representation Key | Deep Neural Network | Technical Description & Tensor Shape |
| :--- | :--- | :--- |
| **`rep1_doppler`** | **ResNet-18** | **Kinematic Micro-Doppler Map**: Discretizes Doppler velocity $v$ across 10 temporal frames into 2D intensity grid `(3, 32, 32)`. |
| **`rep2_projections`** | **ResNet-18** | **Spatial 2D Orthogonal Projections**: Occupancy grids ($\Delta x$-$z$ Elevation, $\Delta x$-$\Delta y$ Ground, $z$-$v$ Height-Doppler) `(3, 32, 32)`. |
| **`rep3_pointset`** | **PointNet++** | **Geometric Native 3D Point Tensor**: Continuous 3D point vectors `[delta_x, delta_y, z, v, delta_z]` `(5, 320)`. |

---

## Repository Architecture

```text
mmWaves-Fall-Detection/
├── src/                                  # Shared Core Python Package
│   ├── config.py                         # Central Dataset Registries & Hyperparameters
│   ├── dataset_parser.py                 # Ingestion of 102 CSV files & 3-Fold Train/Val/Test LOSO Splits
│   ├── transforms.py                     # Target-Relative Centering & Dynamic Representation Selector
│   ├── dataset_utils.py                  # Window Extraction, Resampling (N=32), Scaling
│   ├── models.py                         # ResNet-18 & PointNet++ PyTorch Architectures
│   ├── trainer.py                        # Class-Weighted Loss & Threshold Tuning Engine
│   ├── train_loso.py                     # 3-Fold Train/Val/Test LOSO Cross-Validation Engine
│   ├── benchmark_eval.py                 # Out-of-Fold Metrics & Inference Latency Benchmark
│   └── pipeline.py                       # Unified Master Pipeline API
│
├── notebooks/                            # 5 Standardized Jupyter Notebooks
│   ├── 00_train_val_test_split.ipynb     # Step 00: Dataset Integrity & 3-Way LOSO Split
│   ├── 01_dataset_explorer.ipynb         # Step 01: Multi-Representation Exploration
│   ├── 02_data_preparation.ipynb         # Step 02: Preprocessing, Centering & Windowing
│   ├── 03_train_and_evaluate_models.ipynb# Step 03: LOSO Model Training & Representation Benchmark
│   └── 04_hyperparameter_tuning_and_ablation.ipynb # Step 04: Interactive Tuning & Sensitivity Analysis
│
├── datasets/                             # Raw & Preprocessed Data
│   └── mmwave-radar-fall-detection/      # TI IWR6843 CSV Data Files (51 Fall, 51 Not)
│
├── models/                               # Models & Benchmarks
│   ├── checkpoints/                      # Model Checkpoint Weights (.pth)
│   └── benchmarks/                       # JSON Benchmark Summary & High-Res PNG Figures
│
└── README.md                             # Repository Documentation
```

---

## Notebook Execution Workflow

All notebooks (`00` through `04`) run seamlessly out of the box:

1. **[`00_train_val_test_split.ipynb`](notebooks/00_train_val_test_split.ipynb)**: Audits all 102 CSV files, constructs 3-fold 3-way Train/Val/Test LOSO splits (54 Train, 14 Val, 34 Test), and verifies zero data leakage.
2. **[`01_dataset_explorer.ipynb`](notebooks/01_dataset_explorer.ipynb)**: Demonstrates target-relative centering ($\Delta x, \Delta y$) and displays point clouds across 3 distinct representations (`rep1_doppler`, `rep2_projections`, `rep3_pointset`).
3. **[`02_data_preparation.ipynb`](notebooks/02_data_preparation.ipynb)**: Performs spatial centering, window extraction, fold-isolated standard scaling, and generates dataset split manifests.
4. **[`03_train_and_evaluate_models.ipynb`](notebooks/03_train_and_evaluate_models.ipynb)**: Executes 3-fold LOSO cross-validation across all representations, generating out-of-fold metrics and representation comparisons.
5. **[`04_hyperparameter_tuning_and_ablation.ipynb`](notebooks/04_hyperparameter_tuning_and_ablation.ipynb)**: Provides an interactive playground to adjust `lr`, `epochs`, `batch_size`, and evaluate decision threshold tuning ($t^*$).

---

## Python CLI Execution

```bash
# Execute 3-Fold Train/Val/Test LOSO Benchmark across all representations via CLI
python -m src.pipeline

# Execute individual LOSO training run
python -m src.train_loso

# Run representation benchmark evaluation & latency benchmark
python -m src.benchmark_eval
```

---

## License & Acknowledgments

This research utilizes the TI IWR6843 mmWave Radar Fall Detection Dataset provided by Texas Instruments radar evaluation contributors.
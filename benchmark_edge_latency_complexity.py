import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# --- Model Architectures ---

class ResNet18CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        feat = torch.flatten(self.features(x), 1)
        return self.fc(feat)

class PointNetClsMMFall(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        return self.fc(x)

class PointNetFallDetector(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0].view(x.size(0), -1)
        out = self.relu(self.fc1(self.dropout(global_feat)))
        return self.fc2(out)


# --- Transformation Benchmark Functions ---

def benchmark_rep1_spectrogram(clip_points, num_bins=64):
    """Transform raw points (10 frames x P points x 4 features) into 3x64x64 Micro-Doppler Spectrogram."""
    v_all = clip_points[..., 3]
    t_idx = np.repeat(np.arange(10), clip_points.shape[1])
    v_flat = v_all.flatten()
    
    # 2D Histogram binning: Velocity (-3.0 to 3.0 m/s) vs Time (10 frames)
    hist, _, _ = np.histogram2d(v_flat, t_idx, bins=[num_bins, num_bins], range=[[-3.0, 3.0], [0, 10]])
    norm_hist = hist / (hist.max() + 1e-6)
    
    spec = np.stack([norm_hist, norm_hist * 0.8, norm_hist * 0.5], axis=0).astype(np.float32)
    return spec

def benchmark_rep2_projections(clip_points, num_bins=64):
    """Transform raw points into 3x64x64 Orthogonal Projections (XZ and XY grids)."""
    x_flat = clip_points[..., 0].flatten()
    y_flat = clip_points[..., 1].flatten()
    z_flat = clip_points[..., 2].flatten()
    
    xz_hist, _, _ = np.histogram2d(x_flat, z_flat, bins=[num_bins, num_bins], range=[[-2.0, 2.0], [-1.0, 2.0]])
    xy_hist, _, _ = np.histogram2d(x_flat, y_flat, bins=[num_bins, num_bins], range=[[-2.0, 2.0], [0.0, 5.0]])
    
    proj_xz = xz_hist / (xz_hist.max() + 1e-6)
    proj_xy = xy_hist / (xy_hist.max() + 1e-6)
    proj_combined = (proj_xz + proj_xy) * 0.5
    
    proj = np.stack([proj_xz, proj_xy, proj_combined], axis=0).astype(np.float32)
    return proj

def benchmark_rep3_pointset(clip_points):
    """Format raw points into unordered point matrix (Channels x N_points)."""
    pts = clip_points.reshape(-1, clip_points.shape[-1]).T.astype(np.float32)
    return pts


# --- Estimation Functions ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_tensor):
    """Estimate approximate FLOPs for 2D CNN or 1D PointNet."""
    flops = 0
    def conv_hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
            flops += output.numel() * kernel_ops * 2
        elif isinstance(module, nn.Conv1d):
            kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
            flops += output.numel() * kernel_ops * 2
        elif isinstance(module, nn.Linear):
            flops += module.in_features * module.out_features * 2 * input[0].shape[0]

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            hooks.append(layer.register_forward_hook(conv_hook))
            
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
        
    for h in hooks:
        h.remove()
    return flops


def run_benchmark():
    print("=" * 80)
    print(" SUBQUESTION 3: EDGE COMPUTATIONAL COMPLEXITY & PROCESSING LATENCY BENCHMARK ")
    print("=" * 80)
    
    models_dir = Path('models')
    preproc_dir = Path('datasets/preprocessed')
    
    datasets_config = {
        'mmfall': {
            'x_clean': preproc_dir / 'X_mmfall_clean_balanced.npy',
            'rep1_ckpt': models_dir / 'resnet18_rep1_spectrogram_mmfall.pth',
            'rep2_ckpt': models_dir / 'resnet18_rep2_projections_mmfall.pth',
            'rep3_ckpt': models_dir / 'pointnet_rep3_pointset_mmfall.pth',
            'rep3_cls': PointNetClsMMFall,
            'pts_per_frame': 64,
            'in_channels_pointnet': 5
        },
        'mmwave': {
            'x_clean': preproc_dir / 'X_ti_clean_balanced.npy',
            'rep1_ckpt': models_dir / 'resnet18_rep1_spectrogram_ti.pth',
            'rep2_ckpt': models_dir / 'resnet18_rep2_projections_ti.pth',
            'rep3_ckpt': models_dir / 'pointnet_rep3_pointset_ti.pth',
            'rep3_cls': PointNetFallDetector,
            'pts_per_frame': 32,
            'in_channels_pointnet': 5
        },
        'combined': {
            'x_clean': preproc_dir / 'X_combined_clean_balanced.npy',
            'rep1_ckpt': models_dir / 'resnet18_rep1_spectrogram_combined.pth',
            'rep2_ckpt': models_dir / 'resnet18_rep2_projections_combined.pth',
            'rep3_ckpt': models_dir / 'pointnet_rep3_pointset_combined.pth',
            'rep3_cls': PointNetFallDetector,
            'pts_per_frame': 64,
            'in_channels_pointnet': 5
        }
    }
    
    results = {}
    
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmark Execution Target: CPU ({device_cpu}), GPU ({device_gpu})")
    
    for ds_name, config in datasets_config.items():
        print(f"\n>>> Benchmarking Dataset Configuration: [{ds_name.upper()}]")
        results[ds_name] = {}
        
        # Load sample raw clip
        if config['x_clean'].exists():
            X_clean = np.load(config['x_clean'])
            sample_clip = X_clean[0] # (10, P, F)
        else:
            P = config['pts_per_frame']
            F = config['in_channels_pointnet']
            sample_clip = np.random.randn(10, P, F).astype(np.float32)
            
        P_total = 10 * config['pts_per_frame']
        
        # 1. Benchmark Preprocessing Latency (ms / clip)
        N_ITER = 500
        
        # Rep 1: Spectrogram
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = benchmark_rep1_spectrogram(sample_clip)
        rep1_preproc_ms = ((time.perf_counter() - t0) / N_ITER) * 1000.0
        
        # Rep 2: Projections
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = benchmark_rep2_projections(sample_clip)
        rep2_preproc_ms = ((time.perf_counter() - t0) / N_ITER) * 1000.0
        
        # Rep 3: PointSet
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = benchmark_rep3_pointset(sample_clip)
        rep3_preproc_ms = ((time.perf_counter() - t0) / N_ITER) * 1000.0
        
        reps_setup = [
            ('Rep1_Spectrogram', config['rep1_ckpt'], ResNet18CNN(in_channels=3), rep1_preproc_ms, torch.randn(1, 3, 64, 64)),
            ('Rep2_Projections', config['rep2_ckpt'], ResNet18CNN(in_channels=3), rep2_preproc_ms, torch.randn(1, 3, 64, 64)),
            ('Rep3_PointSet', config['rep3_ckpt'], config['rep3_cls'](in_channels=config['in_channels_pointnet']), rep3_preproc_ms, torch.randn(1, config['in_channels_pointnet'], P_total))
        ]
        
        for rep_name, ckpt_path, model, preproc_ms, sample_input in reps_setup:
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                model.load_state_dict(ckpt)
                ckpt_size_kb = ckpt_path.stat().st_size / 1024.0
            else:
                ckpt_size_kb = 0.0
                
            model.eval()
            params = count_parameters(model)
            flops = estimate_flops(model, sample_input)
            
            # --- Inference Latency (CPU) ---
            model.to(device_cpu)
            inp_cpu = sample_input.to(device_cpu)
            inp_batch_cpu = sample_input.repeat(16, 1, 1, 1) if inp_cpu.ndim == 4 else sample_input.repeat(16, 1, 1)
            inp_batch_cpu = inp_batch_cpu.to(device_cpu)
            
            # Warmup
            with torch.no_grad():
                for _ in range(20): _ = model(inp_cpu)
                
            # Measure CPU Latency (Single Sample B=1)
            t0 = time.perf_counter()
            N_INF = 300
            with torch.no_grad():
                for _ in range(N_INF):
                    _ = model(inp_cpu)
            cpu_lat_b1_ms = ((time.perf_counter() - t0) / N_INF) * 1000.0
            
            # Measure CPU Latency (Batch B=16)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(N_INF):
                    _ = model(inp_batch_cpu)
            cpu_lat_b16_ms = ((time.perf_counter() - t0) / N_INF) * 1000.0
            
            # --- End-to-End Latency & Viability ---
            total_e2e_ms = preproc_ms + cpu_lat_b1_ms
            realtime_budget_ms = 100.0 # 10 Hz clip window budget (100 ms)
            realtime_viable = total_e2e_ms <= realtime_budget_ms
            speedup_vs_rep1 = rep1_preproc_ms / max(preproc_ms, 1e-6)
            
            rep_res = {
                'preprocessing_latency_ms': round(preproc_ms, 4),
                'cpu_inference_latency_b1_ms': round(cpu_lat_b1_ms, 4),
                'cpu_inference_latency_b16_ms': round(cpu_lat_b16_ms, 4),
                'total_e2e_latency_ms': round(total_e2e_ms, 4),
                'realtime_viable_10hz': realtime_viable,
                'parameters_count': params,
                'parameters_k': round(params / 1000.0, 2),
                'flops_count': flops,
                'flops_m': round(flops / 1e6, 2),
                'checkpoint_size_kb': round(ckpt_size_kb, 2),
                'preproc_speedup_vs_rep1': round(speedup_vs_rep1, 2)
            }
            results[ds_name][rep_name] = rep_res
            
            print(f"  [{rep_name:<20}] Params: {rep_res['parameters_k']}K | FLOPs: {rep_res['flops_m']}M | Preproc: {rep_res['preprocessing_latency_ms']} ms | CPU Inf (B=1): {rep_res['cpu_inference_latency_b1_ms']} ms | E2E: {rep_res['total_e2e_latency_ms']} ms | Viable: {realtime_viable}")
            
    # Save output JSON
    out_file = models_dir / 'edge_latency_complexity_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved latency & complexity benchmark results to: {out_file}")
    
    # Print Synthesis Markdown Table
    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK TABLE: SUBQUESTION 3 (COMBINED DATASET FOCUS) ")
    print("=" * 80)
    print("| Representation Format | Preproc Latency (ms) | CPU Inference (ms) | Total E2E Latency (ms) | Real-Time Viable (10 Hz) | Parameters (K) | FLOPs (M) | Model Size (KB) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for rep, r in results['combined'].items():
        print(f"| **{rep.replace('_', ' ')}** | {r['preprocessing_latency_ms']:.3f} ms | {r['cpu_inference_latency_b1_ms']:.3f} ms | {r['total_e2e_latency_ms']:.3f} ms | {'YES' if r['realtime_viable_10hz'] else 'NO'} | {r['parameters_k']} K | {r['flops_m']} M | {r['checkpoint_size_kb']} KB |")

if __name__ == '__main__':
    run_benchmark()

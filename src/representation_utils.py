import numpy as np
import scipy.ndimage as ndimage

def generate_spectrogram_rep(X_raw: np.ndarray, variant: str = "raw") -> np.ndarray:
    """
    Rep 1: Micro-Doppler Spectrogram (N, 3, 64, 64)
    Variants: 'raw' | 'log_scaled' | 'smooth'
    """
    N = len(X_raw)
    specs = np.zeros((N, 3, 64, 64), dtype=np.float32)
    for i in range(N):
        clip = X_raw[i] # (10, 32, 4)
        for t in range(10):
            v_vals = clip[t, :, 3]
            hist, _ = np.histogram(v_vals, bins=64, range=(-3.0, 3.0))
            hist_ke, _ = np.histogram(v_vals, bins=64, range=(-3.0, 3.0), weights=v_vals**2)
            
            specs[i, 0, :, t*6:(t+1)*6] = np.tile(hist[:, None], (1, 6))
            specs[i, 1, :, t*6:(t+1)*6] = np.tile(hist_ke[:, None], (1, 6))
            specs[i, 2, :, t*6:(t+1)*6] = np.tile(np.abs(hist)[:, None], (1, 6))

        if variant == "log_scaled":
            specs[i] = np.log1p(specs[i])
        elif variant == "smooth":
            for c in range(3):
                specs[i, c] = ndimage.gaussian_filter(specs[i, c], sigma=1.0)
    return specs

def generate_projections_rep(X_raw: np.ndarray, variant: str = "raw") -> np.ndarray:
    """
    Rep 2: Orthogonal Projections (N, 3, 64, 64)
    Variants: 'raw' | 'gaussian' | 'trail'
    """
    N = len(X_raw)
    projs = np.zeros((N, 3, 64, 64), dtype=np.float32)
    for i in range(N):
        if variant == "trail":
            # Multi-frame temporal accumulation with exponential decay
            h_xy = np.zeros((64, 64), dtype=np.float32)
            h_xz = np.zeros((64, 64), dtype=np.float32)
            h_yz = np.zeros((64, 64), dtype=np.float32)
            for t in range(10):
                decay = 0.8 ** (9 - t)
                f_pts = X_raw[i, t] # (32, 4)
                xy, _, _ = np.histogram2d(f_pts[:, 0], f_pts[:, 1], bins=64, range=[[-2.0, 2.0], [0.0, 6.0]])
                xz, _, _ = np.histogram2d(f_pts[:, 0], f_pts[:, 2], bins=64, range=[[-2.0, 2.0], [-0.5, 2.2]])
                yz, _, _ = np.histogram2d(f_pts[:, 1], f_pts[:, 2], bins=64, range=[[0.0, 6.0], [-0.5, 2.2]])
                h_xy += xy * decay
                h_xz += xz * decay
                h_yz += yz * decay
        else:
            pts = X_raw[i].reshape(-1, 4)
            h_xy, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=64, range=[[-2.0, 2.0], [0.0, 6.0]])
            h_xz, _, _ = np.histogram2d(pts[:, 0], pts[:, 2], bins=64, range=[[-2.0, 2.0], [-0.5, 2.2]])
            h_yz, _, _ = np.histogram2d(pts[:, 1], pts[:, 2], bins=64, range=[[0.0, 6.0], [-0.5, 2.2]])

        if variant == "gaussian":
            h_xy = ndimage.gaussian_filter(h_xy, sigma=1.5)
            h_xz = ndimage.gaussian_filter(h_xz, sigma=1.5)
            h_yz = ndimage.gaussian_filter(h_yz, sigma=1.5)

        projs[i, 0] = h_xy
        projs[i, 1] = h_xz
        projs[i, 2] = h_yz
    return projs

def generate_pointnet_rep(X_raw: np.ndarray, variant: str = "raw") -> np.ndarray:
    """
    Rep 3: Native 3D Point Set (N, 5, 320)
    Variants: 'raw' | 'centered' | 'normalized'
    """
    N = len(X_raw)
    pts = X_raw.reshape(N, 320, 4)
    
    if variant == "centered":
        # Subtract mean frame position to isolate relative motion
        centroids = np.mean(pts[:, :, :3], axis=1, keepdims=True)
        pts[:, :, :3] = pts[:, :, :3] - centroids
    elif variant == "normalized":
        # Scale coordinates into unit box [-1.0, 1.0]
        pts[:, :, 0] = np.clip(pts[:, :, 0] / 2.0, -1.0, 1.0)
        pts[:, :, 1] = np.clip((pts[:, :, 1] - 3.0) / 3.0, -1.0, 1.0)
        pts[:, :, 2] = np.clip((pts[:, :, 2] - 0.85) / 1.35, -1.0, 1.0)

    t_channel = np.tile(np.repeat(np.linspace(0, 1, 10), 32), (N, 1))
    rep3 = np.stack([pts[:, :, 0], pts[:, :, 1], pts[:, :, 2], pts[:, :, 3], t_channel], axis=1).astype(np.float32)
    return rep3

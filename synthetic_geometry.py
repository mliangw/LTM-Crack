"""LTM-Crack reproducibility implementation."""

import numpy as np
import cv2
from scipy.interpolate import splprep, splev

MM_PER_PIXEL = 0.01278

def generate_curved_path(img_h=600, img_w=800, n_control=None, n_samples=500):
    """Generate a curved, x-ordered cubic B-spline centerline."""
    if n_control is None:
        n_control = np.random.randint(4, 7)
    margin_x = img_w * 0.12
    margin_y = img_h * 0.12
    usable_x = img_w - 2 * margin_x
    usable_y = img_h - 2 * margin_y
                                                                 
    cx = np.linspace(margin_x, img_w - margin_x, n_control)
    cx += np.random.uniform(-usable_x * 0.04, usable_x * 0.04, n_control)
    cx = np.sort(cx)                                             
                                                     
    cy_start = img_h / 2 + np.random.uniform(-usable_y * 0.15, usable_y * 0.15)
    steps = np.random.uniform(-usable_y * 0.12, usable_y * 0.12, n_control)
    steps[0] = 0
    cy = cy_start + np.cumsum(steps)
    cy = np.clip(cy, margin_y * 1.5, img_h - margin_y * 1.5)
                            
    k = min(3, n_control - 1)
    tck, _ = splprep([cx, cy], s=0, k=k)
    u_new = np.linspace(0, 1, n_samples)
    x_new, y_new = splev(u_new, tck)
    return np.column_stack([x_new, y_new])

def compute_normals(path):
    """Compute unit normals from finite-difference centerline tangents."""
    n = len(path)
    tangents = np.zeros_like(path)
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    tangents[1:-1] = path[2:] - path[:-2]
                                          
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
               
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-8)
    return normals

def generate_width_profile(n_points, target_width_px, taper_frac=0.15):
    """Apply one or two multiplicative sinusoidal components and cosine tip tapering."""
    t = np.linspace(0, 1, n_points)
                
    widths = np.ones(n_points) * target_width_px
                                             
    n_harmonics = np.random.randint(1, 3)
    for _ in range(n_harmonics):
        freq = np.random.uniform(0.8, 2.5)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.03, 0.08)
        widths *= (1.0 + amp * np.sin(freq * 2 * np.pi * t + phase))
                                  
    taper_len = max(int(n_points * taper_frac), 3)
    min_tip = max(1.5, target_width_px * 0.08)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
    widths[:taper_len] = min_tip + (widths[:taper_len] - min_tip) * ramp
    widths[-taper_len:] = min_tip + (widths[-taper_len:] - min_tip) * ramp[::-1]
    return widths

def generate_asymmetry(n_points):
    """Generate a smooth left-width fraction bounded between 0.30 and 0.70."""
    t = np.linspace(0, 1, n_points)
    base = np.random.uniform(0.38, 0.62)
    freq = np.random.uniform(0.5, 2.0)
    amp = np.random.uniform(0.05, 0.15)
    phase = np.random.uniform(0, 2 * np.pi)
    left_ratio = base + amp * np.sin(freq * 2 * np.pi * t + phase)
    return np.clip(left_ratio, 0.30, 0.70)

def add_boundary_noise(offsets, amplitude=0.8, smooth_kernel=5):
    """Smooth Gaussian noise, add it to offsets, and impose a positive offset floor."""
    n = len(offsets)
    noise = amplitude * np.random.randn(n)
                                        
    kernel = np.ones(smooth_kernel) / smooth_kernel
    noise = np.convolve(noise, kernel, mode='same')
    noisy = offsets + noise
    return np.maximum(noisy, 0.5)

def create_mask(path, normals, left_offsets, right_offsets, img_h, img_w):
    """Rasterize the two offset boundaries into a binary polygon mask."""
    left_boundary = path + normals * left_offsets[:, np.newaxis]
    right_boundary = path - normals * right_offsets[:, np.newaxis]
    polygon = np.vstack([left_boundary, right_boundary[::-1]])
                                               
    polygon[:, 0] = np.clip(polygon[:, 0], 0, img_w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, img_h - 1)
    polygon = polygon.astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask, left_boundary, right_boundary

def generate_single_crack(img_h, img_w, target_width_px, seed=None):
    """Return a binary mask and a pre-perturbation design reference. Noise and rasterization can change the final mask maximum."""
    if seed is not None:
        np.random.seed(seed)
    n_samples = 500
                    
    path = generate_curved_path(img_h, img_w, n_samples=n_samples)
    normals = compute_normals(path)
    n_points = len(path)
                                                 
    widths = generate_width_profile(n_points, target_width_px)
                             
    left_ratio = generate_asymmetry(n_points)
    left_offsets_gt = widths * left_ratio
    right_offsets_gt = widths * (1.0 - left_ratio)
                                                                            
                                                           
    noise_amp = np.clip(target_width_px * 0.03, 0.5, 2.0)
    left_actual = add_boundary_noise(left_offsets_gt, amplitude=noise_amp)
    right_actual = add_boundary_noise(right_offsets_gt, amplitude=noise_amp)
                    
    mask, left_bnd, right_bnd = create_mask(
        path, normals, left_actual, right_actual, img_h, img_w
    )
                                                                            
    taper_margin = int(n_points * 0.20)
    interior = slice(taper_margin, n_points - taper_margin)
    max_interior_idx = np.argmax(widths[interior])
    max_idx = max_interior_idx + taper_margin
    gt_width_px = widths[max_idx]
    gt_cx, gt_cy = path[max_idx]
    return mask, float(gt_width_px), float(gt_cx), float(gt_cy), path, widths

def create_overlay(mask, path, gt_cx, gt_cy, gt_width_px):
    """Create an optional visualization of the mask, design centerline, and reference site."""
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                                                         
    crack_region = mask > 0
    vis[crack_region] = (vis[crack_region] * 0.6 + np.array([80, 40, 40]) * 0.4).astype(np.uint8)
                                 
    pts = path.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=False, color=(0, 200, 0), thickness=1)
                                                     
    cx, cy = int(gt_cx), int(gt_cy)
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(vis, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 1)
    cv2.line(vis, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 1)
                         
    gt_w_mm = gt_width_px * MM_PER_PIXEL
    label = f"GT: {gt_width_px:.1f}px = {gt_w_mm:.4f}mm"
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis

"""Compare candidate selection with sampled skeleton-wide width searches."""

import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

from baseline_width_methods import (get_boundary_points, pca_direction, measure_op, measure_orthoboundary, measure_esd, measure_eob)

MM_PER_PIXEL = 0.01278
NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

CONFIGS = [
    "MCM_Proposed",                 
    "OP_Exhaust",                    
    "OB_Exhaust",                    
    "ESD_Exhaust",                    
    "EOB_Exhaust",                    
    "EDT_SkelMax",                
    "OP_Deg2",                             
    "OB_Deg2",                             
    "ESD_Deg2",                             
    "EOB_Deg2",                             
]

CONFIG_LABELS = {
    "MCM_Proposed": "MCM (Proposed)",
    "OP_Exhaust":   "OP — All Skeleton",
    "OB_Exhaust":   "OB — All Skeleton",
    "ESD_Exhaust":  "ESD — All Skeleton",
    "EOB_Exhaust":  "EOB — All Skeleton",
    "EDT_SkelMax":  "EDT — Skeleton Max",
    "OP_Deg2":      "OP — Deg≤2 Skeleton",
    "OB_Deg2":      "OB — Deg≤2 Skeleton",
    "ESD_Deg2":     "ESD — Deg≤2 Skeleton",
    "EOB_Deg2":     "EOB — Deg≤2 Skeleton",
}

GROUP_MAIN = ["MCM_Proposed", "OP_Exhaust", "OB_Exhaust",
              "ESD_Exhaust", "EOB_Exhaust", "EDT_SkelMax"]
GROUP_DEG2 = ["MCM_Proposed", "OP_Deg2", "OB_Deg2",
              "ESD_Deg2", "EOB_Deg2", "EDT_SkelMax"]

def skeleton_degree(skel):
    ys, xs = np.where(skel)
    if ys.size == 0:
        return ys, xs, np.array([], dtype=np.int32)
    h, w = skel.shape
    skel_u8 = skel.astype(np.uint8)
    deg = np.zeros_like(ys, dtype=np.int32)
    for idx in range(len(ys)):
        y, x = int(ys[idx]), int(xs[idx])
        c = 0
        for dy, dx in NEI8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                c += int(skel_u8[yy, xx] > 0)
        deg[idx] = c
    return ys, xs, deg

def march_to_boundary(mask_bool, start_xy, dir_xy, step=0.5, max_steps=4000):
    h, w = mask_bool.shape
    x, y = float(start_xy[0]), float(start_xy[1])
    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    last_in = (x, y)
    for _ in range(max_steps):
        x += dx * step
        y += dy * step
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            break
        if bool(mask_bool[iy, ix]):
            last_in = (x, y)
        else:
            break
    return last_in

def boundary_next_is_outside(mask_bool, p, dir_xy, step=1.0):
    h, w = mask_bool.shape
    x = p[0] + dir_xy[0] * step
    y = p[1] + dir_xy[1] * step
    ix, iy = int(round(x)), int(round(y))
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return True
    return not bool(mask_bool[iy, ix])

def measure_mcm(mask_bool, center_xy, expected_min_px,
                coarse_step_deg=5, fine_step_deg=1,
                fine_halfspan_deg=5, step_px=0.5):
    cx, cy = center_xy
    best = None

    def eval_theta(theta):
        nonlocal best
        dx = float(np.cos(theta))
        dy = float(np.sin(theta))
        p1 = march_to_boundary(mask_bool, (cx, cy), (dx, dy), step=step_px)
        p2 = march_to_boundary(mask_bool, (cx, cy), (-dx, -dy), step=step_px)
        L = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        if L < 1.0:
            return
        if L < 0.90 * expected_min_px:
            return
        if not boundary_next_is_outside(mask_bool, p1, (dx, dy), step=1.0):
            return
        if not boundary_next_is_outside(mask_bool, p2, (-dx, -dy), step=1.0):
            return
        if (best is None) or (L < best[0]):
            best = (L, float(np.degrees(theta)))

    for deg in range(0, 180, coarse_step_deg):
        eval_theta(np.radians(deg))
    if best is None:
        return None
    best_theta = best[1]
    for deg in range(int(round(best_theta - fine_halfspan_deg)),
                     int(round(best_theta + fine_halfspan_deg)) + 1,
                     fine_step_deg):
        eval_theta(np.radians(deg % 180))
    if best is None:
        return None
    return best[0]

def exhaustive_max_width(method_func, mask, skel_points_xy, boundary_pts,
                         kernel_radius=2, step=3):
    """Sample skeleton coordinates, evaluate a width estimator, and return its maximum valid result."""
    max_w = None
    n_pts = len(skel_points_xy)
                  
    h, w = mask.shape
    for i in range(0, n_pts, step):
        cx, cy = skel_points_xy[i]
        center_xy = (float(cx), float(cy))

        local_skel = np.zeros((h, w), dtype=bool)
        r = kernel_radius + 1
        y0 = max(0, int(cy) - r * 3)
        y1 = min(h, int(cy) + r * 3 + 1)
        x0 = max(0, int(cx) - r * 3)
        x1 = min(w, int(cx) + r * 3 + 1)
        for j in range(max(0, i - r * 3), min(n_pts, i + r * 3)):
            px, py = skel_points_xy[j]
            if x0 <= px < x1 and y0 <= py < y1:
                local_skel[int(py), int(px)] = True

        try:
            val = method_func(mask, local_skel, boundary_pts,
                              center_xy, kernel_radius=kernel_radius)
            if val is not None and (max_w is None or val > max_w):
                max_w = val
        except Exception:
            pass
    return max_w

def load_groundtruth(csv_path):
    records = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id'].strip()
            records[img_id] = {
                'level': row['level'].strip(),
                'gt_width_mm': float(row['gt_max_width_mm']),
                'gt_width_px': float(row['gt_max_width_px']),
                'gt_cx': float(row['gt_center_x']),
                'gt_cy': float(row['gt_center_y']),
            }
    return records

def main():
    parser = argparse.ArgumentParser(
        description="Candidate Point Selection Importance — Synthetic Cracks")
    parser.add_argument('--syn-dir', '--syn_dir', type=str, required=True)
    parser.add_argument('--output-dir', '--output_dir', type=str, default="outputs/candidate_search")
    parser.add_argument('--kernel-radius', '--kernel_radius', type=int, default=2)
    parser.add_argument('--sample-step', '--sample_step', type=int, default=3,
                        help="Skeleton sampling step; use 1 for all eligible skeleton points")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional plot generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k = MM_PER_PIXEL

    gt_path = os.path.join(args.syn_dir, 'groundtruth.csv')
    gt_records = load_groundtruth(gt_path)
    img_dir = os.path.join(args.syn_dir, 'images')
    img_ids = sorted(gt_records.keys())
    print(f"Synthetic masks: {len(img_ids)} images")
    print(f"Skeleton sampling step: {args.sample_step}")
    print("=" * 80)

    method_funcs = {
        'OP': measure_op,
        'OB': measure_orthoboundary,
        'ESD': measure_esd,
        'EOB': measure_eob,
    }

    rows = []
    t0 = time.time()

    for n, img_id in enumerate(img_ids):
        gt = gt_records[img_id]
        img_path = os.path.join(img_dir, img_id)
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        mask = raw > 127
        if mask.sum() < 10:
            continue

        dist = distance_transform_edt(mask)
        skel = skeletonize(mask)
        ys, xs, deg = skeleton_degree(skel)
        if ys.size == 0:
            continue

        boundary_pts = get_boundary_points(mask)
        gt_mm = gt['gt_width_mm']

        all_pts = list(zip(xs.tolist(), ys.tolist()))
        deg2_mask = deg <= 2
        deg2_pts = list(zip(xs[deg2_mask].tolist(), ys[deg2_mask].tolist()))

        if n % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n+1}/{len(img_ids)}] {img_id} "
                  f"skel={len(all_pts)} deg2={len(deg2_pts)} "
                  f"gt={gt_mm:.4f}mm  ({elapsed:.0f}s)")

        row = {
            'image_id': img_id,
            'level': gt['level'],
            'gt_width_mm': gt_mm,
            'skel_points': len(all_pts),
            'deg2_points': len(deg2_pts),
        }

        dvals = dist[ys, xs]
        if np.any(deg2_mask):
            ys2, xs2, dv2 = ys[deg2_mask], xs[deg2_mask], dvals[deg2_mask]
            idx = int(np.argmax(dv2))
            cx_mcm, cy_mcm = float(xs2[idx]), float(ys2[idx])
            r_px = float(dv2[idx])
        else:
            idx = int(np.argmax(dvals))
            cx_mcm, cy_mcm = float(xs[idx]), float(ys[idx])
            r_px = float(dvals[idx])

        expected_min = max(2.0 * r_px, 1.0)
        mcm_px = measure_mcm(mask, (cx_mcm, cy_mcm), expected_min)
        row['MCM_Proposed'] = mcm_px * k if mcm_px is not None else None

        if np.any(deg2_mask):
            edt_max = float(dv2[int(np.argmax(dv2))])
        else:
            edt_max = float(dvals[int(np.argmax(dvals))])
        row['EDT_SkelMax'] = 2.0 * edt_max * k

        for mname, mfunc in method_funcs.items():
                   
            w_all = exhaustive_max_width(
                mfunc, mask, all_pts, boundary_pts,
                kernel_radius=args.kernel_radius,
                step=args.sample_step)
            row[f'{mname}_Exhaust'] = w_all * k if w_all is not None else None

            if deg2_pts:
                w_deg2 = exhaustive_max_width(
                    mfunc, mask, deg2_pts, boundary_pts,
                    kernel_radius=args.kernel_radius,
                    step=args.sample_step)
                row[f'{mname}_Deg2'] = w_deg2 * k if w_deg2 is not None else None
            else:
                row[f'{mname}_Deg2'] = row[f'{mname}_Exhaust']

        rows.append(row)

    elapsed_total = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed_total:.1f}s")

    csv_path = os.path.join(args.output_dir, 'candidate_importance_results.csv')
    base_fields = ['image_id', 'level', 'gt_width_mm',
                   'skel_points', 'deg2_points']
    config_fields = [c for c in CONFIGS]
    fieldnames = base_fields + config_fields

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            writer.writerow(out)
    print(f"CSV: {csv_path}")

    ref = np.array([r['gt_width_mm'] for r in rows])

    summary = "\n" + "=" * 100 + "\n"
    summary += "Candidate Point Selection Importance — Synthetic Cracks\n"
    summary += f"Total: {len(rows)} images, Sample step: {args.sample_step}\n"
    summary += f"Time: {elapsed_total:.1f}s\n"
    summary += "=" * 100 + "\n\n"

    summary += "[Main Comparison: MCM single-point vs Exhaustive search]\n"
    summary += (f"{'Config':<28} {'n':>4} {'MAE(mm)':>10} {'RMSE(mm)':>10} "
                f"{'R²':>8} {'Bias(mm)':>10} {'MaxErr':>10}\n")
    summary += "-" * 85 + "\n"

    all_metrics = {}
    for cfg in CONFIGS:
        vals = np.array([r.get(cfg, np.nan) if r.get(cfg) is not None
                         else np.nan for r in rows])
        valid = ~(np.isnan(ref) | np.isnan(vals))
        rv, pv = ref[valid], vals[valid]
        n = len(rv)
        if n == 0:
            all_metrics[cfg] = {}
            continue
        err = pv - rv
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((rv - rv.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        bias = np.mean(err)
        max_err = np.max(np.abs(err))
        met = {'n': n, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
               'Bias': bias, 'MaxErr': max_err}
        all_metrics[cfg] = met
        summary += (f"  {CONFIG_LABELS[cfg]:<26} {n:>4} {mae:>10.4f} "
                    f"{rmse:>10.4f} {r2:>8.4f} {bias:>10.4f} "
                    f"{max_err:>10.4f}\n")

    summary += "\n"

    summary += "[Effect of Degree≤2 Filtering on Exhaustive Methods]\n"
    for mname in ['OP', 'OB', 'ESD', 'EOB']:
        e_key = f'{mname}_Exhaust'
        d_key = f'{mname}_Deg2'
        if e_key in all_metrics and d_key in all_metrics:
            e = all_metrics[e_key]
            d = all_metrics[d_key]
            if 'MAE' in e and 'MAE' in d:
                delta = d['MAE'] - e['MAE']
                summary += (f"  {mname:<6}: All={e['MAE']:.4f} → "
                            f"Deg2={d['MAE']:.4f} (ΔMAE={delta:+.4f})\n")
    summary += "\n"

    levels = sorted(set(r['level'] for r in rows))
    for level in levels:
        sub = [r for r in rows if r['level'] == level]
        ref_l = np.array([r['gt_width_mm'] for r in sub])
        summary += f"--- {level} ({len(sub)} images) ---\n"
        for cfg in GROUP_MAIN:
            vals_l = np.array([r.get(cfg, np.nan) if r.get(cfg) is not None
                               else np.nan for r in sub])
            v = ~(np.isnan(ref_l) | np.isnan(vals_l))
            if v.sum() == 0:
                continue
            e = vals_l[v] - ref_l[v]
            mae_l = np.mean(np.abs(e))
            rmse_l = np.sqrt(np.mean(e**2))
            summary += (f"  {CONFIG_LABELS[cfg]:<28} "
                        f"MAE={mae_l:.4f}  RMSE={rmse_l:.4f}\n")
        summary += "\n"

    summary += "=" * 100 + "\n"
    print(summary)

    with open(os.path.join(args.output_dir, 'candidate_importance_summary.txt'),
              'w', encoding='utf-8') as f:
        f.write(summary)

    if args.no_plots:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 6.5,
            'figure.dpi': 600,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'axes.linewidth': 0.6,
            'mathtext.default': 'regular',
        })

        colors = {
            'MCM_Proposed': '#1f77b4',
            'OP_Exhaust':   '#d62728',
            'OB_Exhaust':   '#2ca02c',
            'ESD_Exhaust':  '#ff7f0e',
            'EOB_Exhaust':  '#9467bd',
            'EDT_SkelMax':  '#7f7f7f',
            'OP_Deg2':      '#d62728',
            'OB_Deg2':      '#2ca02c',
            'ESD_Deg2':     '#ff7f0e',
            'EOB_Deg2':     '#9467bd',
        }

        fig_w = 190 / 25.4
        fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_w * 0.56))
        axes = axes.flatten()
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        for i, cfg in enumerate(GROUP_MAIN):
            ax = axes[i]
            vals = np.array([r.get(cfg, np.nan) if r.get(cfg) is not None
                             else np.nan for r in rows])
            valid = ~np.isnan(vals)
            gv, pv = ref[valid], vals[valid]

            if len(gv) == 0:
                continue

            ax.scatter(gv, pv, c=colors[cfg], s=14, alpha=0.6,
                       edgecolors='white', linewidth=0.2, zorder=3)
            lims = [0, max(gv.max(), pv.max()) * 1.08]
            ax.plot(lims, lims, 'k--', linewidth=0.6, alpha=0.4, zorder=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

            err = pv - gv
            mae_i = np.mean(np.abs(err))
            ss_r = np.sum(err**2)
            ss_t = np.sum((gv - gv.mean())**2)
            r2_i = 1 - ss_r / ss_t if ss_t > 0 else 0

            ax.text(0.04, 0.96,
                    f'MAE = {mae_i:.4f} mm\n$R^2$ = {r2_i:.4f}\nn = {len(gv)}',
                    transform=ax.transAxes, fontsize=6.5,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', edgecolor='#cccccc',
                              alpha=0.9))

            ax.set_title(f'{panel_labels[i]} {CONFIG_LABELS[cfg]}',
                         fontsize=7.5, fontweight='bold', pad=4)
            if i >= 3:
                ax.set_xlabel('Design reference (mm)', fontsize=8)
            if i % 3 == 0:
                ax.set_ylabel('Measured max width (mm)', fontsize=8)
            ax.tick_params(axis='both', direction='in')

        plt.subplots_adjust(wspace=0.28, hspace=0.35)
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(args.output_dir,
                        f'fig_candidate_scatter.{ext}'),
                        dpi=600, facecolor='white')
        plt.close(fig)
        print("Scatter plot saved")

        fig2, ax2 = plt.subplots(figsize=(140 / 25.4, 70 / 25.4))
        x_pos = np.arange(len(GROUP_MAIN))
        w_bar = 0.35
        maes = [all_metrics.get(c, {}).get('MAE', 0) for c in GROUP_MAIN]
        rmses = [all_metrics.get(c, {}).get('RMSE', 0) for c in GROUP_MAIN]
        bar_colors = [colors[c] for c in GROUP_MAIN]

        bars1 = ax2.bar(x_pos - w_bar / 2, maes, w_bar,
                        color=bar_colors, alpha=0.85, label='MAE',
                        edgecolor='white', linewidth=0.5)
        bars2 = ax2.bar(x_pos + w_bar / 2, rmses, w_bar,
                        color=bar_colors, alpha=0.45, label='RMSE',
                        edgecolor='white', linewidth=0.5, hatch='//')

        short_labels = ['MCM\n(Proposed)', 'OP\nExhaust', 'OB\nExhaust',
                        'ESD\nExhaust', 'EOB\nExhaust', 'EDT\nSkelMax']
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(short_labels, fontsize=6.5)
        ax2.set_ylabel('Error (mm)', fontsize=8)
        ax2.set_title('Max Width Finding: MCM Single-Point vs Exhaustive Search',
                       fontsize=8.5, fontweight='bold')
        ax2.legend(fontsize=7)
        ax2.tick_params(axis='both', direction='in')

        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.0005,
                         f'{h:.4f}', ha='center', va='bottom', fontsize=5)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.0005,
                         f'{h:.4f}', ha='center', va='bottom', fontsize=5)

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig2.savefig(os.path.join(args.output_dir,
                         f'fig_candidate_bar.{ext}'),
                         dpi=600, facecolor='white')
        plt.close(fig2)
        print("Bar chart saved")

        fig3, ax3 = plt.subplots(figsize=(120 / 25.4, 65 / 25.4))
        mnames = ['OP', 'OB', 'ESD', 'EOB']
        x3 = np.arange(len(mnames))
        w3 = 0.35
        mae_all = [all_metrics.get(f'{m}_Exhaust', {}).get('MAE', 0)
                    for m in mnames]
        mae_deg2 = [all_metrics.get(f'{m}_Deg2', {}).get('MAE', 0)
                     for m in mnames]

        ax3.bar(x3 - w3 / 2, mae_all, w3, color='#d9534f', alpha=0.7,
                label='All Skeleton', edgecolor='white', linewidth=0.5)
        ax3.bar(x3 + w3 / 2, mae_deg2, w3, color='#5cb85c', alpha=0.7,
                label='Deg≤2 Skeleton', edgecolor='white', linewidth=0.5)

        ax3.set_xticks(x3)
        ax3.set_xticklabels(mnames, fontsize=8)
        ax3.set_ylabel('MAE (mm)', fontsize=8)
        ax3.set_title('Effect of Degree≤2 Filtering on Exhaustive Methods',
                       fontsize=8.5, fontweight='bold')
        ax3.legend(fontsize=7)
        ax3.tick_params(axis='both', direction='in')
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig3.savefig(os.path.join(args.output_dir,
                         f'fig_deg2_effect.{ext}'),
                         dpi=600, facecolor='white')
        plt.close(fig3)
        print("Degree-filter comparison saved")

    except Exception as e:
        print(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nResults directory: {args.output_dir}")

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()

"""Evaluate six width estimators on synthetic masks in reference-site or automatic-site mode."""

import os
import sys
import csv
import argparse
import time
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

from baseline_width_methods import (
    get_boundary_points,
    measure_op,
    measure_orthoboundary,
    measure_esd,
    measure_eob,
)

MM_PER_PIXEL = 0.01278                   
NEI8 = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)]

def skeleton_degree(skel):
    ys, xs = np.where(skel)
    if ys.size == 0:
        return ys, xs, np.array([], dtype=np.int32)
    h, w = skel.shape
    skel_u8 = skel.astype(np.uint8)
    deg = np.zeros_like(ys, dtype=np.int32)
    for idx in range(len(ys)):
        y, x = int(ys[idx]), int(xs[idx])
        count = 0
        for dy, dx in NEI8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                count += int(skel_u8[yy, xx] > 0)
        deg[idx] = count
    return ys, xs, deg

def pick_global_max_center(mask_bool, dist):
    skel = skeletonize(mask_bool)
    if skel.sum() < 10:
        return None, skel
    ys, xs, deg = skeleton_degree(skel)
    if ys.size == 0:
        return None, skel
    dvals = dist[ys, xs]
    valid = deg <= 2
    if np.any(valid):
        ys2, xs2, dvals2 = ys[valid], xs[valid], dvals[valid]
        idx = int(np.argmax(dvals2))
        return (float(xs2[idx]), float(ys2[idx]), float(dvals2[idx])), skel
    idx = int(np.argmax(dvals))
    return (float(xs[idx]), float(ys[idx]), float(dvals[idx])), skel

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

def seg_len(p1, p2):
    return float(((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5)

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
        L = seg_len(p1, p2)
        if L < 1.0:
            return
        if L < 0.90 * expected_min_px:
            return
        if not boundary_next_is_outside(mask_bool, p1, (dx, dy), step=1.0):
            return
        if not boundary_next_is_outside(mask_bool, p2, (-dx, -dy), step=1.0):
            return
        if (best is None) or (L < best[0]):
            best = (L, p1, p2, float(np.degrees(theta)))

    for deg in range(0, 180, coarse_step_deg):
        eval_theta(np.radians(deg))
    if best is None:
        return None
    best_theta = best[3]
    for deg in range(int(round(best_theta - fine_halfspan_deg)),
                     int(round(best_theta + fine_halfspan_deg)) + 1,
                     fine_step_deg):
        eval_theta(np.radians(deg % 180))
    if best is None:
        return None
    return best[0]

def measure_edt_direct(dist, center_xy):
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    d = dist[cy, cx]
    return 2.0 * d

METHOD_KEYS = ["MCM", "OP", "OB", "ESD", "EOB", "EDT"]
METHOD_LABELS = {
    "MCM": "MCM (Proposed)",
    "OP":  "OP (Qiu 2017)",
    "OB":  "OB (Li 2024)",
    "ESD": "ESD (Ong 2022)",
    "EOB": "EOB (Li 2025)",
    "EDT": "EDT Direct",
}
METHOD_COLORS = {
    "MCM": '#2196F3',
    "OP":  '#9C27B0',
    "OB":  '#4CAF50',
    "ESD": '#FF9800',
    "EOB": '#E91E63',
    "EDT": '#607D8B',
}

def load_groundtruth(csv_path):
    """Read the synthetic design-reference CSV."""
    records = []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'image_id': row['image_id'],
                'level': row['level'],
                'target_width_px': int(row['target_width_px']),
                'gt_max_width_px': float(row['gt_max_width_px']),
                'gt_max_width_mm': float(row['gt_max_width_mm']),
                'gt_center_x': float(row['gt_center_x']),
                'gt_center_y': float(row['gt_center_y']),
            })
    return records

def calc_metrics(gt_arr, pred_arr):
    """Compute valid-sample count, MAE, RMSE, R-squared, maximum error, and signed bias."""
    valid = ~(np.isnan(gt_arr) | np.isnan(pred_arr))
    gt_v, pred_v = gt_arr[valid], pred_arr[valid]
    n = len(gt_v)
    if n == 0:
        return {"n": 0}
    err = pred_v - gt_v
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((gt_v - np.mean(gt_v)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    max_err = np.max(np.abs(err))
    bias = np.mean(err)
    return {"n": n, "MAE": mae, "RMSE": rmse, "R2": r2,
            "MaxErr": max_err, "Bias": bias}

def main():
    parser = argparse.ArgumentParser(
        description="Six-method synthetic crack-width evaluation"
    )
    parser.add_argument('--syn-dir', '--syn_dir', type=str, default="./data/synthetic",
                        help="Synthetic dataset directory containing images/ and groundtruth.csv")
    parser.add_argument('--output-dir', '--output_dir', type=str, default=None,
                        help="Output directory (defaults to the dataset directory)")
    parser.add_argument('--center-mode', '--center_mode', type=str, default="gt",
                        choices=["gt", "auto"],
                        help="Site mode: gt uses the design site with skeleton snapping; auto selects a candidate using EDT")
    parser.add_argument('--kernel-radius', '--kernel_radius', type=int, default=2,
                        help="PCA neighborhood radius shared by OP, OB, ESD, and EOB (default: 2)")
    parser.add_argument('--mm-per-px', '--mm_per_px', type=float, default=MM_PER_PIXEL)
    parser.add_argument("--no-plots", action="store_true", help="Skip optional plot generation")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.syn_dir
    os.makedirs(args.output_dir, exist_ok=True)
    k = args.mm_per_px

    gt_csv = os.path.join(args.syn_dir, 'groundtruth.csv')
    gt_records = load_groundtruth(gt_csv)
    print(f"Loaded design references: {len(gt_records)} records")

    img_dir = os.path.join(args.syn_dir, 'images')
    rows = []
    fail_counts = {m: 0 for m in METHOD_KEYS}

    t_start = time.time()

    for idx, rec in enumerate(gt_records):
        img_id = rec['image_id']
        level = rec['level']
        gt_mm = rec['gt_max_width_mm']
        gt_cx = rec['gt_center_x']
        gt_cy = rec['gt_center_y']

        print(f"[{idx + 1}/{len(gt_records)}] {img_id} (L={level}) ... ", end="", flush=True)

        img_path = os.path.join(img_dir, img_id)
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print("Image read failed")
            continue
        mask = raw > 127       

        if mask.sum() < 10:
            print("Mask too small; skipped")
            continue

        dist = distance_transform_edt(mask)

        skel_full = skeletonize(mask)

        if args.center_mode == "gt":
            center_xy = (gt_cx, gt_cy)
            cy_int, cx_int = int(round(gt_cy)), int(round(gt_cx))

            h, w = mask.shape
            if not (0 <= cy_int < h and 0 <= cx_int < w and mask[cy_int, cx_int]):
                print(f"Reference site ({gt_cx:.1f},{gt_cy:.1f}) lies outside the mask; skipped")
                continue

            r_px = float(dist[cy_int, cx_int])

            skel_all_ys, skel_all_xs = np.where(skel_full)
            if len(skel_all_ys) > 0:
                d2skel = (skel_all_xs - gt_cx) ** 2 + (skel_all_ys - gt_cy) ** 2
                nearest_idx = int(np.argmin(d2skel))
                snap_x = float(skel_all_xs[nearest_idx])
                snap_y = float(skel_all_ys[nearest_idx])
                snap_dist = np.sqrt(d2skel[nearest_idx])
                                         
                if snap_dist <= r_px + 2:
                    center_xy = (snap_x, snap_y)
                    cy_int, cx_int = int(round(snap_y)), int(round(snap_x))
                    r_px = float(dist[cy_int, cx_int])
        else:
            pick, skel_full = pick_global_max_center(mask, dist)
            if pick is None:
                print("Candidate selection failed; skipped")
                continue
            center_xy = (pick[0], pick[1])
            r_px = float(pick[2])

        expected_min_px = max(2.0 * r_px, 1.0)

        search_radius = max(r_px * 2, 8.0)

        skel = skel_full.copy()
        skel_ys, skel_xs = np.where(skel)
        if len(skel_ys) > 0:
            skel_dists = np.sqrt(
                (skel_xs - center_xy[0]) ** 2 + (skel_ys - center_xy[1]) ** 2)
            far_mask = skel_dists > search_radius
            skel[skel_ys[far_mask], skel_xs[far_mask]] = False

        boundary_pts_all = get_boundary_points(mask)
        if len(boundary_pts_all) > 0:
            dists_to_center = np.sqrt(
                (boundary_pts_all[:, 0] - center_xy[0]) ** 2 +
                (boundary_pts_all[:, 1] - center_xy[1]) ** 2)
            local_mask = dists_to_center <= search_radius
            boundary_pts = boundary_pts_all[local_mask]
            if len(boundary_pts) < 4:
                boundary_pts = boundary_pts_all         
        else:
            boundary_pts = boundary_pts_all

        results_px = {}

        results_px["MCM"] = measure_mcm(mask, center_xy, expected_min_px)

        results_px["OP"] = measure_op(
            mask, skel, boundary_pts, center_xy,
            kernel_radius=args.kernel_radius)

        results_px["OB"] = measure_orthoboundary(
            mask, skel, boundary_pts, center_xy,
            kernel_radius=args.kernel_radius)

        results_px["ESD"] = measure_esd(
            mask, skel, boundary_pts, center_xy,
            kernel_radius=args.kernel_radius)

        results_px["EOB"] = measure_eob(
            mask, skel, boundary_pts, center_xy,
            kernel_radius=args.kernel_radius)

        results_px["EDT"] = measure_edt_direct(dist, center_xy)

        results_mm = {}
        for m in METHOD_KEYS:
            v = results_px[m]
            if v is not None:
                results_mm[m] = v * k
            else:
                results_mm[m] = None
                fail_counts[m] += 1

        parts = [f"{m}={results_mm[m]:.4f}" if results_mm[m] is not None
                 else f"{m}=FAIL" for m in METHOD_KEYS]
        print(f"GT={gt_mm:.4f}  " + "  ".join(parts))

        row = {
            "image_id": img_id,
            "level": level,
            "target_width_px": rec['target_width_px'],
            "gt_mm": gt_mm,
            "reference_center_x": gt_cx,
            "reference_center_y": gt_cy,
            "center_x": float(center_xy[0]),
            "center_y": float(center_xy[1]),
            "center_mode": args.center_mode,
        }
        for m in METHOD_KEYS:
            row[f"{m.lower()}_mm"] = results_mm[m]
        rows.append(row)

    elapsed = time.time() - t_start
    print(f"\nCompleted: {len(rows)} valid samples; elapsed {elapsed:.1f}s")

    if not rows:
        print("No valid results")
        return

    csv_path = os.path.join(args.output_dir, "syn_comparison_6methods.csv")
    fieldnames = ["image_id", "level", "target_width_px", "gt_mm", "reference_center_x", "reference_center_y", "center_x", "center_y", "center_mode"] + \
                 [f"{m.lower()}_mm" for m in METHOD_KEYS]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}")

    gt_all = np.array([r["gt_mm"] for r in rows])

    summary = "=" * 90 + "\n"
    summary += "Synthetic Crack Width Measurement — 6-Method Comparison\n"
    summary += "=" * 90 + "\n\n"

    summary += "[Failure Count]\n"
    for m in METHOD_KEYS:
        summary += f"  {METHOD_LABELS[m]:<22} : {fail_counts[m]}\n"
    summary += "\n"

    summary += "[Overall]\n"
    summary += f"{'Method':<22} {'n':>4} {'MAE(mm)':>10} {'RMSE(mm)':>10} " \
               f"{'R²':>8} {'Bias(mm)':>10} {'MaxErr':>10}\n"
    summary += "-" * 90 + "\n"

    all_metrics = {}
    for m in METHOD_KEYS:
        col = f"{m.lower()}_mm"
        vals = np.array([r[col] if r[col] is not None else np.nan for r in rows])
        met = calc_metrics(gt_all, vals)
        all_metrics[m] = met
        if met["n"] > 0:
            summary += f"{METHOD_LABELS[m]:<22} {met['n']:>4} " \
                        f"{met['MAE']:>10.4f} {met['RMSE']:>10.4f} " \
                        f"{met['R2']:>8.4f} {met['Bias']:>10.4f} " \
                        f"{met['MaxErr']:>10.4f}\n"
        else:
            summary += f"{METHOD_LABELS[m]:<22}    0  (no valid samples)\n"
    summary += "\n"

    levels = sorted(set(r["level"] for r in rows))
    for lv in levels:
        lv_rows = [r for r in rows if r["level"] == lv]
        gt_lv = np.array([r["gt_mm"] for r in lv_rows])
        n_lv = len(lv_rows)
        summary += f"[{lv}] n={n_lv}\n"
        summary += f"{'Method':<22} {'MAE(mm)':>10} {'RMSE(mm)':>10} " \
                   f"{'R²':>8} {'Bias(mm)':>10}\n"
        summary += "-" * 70 + "\n"
        for m in METHOD_KEYS:
            col = f"{m.lower()}_mm"
            vals = np.array([r[col] if r[col] is not None else np.nan
                             for r in lv_rows])
            met = calc_metrics(gt_lv, vals)
            if met["n"] > 0:
                summary += f"{METHOD_LABELS[m]:<22} " \
                            f"{met['MAE']:>10.4f} {met['RMSE']:>10.4f} " \
                            f"{met['R2']:>8.4f} {met['Bias']:>10.4f}\n"
            else:
                summary += f"{METHOD_LABELS[m]:<22}  (no valid)\n"
        summary += "\n"

    print(summary)
    summary_path = os.path.join(args.output_dir, "syn_6methods_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved: {summary_path}")
    import json
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, default=float)

    if args.no_plots:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
        axes = axes.flatten()

        for i, m in enumerate(METHOD_KEYS):
            ax = axes[i]
            col = f"{m.lower()}_mm"
            vals = np.array([r[col] if r[col] is not None else np.nan
                             for r in rows])
            valid = ~np.isnan(vals)
            gt_v = gt_all[valid]
            pred_v = vals[valid]

            if len(gt_v) == 0:
                ax.set_title(f'{METHOD_LABELS[m]} (no data)', fontsize=10)
                continue

            levels_v = [rows[j]["level"] for j in range(len(rows)) if valid[j]]
            level_colors = {'L1': '#4CAF50', 'L2': '#2196F3',
                            'L3': '#FF9800', 'L4': '#E91E63'}
            colors = [level_colors.get(lv, '#999') for lv in levels_v]

            ax.scatter(gt_v, pred_v, c=colors, s=25, alpha=0.7,
                       edgecolors='white', linewidth=0.3)

            lims = [min(gt_v.min(), pred_v.min()) - 0.01,
                    max(gt_v.max(), pred_v.max()) + 0.01]
            ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.set_xlabel('Design reference (mm)', fontsize=9)
            ax.set_ylabel(f'{METHOD_LABELS[m]} (mm)', fontsize=9)
            ax.set_title(f'({chr(97 + i)}) {METHOD_LABELS[m]}', fontsize=10,
                         fontweight='bold')
            ax.grid(True, alpha=0.3)

            met = all_metrics[m]
            if met["n"] > 0:
                textstr = (f'R² = {met["R2"]:.4f}\n'
                           f'MAE = {met["MAE"]:.4f} mm\n'
                           f'Bias = {met["Bias"]:+.4f} mm\n'
                           f'n = {met["n"]}')
                props = dict(boxstyle='round,pad=0.4', facecolor='wheat',
                             alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                        fontsize=8, va='top', bbox=props)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label='L1: nominal widths 6-12 px'),
            Patch(facecolor='#2196F3', label='L2: nominal widths 14-18 px'),
            Patch(facecolor='#FF9800', label='L3: nominal widths 22-30 px'),
            Patch(facecolor='#E91E63', label='L4: nominal widths 34-44 px'),
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=4, fontsize=9, frameon=True,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        fig_path = os.path.join(args.output_dir, "syn_6methods_scatter.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot saved: {fig_path}")

    except Exception as e:
        print(f"Plot generation failed: {e}")

    ranked = sorted(
        [(m, all_metrics[m]) for m in METHOD_KEYS if all_metrics[m]["n"] > 0],
        key=lambda x: x[1]["MAE"]
    )
    print("\n" + "=" * 60)
    print("Methods ranked by increasing MAE")
    print("=" * 60)
    for rank, (m, met) in enumerate(ranked, 1):
        print(f"  {rank}. {METHOD_LABELS[m]:<22} MAE={met['MAE']:.4f}mm  "
              f"R²={met['R2']:.4f}")
    print()

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()

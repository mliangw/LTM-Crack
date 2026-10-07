"""Boundary-based comparison implementations used in the LTM-Crack experiments."""

import numpy as np
import cv2

def get_boundary_points(mask_bool):
    """Extract boundary coordinates in x, y order from a binary mask."""
    m = (mask_bool.astype(np.uint8) * 255)
    edges = cv2.Canny(m, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(ys) == 0:
        kernel = np.ones((3, 3), np.uint8)
        grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
        ys, xs = np.where(grad > 0)
    return np.column_stack([xs, ys]).astype(float)

def pca_direction(points, center_xy, kernel_radius=2):
    """Estimate a local tangent direction from skeleton coordinates using PCA."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    mask = (np.abs(dx) <= kernel_radius) & (np.abs(dy) <= kernel_radius)
    local = points[mask]
    if len(local) < 3:
        mask2 = (np.abs(dx) <= kernel_radius * 3) & (np.abs(dy) <= kernel_radius * 3)
        local = points[mask2]
        if len(local) < 3:
            return None
    centered = local - local.mean(axis=0)
    try:
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    tangent = Vt[0]
    theta = np.arctan2(tangent[1], tangent[0]) % np.pi
    return theta

def get_skeleton_points_array(skel):
    """Return skeleton coordinates in x, y order."""
    ys, xs = np.where(skel)
    return np.column_stack([xs, ys]).astype(float)

def _align_angles(base, *others):
    """Align angular estimates around a reference direction."""
    aligned = [base]
    for a in others:
        diff = a - base
        if diff > np.pi / 2:
            a -= np.pi
        elif diff < -np.pi / 2:
            a += np.pi
        aligned.append(a)
    return aligned

def measure_op(mask_bool, skel, boundary_pts, center_xy, kernel_radius=2):
    """Evaluate the experimental orthogonal-projection baseline."""
    skel_pts = get_skeleton_points_array(skel)
    if len(skel_pts) < 3 or len(boundary_pts) < 2:
        return None

    theta = pca_direction(skel_pts, center_xy, kernel_radius=kernel_radius)
    if theta is None:
        return None

    ortho_theta = theta + np.pi / 2
    O = np.array([np.cos(ortho_theta), np.sin(ortho_theta)])

    cx, cy = float(center_xy[0]), float(center_xy[1])
    center = np.array([cx, cy])
    vecs = boundary_pts - center
    projections = vecs @ O

    width = float(projections.max() - projections.min())
    if width < 1.0:
        return None
    return width

def measure_orthoboundary(mask_bool, skel, boundary_pts, center_xy,
                          kernel_radius=2):
    """Evaluate the experimental OrthoBoundary baseline."""
    skel_pts = get_skeleton_points_array(skel)
    if len(skel_pts) < 3 or len(boundary_pts) < 2:
        return None

    cx, cy = float(center_xy[0]), float(center_xy[1])
    center = np.array([cx, cy])

    theta_skel = pca_direction(skel_pts, center_xy, kernel_radius=kernel_radius)
    if theta_skel is None:
        return None

    ortho = theta_skel + np.pi / 2
    O = np.array([np.cos(ortho), np.sin(ortho)])
    vecs = boundary_pts - center
    projections = vecs @ O
    idx_max = int(np.argmax(projections))
    idx_min = int(np.argmin(projections))
    p1 = boundary_pts[idx_max]
    p2 = boundary_pts[idx_min]

    theta1 = pca_direction(boundary_pts, tuple(p1), kernel_radius=kernel_radius)
    theta2 = pca_direction(boundary_pts, tuple(p2), kernel_radius=kernel_radius)
    if theta1 is None:
        theta1 = theta_skel
    if theta2 is None:
        theta2 = theta_skel

    aligned = _align_angles(theta_skel, theta1, theta2)
    theta_corrected = np.mean(aligned) % np.pi

    ortho_c = theta_corrected + np.pi / 2
    O_new = np.array([np.cos(ortho_c), np.sin(ortho_c)])
    projections_new = vecs @ O_new

    width = float(projections_new.max() - projections_new.min())
    if width < 1.0:
        return None
    return width

def _split_lr_and_min_dist(boundary_pts, center, ortho_vec):
    """ split lr and min dist."""
    vecs = boundary_pts - center
    side = vecs @ ortho_vec                   

    D1 = boundary_pts[side > 0.5]                    
    D2 = boundary_pts[side < -0.5]

    if len(D1) == 0 or len(D2) == 0:
              
        D1 = boundary_pts[side > 0]
        D2 = boundary_pts[side < 0]
        if len(D1) == 0 or len(D2) == 0:
            return None, None, None

    diff = D1[:, np.newaxis, :] - D2[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = D1[min_idx[0]]
    p2 = D2[min_idx[1]]
    width = float(dists[min_idx])
    return width, p1, p2

def measure_esd(mask_bool, skel, boundary_pts, center_xy,
                kernel_radius=2, **kwargs):
    """Evaluate the experimental shortest-distance baseline."""
    skel_pts = get_skeleton_points_array(skel)
    if len(skel_pts) < 3 or len(boundary_pts) < 2:
        return None

    cx, cy = float(center_xy[0]), float(center_xy[1])
    center = np.array([cx, cy])

    theta = pca_direction(skel_pts, center_xy, kernel_radius=kernel_radius)
    if theta is None:
        return None

    ortho_theta = theta + np.pi / 2
    ortho_vec = np.array([np.cos(ortho_theta), np.sin(ortho_theta)])

    width, _, _ = _split_lr_and_min_dist(boundary_pts, center, ortho_vec)
    if width is None or width < 1.0:
        return None
    return width

def measure_eob(mask_bool, skel, boundary_pts, center_xy,
                kernel_radius=2, **kwargs):
    """Evaluate the experimental edge-OrthoBoundary baseline."""
    skel_pts = get_skeleton_points_array(skel)
    if len(skel_pts) < 3 or len(boundary_pts) < 2:
        return None

    cx, cy = float(center_xy[0]), float(center_xy[1])
    center = np.array([cx, cy])

    theta = pca_direction(skel_pts, center_xy, kernel_radius=kernel_radius)
    if theta is None:
        return None

    ortho = theta + np.pi / 2
    ortho_vec = np.array([np.cos(ortho), np.sin(ortho)])
    w0, p1, p2 = _split_lr_and_min_dist(boundary_pts, center, ortho_vec)
    if w0 is None:
        return None

    theta1 = pca_direction(boundary_pts, tuple(p1), kernel_radius=kernel_radius)
    theta2 = pca_direction(boundary_pts, tuple(p2), kernel_radius=kernel_radius)
    if theta1 is None:
        theta1 = theta
    if theta2 is None:
        theta2 = theta

    aligned = _align_angles(theta, theta1, theta2)
    theta_corrected = np.mean(aligned) % np.pi

    ortho_c = theta_corrected + np.pi / 2
    ortho_vec_c = np.array([np.cos(ortho_c), np.sin(ortho_c)])
    width, _, _ = _split_lr_and_min_dist(boundary_pts, center, ortho_vec_c)

    if width is None:
        return w0           
    if width < 1.0:
        return None
    return width

def measure_all_baselines(mask_bool, skel, center_xy, kernel_radius=2):
    """Measure all baselines."""
    boundary_pts = get_boundary_points(mask_bool)
    if len(boundary_pts) < 4:
        return {"OP": None, "OB": None, "ESD": None, "EOB": None}
    return {
        "OP": measure_op(mask_bool, skel, boundary_pts, center_xy,
                         kernel_radius=kernel_radius),
        "OB": measure_orthoboundary(mask_bool, skel, boundary_pts, center_xy,
                                     kernel_radius=kernel_radius),
        "ESD": measure_esd(mask_bool, skel, boundary_pts, center_xy,
                           kernel_radius=kernel_radius),
        "EOB": measure_eob(mask_bool, skel, boundary_pts, center_xy,
                           kernel_radius=kernel_radius),
    }

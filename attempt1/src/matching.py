"""Centroid matching of measured particles to synthetic ground truth."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def match(meas_xy, gt_xy, gt_d_px, tol_frac=0.5, tol_min=3.0):
    """One-to-one match: a measured centroid matches a GT centre if within
    max(tol_min, tol_frac * GT diameter). Returns (meas_idx, gt_idx) arrays."""
    if len(meas_xy) == 0 or len(gt_xy) == 0:
        return np.array([], int), np.array([], int)
    tol = np.maximum(tol_min, tol_frac * np.asarray(gt_d_px))
    tree = cKDTree(meas_xy)
    big = 1e6
    cost = np.full((len(gt_xy), len(meas_xy)), big)
    for i, (p, t) in enumerate(zip(gt_xy, tol)):
        for j in tree.query_ball_point(p, t):
            cost[i, j] = np.hypot(*(meas_xy[j] - p))
    gi, mj = linear_sum_assignment(cost)
    ok = cost[gi, mj] < big
    return mj[ok], gi[ok]

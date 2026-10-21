"""Particle characteristics (static image analysis, ISO 13322-1 practice; shape
descriptors as defined in ISO 9276-6) and per-image / per-sample summaries.

Per particle (px; *_um when calibrated):
  area, ecd (equivalent circle diameter), perimeter (Crofton estimate, less
  pixel-staircase bias than a boundary trace), feret_max, feret_min (exact
  calipers over the convex hull),
  circularity  = 4*pi*A / P^2               (1 = circle)
  feret_ar     = feret_min / feret_max      (1 = equiaxed)
  ellipse_ar   = major / minor axis of the moment ellipse (>= 1)
  roundness    = 4*A / (pi * feret_max^2)
  convexity    = convex hull perimeter / perimeter (1 = no concavities)
  solidity     = A / convex hull area
  n_touching   = particles touching it (agglomeration indicator)
  n_attached_small = touching particles with ECD <= 0.4x its own (satellite-like)
  edge_weight  = Miles-Lantuejoul weight correcting the loss of edge-touching particles
Sample summary: count; D10/D50/D90 by number, volume (w ~ ECD^3) and volume
edge-corrected; D[4,3], D[3,2]; span; median and P10/P90 of each shape
descriptor; % spherical, % touching, % with satellite-like neighbours; flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

PCTS = (10, 50, 90)
SPHERICAL = {"circularity": 0.90, "feret_ar": 0.80}   # both must hold
SMALL_NEIGHBOUR = 0.4
MIN_COUNT_D90 = 300   # fewer particles per sample: D90 flagged as unreliable


def _feret_min_max(coords):
    """Exact min/max caliper width of a pixel set (via convex hull of pixel corners)."""
    import cv2
    pts = np.concatenate([coords + o for o in ([0, 0], [0, 1], [1, 0], [1, 1])]).astype(np.float32)
    hull = cv2.convexHull(pts[:, ::-1]).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return 1.0, 1.0, 4.0
    d = hull[:, None, :] - hull[None, :, :]
    fmax = float(np.sqrt((d ** 2).sum(-1)).max())
    e = np.roll(hull, -1, axis=0) - hull
    n = np.stack([-e[:, 1], e[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    widths = (hull @ n.T)
    fmin = float((widths.max(0) - widths.min(0)).min())
    hull_perim = float(np.linalg.norm(e, axis=1).sum())
    return fmin, fmax, hull_perim


def _touching(labels, reach=2):
    """Set of touching label pairs (within `reach` px)."""
    pairs = set()
    H, W = labels.shape
    for dy in range(0, reach + 1):
        for dx in range(-reach, reach + 1):
            if dy == 0 and dx <= 0:
                continue
            a = labels[0:H - dy, max(0, -dx):W - max(0, dx)]
            b = labels[dy:H, max(0, dx):W - max(0, -dx) if dx < 0 else W][:, :a.shape[1]]
            m = (a > 0) & (b > 0) & (a != b)
            if m.any():
                for p, q in np.unique(np.stack([a[m], b[m]], 1), axis=0):
                    pairs.add((min(p, q), max(p, q)))
    return pairs


def particle_table(labels: np.ndarray, um_per_px: float | None = None) -> pd.DataFrame:
    H, W = labels.shape
    rows = []
    for rp in measure.regionprops(labels):
        a = float(rp.area)
        per = float(rp.perimeter_crofton) or float(rp.perimeter) or np.nan
        fmin, fmax, hull_per = _feret_min_max(rp.coords)
        y0, x0, y1, x1 = rp.bbox
        mj, mn = float(rp.axis_major_length), float(rp.axis_minor_length)
        rows.append({
            "id": rp.label, "area_px": a, "ecd_px": 2 * np.sqrt(a / np.pi), "perimeter_px": per,
            "feret_max_px": fmax, "feret_min_px": fmin,
            "circularity": min(1.0, 4 * np.pi * a / per ** 2) if per == per else np.nan,
            "feret_ar": fmin / fmax if fmax > 0 else np.nan,
            "ellipse_ar": mj / mn if mn > 0 else np.nan,
            "roundness": min(1.0, 4 * a / (np.pi * fmax ** 2)) if fmax > 0 else np.nan,
            "convexity": min(1.0, hull_per / per) if per == per and per > 0 else np.nan,
            "solidity": float(rp.solidity),
            "centroid_x": float(rp.centroid[1]), "centroid_y": float(rp.centroid[0]),
            "edge_weight": (W * H) / max(1.0, (W - (x1 - x0)) * (H - (y1 - y0))),
        })
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame(columns=["id", "area_px", "ecd_px", "perimeter_px", "feret_max_px", "feret_min_px",
                                     "circularity", "feret_ar", "ellipse_ar", "roundness", "convexity", "solidity",
                                     "centroid_x", "centroid_y", "edge_weight", "n_touching", "n_attached_small"])
    ecd = dict(zip(df.id, df.ecd_px))
    nt = {i: 0 for i in df.id}
    ns = {i: 0 for i in df.id}
    for p, q in _touching(labels):
        if p in nt and q in nt:
            nt[p] += 1
            nt[q] += 1
            if ecd[q] <= SMALL_NEIGHBOUR * ecd[p]:
                ns[p] += 1
            if ecd[p] <= SMALL_NEIGHBOUR * ecd[q]:
                ns[q] += 1
    df["n_touching"] = df.id.map(nt)
    df["n_attached_small"] = df.id.map(ns)
    if um_per_px:
        s = float(um_per_px)
        df["area_um2"] = df.area_px * s * s
        for c in ("ecd", "perimeter", "feret_max", "feret_min"):
            df[f"{c}_um"] = df[f"{c}_px"] * s
    return df


def wpct(x, w, pcts=PCTS):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if not len(x):
        return [np.nan] * len(pcts)
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(p / 100.0, c, x)) for p in pcts]


SHAPES = ["circularity", "feret_ar", "ellipse_ar", "roundness", "convexity", "solidity"]


def summarize(df: pd.DataFrame, calibrated: bool) -> dict:
    """Summary of one image or a pooled sample (df may span several images)."""
    col = "ecd_um" if calibrated else "ecd_px"
    unit = "um" if calibrated else "px"
    d = df[col].to_numpy(float) if len(df) else np.array([])
    ew = df["edge_weight"].to_numpy(float) if len(df) else np.array([])
    s = {"count": int(len(df)), "size_unit": unit}
    for name, w in (("num", np.ones_like(d)), ("vol", d ** 3), ("vol_ec", d ** 3 * ew)):
        for p, v in zip(PCTS, wpct(d, w)):
            s[f"D{p}_{name}"] = v
    s["D43"] = float((d ** 4 * ew).sum() / (d ** 3 * ew).sum()) if len(d) else np.nan
    s["D32"] = float((d ** 3 * ew).sum() / (d ** 2 * ew).sum()) if len(d) else np.nan
    s["span_vol_ec"] = (s["D90_vol_ec"] - s["D10_vol_ec"]) / s["D50_vol_ec"] if len(d) else np.nan
    for k in SHAPES:
        v = df[k].dropna() if len(df) else pd.Series(dtype=float)
        s[f"{k}_median"] = float(v.median()) if len(v) else np.nan
        s[f"{k}_p10"] = float(v.quantile(0.1)) if len(v) else np.nan
        s[f"{k}_p90"] = float(v.quantile(0.9)) if len(v) else np.nan
    if len(df):
        sph = (df.circularity >= SPHERICAL["circularity"]) & (df.feret_ar >= SPHERICAL["feret_ar"])
        s["pct_spherical"] = float(100 * sph.mean())
        s["pct_spherical_vol"] = float(100 * (d ** 3 * ew)[sph.to_numpy()].sum() / (d ** 3 * ew).sum())
        s["pct_touching"] = float(100 * (df.n_touching > 0).mean())
        s["pct_with_satellite_like"] = float(100 * (df.n_attached_small > 0).mean())
    else:
        s.update(pct_spherical=np.nan, pct_spherical_vol=np.nan, pct_touching=np.nan, pct_with_satellite_like=np.nan)
    flags = []
    if not calibrated:
        flags.append("UNCALIBRATED (sizes in px)")
    if len(df) < MIN_COUNT_D90:
        flags.append(f"only {len(df)} particles: D90 unreliable (aim for >= {MIN_COUNT_D90}; add fields)")
    s["flags"] = "; ".join(flags)
    return s

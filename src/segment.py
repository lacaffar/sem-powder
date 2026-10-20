"""Particle segmentation: watershed baseline and Cellpose cyto3 (zero-shot).

No alloy-specific thresholds. Every parameter is either global (Otsu) or scaled
by the particle radius estimated from the image itself.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

MIN_AREA_PX = 20
_CP_MODEL = None


# ------------------------------------------------------------ preprocessing
def preprocess(img_u8: np.ndarray) -> np.ndarray:
    """Grayscale uint8 -> CLAHE -> float32 in [0, 1]."""
    tile = max(2, int(round(max(img_u8.shape) / 128)))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    g = clahe.apply(img_u8).astype(np.float32)
    lo, hi = np.percentile(g, [0.5, 99.5])
    return np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)


def foreground(g: np.ndarray) -> np.ndarray:
    """Foreground = above the lower threshold of a 3-class multi-Otsu.

    SE images of powder have three populations: dark substrate, shaded particle
    flanks, bright particle tops. Plain Otsu splits flanks from tops and cuts
    spheres in half; the lower multi-Otsu threshold separates substrate from
    everything else. Tested on synthetic GT masks: IoU 0.94-0.96 vs 0.85-0.89
    for Otsu (CLAHE input).
    """
    s = filters.gaussian(g, 1.0)
    try:
        t = filters.threshold_multiotsu(s, 3)[0]
    except ValueError:
        t = filters.threshold_otsu(s)
    fg = s > t
    fg = morphology.binary_opening(fg, morphology.disk(1))
    fg = ndi.binary_fill_holes(fg)
    return morphology.remove_small_objects(fg, MIN_AREA_PX)


def estimate_radius(fg: np.ndarray) -> float:
    """Typical particle radius (px): median over components of the max inscribed radius."""
    dist = ndi.distance_transform_edt(fg)
    lab = measure.label(fg)
    if lab.max() == 0:
        return 5.0
    maxd = ndi.maximum(dist, lab, index=np.arange(1, lab.max() + 1))
    return float(np.median(maxd))


# ------------------------------------------------------------ watershed
WS_H = 0.05          # h-maxima height on the combined surface
WS_SMOOTH = 0.25     # brightness smoothing sigma, as a fraction of est. radius


def segment_watershed(img_u8: np.ndarray, g=None, h=WS_H, smooth=WS_SMOOTH):
    """Marker-controlled watershed on (distance / r_est) + smoothed brightness.

    Distance transform alone merges overlapping spheres (wide necks, low
    prominence). SE images of spheres are brightest near the top centre, so adding
    brightness separates them. Brightness is smoothed at 0.25 * r_est so surface
    texture (dendritic/wrinkled skins) does not seed extra markers.
    Tuned on held-out synthetic renders 49-50 and checked visually on AA7075:
    smooth=0 gave +6 pts synthetic recall but shattered textured real spheres.
    """
    g = preprocess(img_u8) if g is None else g
    fg = foreground(g)
    dist = ndi.distance_transform_edt(fg)
    r_est = estimate_radius(fg)
    bright = filters.gaussian(g, max(1.0, smooth * r_est))
    surf = filters.gaussian(dist, 0.5, preserve_range=True) / max(r_est, 1.0) + bright
    markers = measure.label(morphology.h_maxima(surf, h))
    labels = segmentation.watershed(-surf, markers, mask=fg)
    return labels.astype(np.int32), {"r_est_px": r_est}


# ------------------------------------------------------------ cellpose
def _cellpose():
    global _CP_MODEL
    if _CP_MODEL is None:
        import torch
        from cellpose import models
        _CP_MODEL = models.Cellpose(gpu=torch.cuda.is_available(), model_type="cyto3")
    return _CP_MODEL


def segment_cellpose(img_u8: np.ndarray, g=None, diameter=None):
    g = preprocess(img_u8) if g is None else g
    m = _cellpose()
    masks, _flows, _styles, diam = m.eval(g, diameter=diameter, channels=[0, 0])
    return masks.astype(np.int32), {"diameter_px": float(np.atleast_1d(diam)[0])}


# ------------------------------------------------------------ post
def postprocess(labels: np.ndarray, min_area=MIN_AREA_PX) -> np.ndarray:
    labels = segmentation.clear_border(labels)
    labels = morphology.remove_small_objects(labels, min_area)
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def segment(img_u8: np.ndarray, method: str = "cellpose", **kw):
    """method: watershed | cellpose (auto diameter) | cellpose_ws (diameter from
    watershed median ECD). Returns (labels, info)."""
    t0 = time.time()
    g = preprocess(img_u8)
    if method == "watershed":
        lab, info = segment_watershed(img_u8, g)
    elif method == "cellpose":
        lab, info = segment_cellpose(img_u8, g, diameter=kw.get("diameter"))
    elif method == "cellpose_ws":
        wl, winfo = segment_watershed(img_u8, g)
        wl = postprocess(wl)
        areas = np.bincount(wl.ravel())[1:]
        d = float(np.median(2 * np.sqrt(areas / np.pi))) if len(areas) else None
        lab, info = segment_cellpose(img_u8, g, diameter=d)
        info["ws_median_ecd_px"] = d
    else:
        raise ValueError(f"unknown method {method}")
    lab = postprocess(lab)
    info["seconds"] = round(time.time() - t0, 2)
    info["n"] = int(lab.max())
    return lab, info


# ------------------------------------------------------------ overlay
def overlay(img_u8: np.ndarray, labels: np.ndarray, alpha=0.35, title=None) -> np.ndarray:
    """BGR overlay: random colour fill + white-ish boundaries."""
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    n = int(labels.max())
    if n:
        rng = np.random.default_rng(0)
        cols = rng.integers(60, 255, size=(n + 1, 3)).astype(np.float32)
        cols[0] = 0
        m = labels > 0
        rgb[m] = (1 - alpha) * rgb[m] + alpha * cols[labels[m]]
        b = segmentation.find_boundaries(labels, mode="inner")
        rgb[b] = cols[labels[b]] * 0.5 + 127
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    if title:
        s = max(0.5, img_u8.shape[1] / 1400)
        cv2.putText(out, title, (8, int(28 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4)
        cv2.putText(out, title, (8, int(28 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 255, 255), 1)
    return out

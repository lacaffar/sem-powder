"""Classical pieces: preprocessing, foreground, watershed baseline, post-processing,
overlay drawing. Carried over from attempt 1 (attempt1/src/segment.py), where
they were tuned and validated (see attempt1/PROGRESS.md, Block 3)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

MIN_AREA_PX = 20


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


def postprocess(labels: np.ndarray, min_area=MIN_AREA_PX) -> np.ndarray:
    labels = segmentation.clear_border(labels)
    labels = morphology.remove_small_objects(labels, min_area)
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def overlay(img_u8: np.ndarray, labels: np.ndarray, alpha=0.35, title=None, sats=None) -> np.ndarray:
    """BGR overlay: random colour fill + white-ish boundaries; satellites (if
    given, DataFrame with x, y, r_px) as red circles."""
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
    if sats is not None and len(sats):
        for r in sats.itertuples():
            cv2.circle(out, (int(round(r.x)), int(round(r.y))), max(2, int(round(r.r_px))), (0, 0, 255), 1,
                       cv2.LINE_AA)
    if title:
        # yellow text on a black box (a thick black outline drawn as text spills
        # past the yellow glyphs and looks like extra digits: 'n=176' -> 'n=17676')
        s = max(0.5, img_u8.shape[1] / 1400)
        (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
        y = int(28 * s)
        cv2.rectangle(out, (4, y - th - 4), (12 + tw, y + base + 2), (0, 0, 0), -1)
        cv2.putText(out, title, (8, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 255, 255), 1, cv2.LINE_AA)
    return out

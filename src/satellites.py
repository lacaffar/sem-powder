"""Satellite detection: small particles fused to a larger parent particle.

Two detectors:
- `seg_touch` (baseline): a segmented particle is a satellite if it touches a
  particle whose equivalent radius is >= RATIO x its own. Finds only satellites
  the segmentation already split off.
- `blob` (default): multi-scale Laplacian-of-Gaussian blobs on the preprocessed
  image, restricted to particles, kept when the host particle's equivalent radius
  is >= RATIO x the blob radius. Finds satellites that segmentation merged into
  the parent. Union with `seg_touch`, de-duplicated.

Output: DataFrame with x, y (px), r_px, d_um, parent label, source; plus
per-parent counts via `count_per_parent`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, measure, morphology

from .segment import preprocess

RATIO = 2.5          # parent radius / satellite radius, minimum
R_MIN = 2.0          # px; smaller is not resolvable


def _label_radii(labels):
    areas = np.bincount(labels.ravel())
    return np.sqrt(areas / np.pi)  # index = label


def seg_touch(labels, ratio=RATIO):
    req = _label_radii(labels)
    rows = []
    for rp in measure.regionprops(labels):
        r = req[rp.label]
        y0, x0, y1, x1 = rp.bbox
        pad = 2
        sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        m = labels[sl] == rp.label
        ring = morphology.binary_dilation(m, morphology.disk(2)) & ~m
        nb = np.unique(labels[sl][ring])
        nb = nb[(nb > 0) & (nb != rp.label)]
        if not len(nb):
            continue
        big = nb[np.argmax(req[nb])]
        if req[big] >= ratio * r:
            rows.append({"x": rp.centroid[1], "y": rp.centroid[0], "r_px": r, "parent": int(big),
                         "score": 1.0, "source": "seg_touch", "self_label": rp.label})
    return pd.DataFrame(rows, columns=["x", "y", "r_px", "parent", "score", "source", "self_label"])


def blob(img_u8, labels, g=None, ratio=RATIO, threshold=0.06, r_max_frac=0.35, num_sigma=10,
         edge_frac=1.0, min_contrast=0.0):
    """LoG blobs inside particles whose host is >= ratio x larger.

    threshold: LoG response threshold (scale-normalised, image in [0, 1]).
    r_max_frac: largest satellite radius as a fraction of the P90 particle radius.
    edge_frac: blob centre may lie up to edge_frac * r outside the host mask
               (rim satellites that protrude past the silhouette).
    min_contrast: centre brightness minus surrounding-ring brightness (0..1).
    """
    g = preprocess(img_u8) if g is None else g
    req = _label_radii(labels)
    if labels.max() == 0:
        return pd.DataFrame(columns=["x", "y", "r_px", "parent", "score", "source"])
    p90 = np.percentile(req[1:][req[1:] > 0], 90)
    r_max = max(R_MIN + 1, r_max_frac * p90)
    smin, smax = R_MIN / np.sqrt(2), r_max / np.sqrt(2)
    blobs = feature.blob_log(g, min_sigma=smin, max_sigma=smax, num_sigma=num_sigma,
                             threshold=threshold, overlap=0.5, exclude_border=False)
    if not len(blobs):
        return pd.DataFrame(columns=["x", "y", "r_px", "parent", "score", "source"])
    # LoG response at each blob for scoring
    H, W = labels.shape
    # nearest-label lookup for rim satellites outside the mask
    dist, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
    rows = []
    for y, x, s in blobs:
        r = s * np.sqrt(2)
        yi, xi = int(round(y)), int(round(x))
        if not (0 <= yi < H and 0 <= xi < W):
            continue
        lab = labels[yi, xi]
        if lab == 0:
            if dist[yi, xi] > edge_frac * r:
                continue
            lab = labels[iy[yi, xi], ix[yi, xi]]
        if req[lab] < ratio * r:
            continue
        if min_contrast > 0:
            k = 2 * int(r) + 2
            ya, yb, xa, xb = max(0, yi - k), min(H, yi + k + 1), max(0, xi - k), min(W, xi + k + 1)
            yy, xx = np.ogrid[ya:yb, xa:xb]
            d = np.hypot(yy - y, xx - x)
            patch = g[ya:yb, xa:xb]
            inner = patch[d <= 0.6 * r]
            ring = patch[(d > 1.0 * r) & (d <= 1.6 * r)]
            if len(inner) and len(ring) and (inner.mean() - ring.mean()) < min_contrast:
                continue
        rows.append({"x": x, "y": y, "r_px": r, "parent": int(lab), "score": 1.0, "source": "blob"})
    return pd.DataFrame(rows, columns=["x", "y", "r_px", "parent", "score", "source"])


def dedupe(df, frac=0.7):
    """Drop detections whose centres lie within frac * max(r) of an earlier one."""
    if len(df) < 2:
        return df.reset_index(drop=True)
    keep = []
    pts = df[["x", "y"]].to_numpy()
    rs = df["r_px"].to_numpy()
    for i in range(len(df)):
        ok = True
        for j in keep:
            if np.hypot(*(pts[i] - pts[j])) < frac * max(rs[i], rs[j]):
                ok = False
                break
        if ok:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def detect(img_u8, labels, method="model", **kw):
    """method: model (default; candidates + logistic scoring), blob (blob +
    seg_touch union), seg_touch, none."""
    if method == "model":
        return detect_model(img_u8, labels, **kw)
    if method == "none":
        return pd.DataFrame(columns=["x", "y", "r_px", "parent", "score", "source"])
    st = seg_touch(labels)
    if method == "seg_touch":
        return st
    b = blob(img_u8, labels, **kw)
    # segmentation-split satellites first (their centroid/size are better)
    allc = pd.concat([st, b], ignore_index=True) if len(st) else b
    return dedupe(allc)


def count_per_parent(sats, labels):
    """Series: label -> number of satellites attached."""
    ids = np.arange(1, int(labels.max()) + 1)
    c = sats["parent"].value_counts() if len(sats) else pd.Series(dtype=int)
    return pd.Series([int(c.get(i, 0)) for i in ids], index=ids)


# ---------------------------------------------------------------- model-based
# Candidates: LoG blobs on a white top-hat image (removes parent-scale brightness,
# keeps satellite domes). Each candidate gets a feature vector; a logistic
# regression (coefficients in satellite_model.json, fit by src/train_satellites.py
# on the synthetic benchmark tuning split only) scores it.
FEATURES = ["log_g", "log_t", "t_peak", "contrast", "edge_ratio", "edge_frac", "rel_r",
            "sdist", "log_r", "tex_norm_log", "conc"]
MODEL_PATH = Path(__file__).with_name("satellite_model.json")


def candidates(img_u8, labels, g=None, threshold=0.04, ratio=RATIO, r_max_frac=0.35, edge_frac=1.0,
               plain_threshold=0.04, own_ratio=1 / 0.3):
    from scipy.ndimage import gaussian_laplace, sobel
    g = preprocess(img_u8) if g is None else g
    req = _label_radii(labels)
    cols = ["x", "y", "r_px", "parent"] + FEATURES
    if labels.max() == 0:
        return pd.DataFrame(columns=cols)
    p90 = np.percentile(req[1:][req[1:] > 0], 90)
    r_max = max(R_MIN + 1, r_max_frac * p90)
    t = morphology.white_tophat(g, morphology.disk(max(3, int(1.5 * r_max))))
    kw_b = dict(min_sigma=R_MIN / np.sqrt(2), max_sigma=r_max / np.sqrt(2), num_sigma=10,
                overlap=0.5, exclude_border=False)
    # two candidate sources: top-hat (bright domes on a shaded parent) and the
    # plain image (ring-bounded domes in BSE); merge near-duplicates
    bl = feature.blob_log(t, threshold=threshold, **kw_b)
    if plain_threshold:
        b2 = feature.blob_log(g, threshold=plain_threshold, **kw_b)
        if len(b2):
            if len(bl):
                d = np.hypot(b2[:, None, 0] - bl[None, :, 0], b2[:, None, 1] - bl[None, :, 1])
                rr = np.maximum(b2[:, None, 2], bl[None, :, 2]) * np.sqrt(2)
                b2 = b2[~(d < 0.7 * rr).any(1)]
            bl = np.vstack([bl, b2]) if len(bl) else b2
    H, W = labels.shape
    dist_out, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
    gs = ndi.gaussian_filter(g, 0.7)
    grad = np.hypot(sobel(gs, 0), sobel(gs, 1))
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    n_lab = int(labels.max())
    cents = np.array(ndi.center_of_mass(np.ones_like(labels), labels, np.arange(1, n_lab + 1))).reshape(-1, 2)
    rows = []
    for y, x, s in bl:
        r = s * np.sqrt(2)
        yi, xi = int(round(y)), int(round(x))
        if not (0 <= yi < H and 0 <= xi < W):
            continue
        lab = labels[yi, xi]
        if lab == 0:
            if dist_out[yi, xi] > edge_frac * r:
                continue
            lab = labels[iy[yi, xi], ix[yi, xi]]
        R = req[lab]
        if R < ratio * r:
            # satellite split off by segmentation: host = largest touching label.
            # Judge size by the split-off particle's own area (the blob scale is
            # capped and understates whole small particles), at most 0.3 x host.
            big = _big_neighbour(labels, lab, req)
            if big is None or req[big] < max(ratio * r, own_ratio * req[lab]):
                continue
            lab, R = big, req[big]
        k = int(2.2 * r) + 3
        ya, yb, xa, xb = max(0, yi - k), min(H, yi + k + 1), max(0, xi - k), min(W, xi + k + 1)
        pg, pt = g[ya:yb, xa:xb], t[ya:yb, xa:xb]
        yy, xx = np.ogrid[ya:yb, xa:xb]
        d = np.hypot(yy - y, xx - x)
        cy, cx = yi - ya, xi - xa
        log_g = float(-s * s * gaussian_laplace(pg, s)[cy, cx])
        log_t = float(-s * s * gaussian_laplace(pt, s)[cy, cx])
        inner = d <= max(1.0, 0.6 * r)
        ring = (d > 1.1 * r) & (d <= 1.6 * r + 1)
        contrast = float(pg[inner].mean() - pg[ring].mean()) if ring.any() else 0.0
        t_peak = float(pt[inner].mean())
        cyy = np.clip(np.round(y + r * np.sin(ang)).astype(int), 0, H - 1)
        cxx = np.clip(np.round(x + r * np.cos(ang)).astype(int), 0, W - 1)
        circ = grad[cyy, cxx]
        gin = grad[ya:yb, xa:xb][d <= max(1.0, 0.5 * r)].mean()
        edge_ratio = float(circ.mean() / (gin + 0.02))
        edge_frac_v = float((circ > 0.5 * circ.max() + 1e-6).mean())
        # signed distance of the centre to the host boundary, in satellite radii
        host = labels[ya:yb, xa:xb] == lab
        if host[cy, cx]:
            sd = float(ndi.distance_transform_edt(host)[cy, cx]) / r
        else:
            sd = -float(dist_out[yi, xi]) / r
        tex = float(pg[host & (d > 1.6 * r)].std()) if (host & (d > 1.6 * r)).any() else 0.05
        conc = float(np.hypot(y - cents[lab - 1, 0], x - cents[lab - 1, 1]) / R)  # 0 = host centre
        rows.append([x, y, r, int(lab), log_g, log_t, t_peak, contrast, edge_ratio, edge_frac_v,
                     r / R, sd, np.log(r), log_g / (tex + 0.02), conc])
    return pd.DataFrame(rows, columns=cols)


def _big_neighbour(labels, lab, req):
    ys, xs = np.nonzero(labels == lab)
    if not len(ys):
        return None
    ya, yb = max(0, ys.min() - 2), ys.max() + 3
    xa, xb = max(0, xs.min() - 2), xs.max() + 3
    m = labels[ya:yb, xa:xb] == lab
    ring = morphology.binary_dilation(m, morphology.disk(2)) & ~m
    nb = np.unique(labels[ya:yb, xa:xb][ring])
    nb = nb[(nb > 0) & (nb != lab)]
    return int(nb[np.argmax(req[nb])]) if len(nb) else None


def load_model(path=MODEL_PATH):
    import json
    m = json.loads(Path(path).read_text())
    return m


def model_scores(cand, m):
    X = cand[m["features"]].to_numpy(float)
    Z = (X - np.array(m["mean"])) / np.array(m["std"])
    return 1 / (1 + np.exp(-(Z @ np.array(m["coef"]) + m["intercept"])))


def detect_model(img_u8, labels, prob=None, model=None, **kw):
    m = load_model() if model is None else model
    cand = candidates(img_u8, labels, **kw)
    if not len(cand):
        return pd.DataFrame(columns=["x", "y", "r_px", "parent", "score", "source"])
    cand["score"] = model_scores(cand, m)
    cand = cand[cand.score >= (m["threshold"] if prob is None else prob)]
    cand = cand.sort_values("score", ascending=False)
    out = dedupe(cand[["x", "y", "r_px", "parent", "score"]].assign(source="model"))
    return out

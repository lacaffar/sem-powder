"""Powder QC: sphericity and satellites per sample, with confidence intervals, and a
before/after comparison (e.g. atomizer jet change).

Per particle (particle model; edge-touching particles excluded):
  sphericity   = minor / major axis of the particle outline (moment ellipse); 1 = circle
  feret_ratio  = min / max caliper diameter (second sphericity view)
  n_satellites = satellites matched to it: a satellite belongs to the particle that covers
                 >= 50% of its area (Cohn et al. 2021 matching rule)
  satellited   = n_satellites >= 1
Per sample (all images pooled):
  n_particles, pct_spherical (sphericity >= SPHERICAL), median sphericity, pct_satellited,
  satellites_per_particle, D10/D50/D90 (number and volume, ECD), each with a 95% bootstrap CI.
  Bootstrap resamples whole images when a sample has >= 3 images (captures field-to-field
  variation), otherwise particles. B = 1000 ("run the same measurement 1000 times").
compare(A, B): difference B - A for every metric, 95% bootstrap CI of the difference,
  two-sided bootstrap p-value, and a verdict (CHANGED when the CI excludes 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage import measure

from . import classic
from .measure import particle_table, wpct

SPHERICAL = 0.90
B = 1000
METRICS = ["pct_spherical", "sphericity_median", "pct_satellited", "satellites_per_particle",
           "D10_num", "D50_num", "D90_num", "D50_vol"]


def match_satellites(particles, satellites):
    """-> dict particle_label -> n satellites; satellites that match no particle are ignored."""
    counts = {}
    if satellites.max() == 0 or particles.max() == 0:
        return counts
    for rp in measure.regionprops(satellites):
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        under = particles[rr, cc]
        under = under[under > 0]
        if not len(under):
            continue
        lab, n = np.unique(under, return_counts=True)
        j = np.argmax(n)
        if n[j] >= 0.5 * rp.area:
            counts[int(lab[j])] = counts.get(int(lab[j]), 0) + 1
    return counts


def image_particles(img, part_lab, sat_lab, um_per_px):
    """Per-particle table for one image. part_lab must NOT be border-cleared yet (satellites
    are matched first, then edge-touching particles are dropped)."""
    counts = match_satellites(part_lab, sat_lab)
    kept = classic.postprocess(part_lab)          # clear border + min area, relabels
    # map kept labels back to original labels through the pixel overlap
    df = particle_table(kept, um_per_px)
    if not len(df):
        return df
    orig = {}
    for rp in measure.regionprops(kept, intensity_image=part_lab):
        v = part_lab[rp.coords[:, 0], rp.coords[:, 1]]
        orig[rp.label] = int(np.bincount(v).argmax())
    props = {rp.label: rp for rp in measure.regionprops(kept)}
    df["sphericity"] = [props[i].axis_minor_length / props[i].axis_major_length
                        if props[i].axis_major_length > 0 else np.nan for i in df.id]
    df["feret_ratio"] = df["feret_ar"]
    df["n_satellites"] = [counts.get(orig[i], 0) for i in df.id]
    df["satellited"] = df.n_satellites > 0
    return df


def sample_metrics(df, calibrated):
    col = "ecd_um" if calibrated else "ecd_px"
    d = df[col].to_numpy(float)
    if not len(df):
        return {m: np.nan for m in METRICS}
    n10, n50, n90 = np.percentile(d, [10, 50, 90])
    return {"pct_spherical": 100 * float((df.sphericity >= SPHERICAL).mean()),
            "sphericity_median": float(df.sphericity.median()),
            "pct_satellited": 100 * float(df.satellited.mean()),
            "satellites_per_particle": float(df.n_satellites.mean()),
            "D10_num": n10, "D50_num": n50, "D90_num": n90,
            "D50_vol": wpct(d, d ** 3 * df.edge_weight.to_numpy(float))[1]}


def bootstrap(df, calibrated, b=B, seed=0):
    """-> array (b, len(METRICS)) of resampled metrics."""
    rng = np.random.default_rng(seed)
    images = df["image"].unique()
    out = np.empty((b, len(METRICS)))
    groups = {k: g for k, g in df.groupby("image")}
    for i in range(b):
        if len(images) >= 3:
            pick = rng.choice(images, len(images), replace=True)
            s = pd.concat([groups[k] for k in pick], ignore_index=True)
        else:
            s = df.iloc[rng.integers(0, len(df), len(df))]
        m = sample_metrics(s, calibrated)
        out[i] = [m[k] for k in METRICS]
    return out


def summarize_sample(df, calibrated):
    m = sample_metrics(df, calibrated)
    bs = bootstrap(df, calibrated) if len(df) else np.full((1, len(METRICS)), np.nan)
    row = {"n_images": int(df["image"].nunique()) if len(df) else 0, "n_particles": int(len(df)),
           "size_unit": "um" if calibrated else "px",
           "bootstrap_unit": "images" if len(df) and df["image"].nunique() >= 3 else "particles"}
    for k, j in zip(METRICS, range(len(METRICS))):
        lo, hi = np.nanpercentile(bs[:, j], [2.5, 97.5])
        row[k] = m[k]
        row[f"{k}_ci95"] = f"{lo:.3g} to {hi:.3g}"
    return row, bs


def compare(bs_a, bs_b, m_a, m_b):
    """Bootstrap comparison of two samples. -> DataFrame."""
    rows = []
    for j, k in enumerate(METRICS):
        diff = bs_b[:, j] - bs_a[:, j]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        p = 2 * min(np.nanmean(diff <= 0), np.nanmean(diff >= 0))
        rows.append({"metric": k, "A": m_a[k], "B": m_b[k], "B_minus_A": m_b[k] - m_a[k],
                     "ci95_low": lo, "ci95_high": hi, "p_value": min(1.0, p),
                     "verdict": "CHANGED" if (lo > 0 or hi < 0) else "no detectable change"})
    return pd.DataFrame(rows)


def load_qc_models():
    from .models import ROOT, make
    cfg_p = ROOT / "models" / "qc_models.json"
    if not cfg_p.exists():
        raise FileNotFoundError("models/qc_models.json missing: run the 'qcmodels' stage first")
    cfg = json.loads(cfg_p.read_text())
    return _load(cfg["particles"]), _load(cfg["satellites"], satellites=True), cfg


def _load(spec, satellites=False):
    from .maskrcnn import MaskRCNNSeg
    from .models import ROOT, CellposeSeg
    p = ROOT / spec["path"]
    if spec["kind"] == "maskrcnn":
        kw = {k: spec[k] for k in ("score_thresh", "scale") if k in spec}
        s = MaskRCNNSeg(p, name=spec["name"], min_area=8 if satellites else classic.MIN_AREA_PX, **kw)
    else:
        s = CellposeSeg(p, "native", name=spec["name"])
    return s


def run(items, out_dir, um_per_px=None, log=print, part_seg=None, sat_seg=None):
    """items: list of (sample, path). Writes per-particle CSVs, overlays, qc_samples.csv.
    Returns (samples DataFrame, {sample: bootstrap array}, {sample: point metrics})."""
    from .imio import load_gray
    from .scale import calibrate
    out = Path(out_dir)
    (out / "particles").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    own = part_seg is None
    if own:
        part_seg, sat_seg, _ = load_qc_models()
    per_sample = {}
    for i, (sample, path) in enumerate(items, 1):
        try:
            img = load_gray(path)
            sc = calibrate(path, img=img, um_per_px=um_per_px)
            y0, y1, x0, x1 = sc.crop_box
            img = img[y0:y1, x0:x1]
            pl = part_seg.segment(img).astype(np.int32)
            sl = sat_seg.segment(img).astype(np.int32)
            df = image_particles(img, pl, sl, sc.um_per_px)
        except Exception as e:
            log(f"  [{i}/{len(items)}] {sample}/{Path(path).name}: FAILED {e}")
            continue
        df.insert(0, "image", Path(path).name)
        (out / "particles" / sample).mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "particles" / sample / f"{Path(path).stem}.csv", index=False)
        ov = classic.overlay(img, classic.postprocess(pl),
                             title=f"{Path(path).name}: {len(df)} particles, {int(df.n_satellites.sum()) if len(df) else 0} satellites")
        b = (classic.segmentation.find_boundaries(sl, mode="inner") if sl.max() else np.zeros_like(sl, bool))
        ov[b] = (0, 0, 255)
        (out / "overlays" / sample).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "overlays" / sample / f"{Path(path).stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
        per_sample.setdefault(sample, []).append((df, sc.um_per_px is not None))
        log(f"  [{i}/{len(items)}] {sample}/{Path(path).name}: {len(df)} particles, "
            f"{100 * df.satellited.mean() if len(df) else float('nan'):.0f}% satellited, "
            f"median sphericity {df.sphericity.median() if len(df) else float('nan'):.3f}")
    rows, boots, points = [], {}, {}
    for sample, parts in per_sample.items():
        calibrated = all(c for _, c in parts)
        df = pd.concat([d for d, _ in parts], ignore_index=True)
        row, bs = summarize_sample(df, calibrated)
        rows.append({"sample": sample, **row})
        boots[sample] = bs
        points[sample] = sample_metrics(df, calibrated)
    samples = pd.DataFrame(rows)
    samples.to_csv(out / "qc_samples.csv", index=False)
    np.savez(out / "qc_bootstrap.npz", **{k: v for k, v in boots.items()})
    json.dump(points, open(out / "qc_points.json", "w"), indent=1, default=float)
    if own:
        part_seg.close()
        sat_seg.close()
    return samples, boots, points

"""Per-particle and per-image size/shape metrics, plus PSD plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

from .imio import load_gray
from .scale import calibrate
from .segment import segment

PCTS = (10, 50, 90)


def particle_table(labels: np.ndarray, um_per_px: float | None) -> pd.DataFrame:
    s = um_per_px if um_per_px else 1.0
    H, W = labels.shape
    rows = []
    for rp in measure.regionprops(labels):
        y0, x0, y1, x1 = rp.bbox
        bw, bh = x1 - x0, y1 - y0
        a = float(rp.area)
        p = float(rp.perimeter) or np.nan
        maj, mnr = float(rp.axis_major_length), float(rp.axis_minor_length)
        ecd_px = 2.0 * np.sqrt(a / np.pi)
        rows.append({
            "id": rp.label,
            "area_px": a,
            "area_um2": a * s * s if um_per_px else np.nan,
            "ecd_px": ecd_px,
            "ecd_um": ecd_px * s if um_per_px else np.nan,
            "perimeter_px": p,
            # pixel perimeters overestimate for tiny objects; clip to the physical max
            "circularity": min(1.0, 4 * np.pi * a / (p * p)) if p == p else np.nan,
            "aspect_ratio": maj / mnr if mnr > 0 else np.nan,
            "solidity": float(rp.solidity),
            "centroid_y": float(rp.centroid[0]),
            "centroid_x": float(rp.centroid[1]),
            # Miles-Lantuejoul edge correction: edge-touching particles are dropped,
            # so a particle of bbox (bw, bh) is only kept if it lies in a
            # (W-bw) x (H-bh) window. Weight = W*H / that area (1 for tiny ones).
            "edge_weight": (W * H) / max(1.0, (W - bw) * (H - bh)),
        })
    cols = ["id", "area_px", "area_um2", "ecd_px", "ecd_um", "perimeter_px", "circularity",
            "aspect_ratio", "solidity", "centroid_y", "centroid_x", "edge_weight"]
    return pd.DataFrame(rows, columns=cols)


def weighted_percentiles(x, w, pcts=PCTS):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if len(x) == 0:
        return [np.nan] * len(pcts)
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(p / 100.0, c, x)) for p in pcts]


def summarize(df: pd.DataFrame, um_per_px: float | None) -> dict:
    col = "ecd_um" if um_per_px else "ecd_px"
    unit = "um" if um_per_px else "px"
    d = df[col].to_numpy() if len(df) else np.array([])
    out = {"count": int(len(df)), "size_unit": unit, "um_per_px": um_per_px}
    nn = np.percentile(d, PCTS) if len(d) else [np.nan] * 3
    vv = weighted_percentiles(d, d ** 3)
    for p, a, b in zip(PCTS, nn, vv):
        out[f"D{p}_num"] = float(a)
        out[f"D{p}_vol"] = float(b)
    # edge-corrected volume-weighted percentiles (Miles-Lantuejoul)
    if len(d):
        ew = df["edge_weight"].to_numpy()
        for p, b in zip(PCTS, weighted_percentiles(d, d ** 3 * ew)):
            out[f"D{p}_vol_ec"] = float(b)
    else:
        for p in PCTS:
            out[f"D{p}_vol_ec"] = np.nan
    for k in ("circularity", "aspect_ratio", "solidity"):
        out[f"mean_{k}"] = float(df[k].mean()) if len(df) else np.nan
    # satellite / agglomerate flag: non-convex outline (satellites attached,
    # merged clusters). Lowest-priority metric, coarse proxy only.
    out["irregular_frac"] = float((df["solidity"] < 0.90).mean()) if len(df) else np.nan
    return out


def analyze_image(path, method="watershed", um_per_px=None, sidecar=None, scale_debug=None):
    """Returns (particles_df, summary_dict, labels, cropped_img, scale_result)."""
    img = load_gray(path)
    sc = calibrate(path, img=img, um_per_px=um_per_px, sidecar=sidecar, debug_path=scale_debug)
    y0, y1, x0, x1 = sc.crop_box
    img = img[y0:y1, x0:x1]
    labels, info = segment(img, method)
    df = particle_table(labels, sc.um_per_px)
    summ = summarize(df, sc.um_per_px)
    summ.update({"image": Path(path).name, "method": method, "scale_source": sc.source,
                 "scale_confidence": round(sc.confidence, 3), "seconds": info.get("seconds")})
    return df, summ, labels, img, sc


def plot_psd(df_or_list, out, unit="um", labels=None, title=None):
    """Number histogram + number/volume CDF. Accepts one df or a list of dfs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dfs = df_or_list if isinstance(df_or_list, list) else [df_or_list]
    labels = labels or [None] * len(dfs)
    col = "ecd_um" if unit == "um" else "ecd_px"
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    allv = np.concatenate([d[col].dropna().to_numpy() for d in dfs if len(d)]) if any(len(d) for d in dfs) else np.array([1.0])
    bins = np.linspace(0, np.percentile(allv, 99.5) * 1.1 + 1e-9, 30)
    for d, lb in zip(dfs, labels):
        v = d[col].dropna().to_numpy()
        if not len(v):
            continue
        ax[0].hist(v, bins=bins, histtype="step" if len(dfs) > 1 else "bar", alpha=0.8, label=lb)
        s = np.sort(v)
        ax[1].plot(s, np.arange(1, len(s) + 1) / len(s), label=(lb + " " if lb else "") + "number")
        w = s ** 3
        ax[1].plot(s, np.cumsum(w) / w.sum(), "--", label=(lb + " " if lb else "") + "volume")
    ax[0].set_xlabel(f"ECD ({unit})")
    ax[0].set_ylabel("count")
    ax[1].set_xlabel(f"ECD ({unit})")
    ax[1].set_ylabel("cumulative fraction")
    ax[1].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

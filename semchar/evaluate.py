"""Scoring a label image against ground-truth instance masks.

Instance match: IoU >= 0.5 (unique by construction). Reported:
  precision, recall, F1 at IoU 0.5 and 0.75, mean IoU of matched particles,
  D50 (number) and D50 (volume) error of the predicted vs true size distribution,
  matched-particle ECD ratio (median), and median circularity / feret_ar error.
"""
from __future__ import annotations

import numpy as np

from .measure import particle_table, wpct


def iou_pairs(gt, pr):
    """IoU for every overlapping (gt, pred) pair: arrays g, p, iou."""
    g = gt.ravel().astype(np.int64)
    p = pr.ravel().astype(np.int64)
    m = (g > 0) & (p > 0)
    if not m.any():
        return np.array([], int), np.array([], int), np.array([])
    K = int(p.max()) + 1
    inter = np.bincount(g[m] * K + p[m])
    idx = np.nonzero(inter)[0]
    gi, pi = idx // K, idx % K
    ag = np.bincount(g)[gi]
    ap = np.bincount(p, minlength=K)[pi]
    iou = inter[idx] / (ag + ap - inter[idx])
    return gi, pi, iou


def score(gt, pr):
    ng = len(np.unique(gt)) - (1 if (gt == 0).any() else 0)
    npred = len(np.unique(pr)) - (1 if (pr == 0).any() else 0)
    gi, pi, iou = iou_pairs(gt, pr)
    out = {"n_gt": ng, "n_pred": npred}
    for t in (0.5, 0.75):
        tp = int((iou >= t).sum())
        P, R = tp / max(1, npred), tp / max(1, ng)
        out[f"P@{t}"], out[f"R@{t}"] = P, R
        out[f"F1@{t}"] = 2 * P * R / max(1e-9, P + R)
    sel = iou >= 0.5
    out["mean_iou_matched"] = float(iou[sel].mean()) if sel.any() else np.nan
    tg, tp_ = particle_table(gt), particle_table(pr)
    for name, w in (("num", lambda d: np.ones_like(d)), ("vol", lambda d: d ** 3)):
        dg, dp = tg.ecd_px.to_numpy(float), tp_.ecd_px.to_numpy(float)
        g50 = wpct(dg, w(dg))[1] if len(dg) else np.nan
        p50 = wpct(dp, w(dp))[1] if len(dp) else np.nan
        out[f"D50_{name}_err"] = p50 / g50 - 1 if g50 == g50 and g50 > 0 else np.nan
    if sel.any():
        eg = dict(zip(tg.id, tg.ecd_px))
        ep = dict(zip(tp_.id, tp_.ecd_px))
        out["matched_ecd_ratio"] = float(np.median([ep[b] / eg[a] for a, b in zip(gi[sel], pi[sel])
                                                    if a in eg and b in ep]))
    else:
        out["matched_ecd_ratio"] = np.nan
    for k in ("circularity", "feret_ar"):
        out[f"{k}_median_err"] = (float(tp_[k].median() - tg[k].median())
                                  if len(tg) and len(tp_) else np.nan)
    return out


def pool(rows):
    """Pool per-image scores: counts summed, ratios averaged."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {"n_images": len(df), "n_gt": int(df.n_gt.sum()), "n_pred": int(df.n_pred.sum())}
    for t in (0.5, 0.75):
        tp = (df[f"R@{t}"] * df.n_gt).sum()
        P, R = tp / max(1, out["n_pred"]), tp / max(1, out["n_gt"])
        out[f"P@{t}"], out[f"R@{t}"], out[f"F1@{t}"] = P, R, 2 * P * R / max(1e-9, P + R)
    for c in ("mean_iou_matched", "matched_ecd_ratio"):
        out[c] = float(df[c].mean())
    for c in ("D50_num_err", "D50_vol_err", "circularity_median_err", "feret_ar_median_err"):
        out[c + "_mean"] = float(df[c].mean())
        out[c + "_absmean"] = float(df[c].abs().mean())
    if "seconds" in df:
        out["sec_per_image"] = float(df.seconds.mean())
    return out

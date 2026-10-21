"""Score satellite identification on the synthetic satellite benchmark.

  python -m src.validate_satellites --split tune|test --seg watershed --det blob [k=v ...]

Metrics per configuration (pooled over the split):
  precision, recall, F1  : detected satellite centres matched 1:1 to visible GT
                           satellites on measurable parents (parent not cut by
                           the frame; the tool drops those parents by design),
                           tolerance max(3 px, 0.75 r_gt). recall_all_gt also
                           counts satellites on edge-cut parents.
  count_ratio            : detected / GT satellites per image (mean)
  frac_err               : |pred - true| fraction of parent particles carrying
                           >= 1 satellite, per image (mean, percentage points)
Baseline `none` scores the tool before this work: no satellites detected, and
the old proxy (solidity < 0.90) is used for the parent fraction.
GT parents for the fraction: parent spheres not touching the frame with
>= 50% visible. Pred parents: segmented particles that are not themselves
satellites; both restricted to equivalent radius >= 0.5 x median parent radius.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/synth_sat"
SPLITS = {"tune": range(0, 30), "test": range(30, 60)}


def load(seed):
    img = cv2.imread(str(BENCH / f"sat_{seed:03d}.png"), cv2.IMREAD_GRAYSCALE)
    meta = json.load(open(BENCH / f"sat_{seed:03d}.json"))
    return img, meta


def gt_tables(meta):
    sp = pd.DataFrame(meta["spheres"])
    H, W = 512, 640
    sats = sp[sp.gt_satellite].copy()
    par = sp[sp.kind == "parent"].copy()
    par["pid"] = par.index
    inside = (par.x - par.r > 1) & (par.x + par.r < W - 1) & (par.y - par.r > 1) & (par.y + par.r < H - 1)
    # satellites on parents the tool can measure (parent not cut by the frame)
    sats["measurable"] = sats.parent.map(inside).fillna(False).astype(bool)
    par = par[inside & (par.vis_frac >= 0.5)]
    par["n_sat"] = [int(((sats.parent == i)).sum()) for i in par.pid]
    return sats, par


def score(seed, seg, det, kw, cache):
    from .matching import match
    from .metrics import particle_table
    from .satellites import detect
    from .segment import segment
    img, meta = load(seed)
    key = (seed, seg)
    if key not in cache:
        cache[key] = segment(img, seg)[0]
    labels = cache[key]
    sats_all, par_gt = gt_tables(meta)
    pred = detect(img, labels, det, **kw)

    def _m(gt):  # tolerance max(3 px, 0.75 r_gt) = max(3, 0.375 d_gt)
        if not len(pred) or not len(gt):
            return np.array([], int), np.array([], int)
        return match(pred[["x", "y"]].to_numpy(float), gt[["x", "y"]].to_numpy(),
                     2 * gt.r.to_numpy(), tol_frac=0.375, tol_min=3.0)

    sats_gt = sats_all[sats_all.measurable]
    mi, gi = _m(sats_gt)
    _, gi_all = _m(sats_all)
    # parent fraction
    pt = particle_table(labels, None)
    sat_labels = set()
    if len(pred) and "self_label" in pred:
        sat_labels = set(pred.self_label.dropna().astype(int))
    pt["r"] = pt.ecd_px / 2
    pt = pt[~pt.id.isin(sat_labels)]
    rmin_gt = 0.5 * par_gt.r.median() if len(par_gt) else 0
    gtp = par_gt[par_gt.r >= rmin_gt]
    true_frac = float((gtp.n_sat > 0).mean()) if len(gtp) else np.nan
    pp = pt[pt.r >= rmin_gt]
    if det == "none":
        pred_frac = float((pp.solidity < 0.90).mean()) if len(pp) else np.nan
    else:
        has = set(pred.parent.astype(int)) if len(pred) else set()
        pred_frac = float(pp.id.isin(has).mean()) if len(pp) else np.nan
    return {"seed": seed, "style": meta["style"], "n_gt": len(sats_gt), "n_pred": len(pred),
            "tp": len(gi), "n_gt_all": len(sats_all), "tp_all": len(gi_all),
            "true_frac": true_frac, "pred_frac": pred_frac}


def run(split, seg, det, kw, cache=None, quiet=False):
    cache = {} if cache is None else cache
    t0 = time.time()
    rows = [score(s, seg, det, kw, cache) for s in SPLITS[split]]
    df = pd.DataFrame(rows)
    tp, npred, ngt = df.tp.sum(), df.n_pred.sum(), df.n_gt.sum()
    P = tp / max(1, npred)
    R = tp / max(1, ngt)
    F = 2 * P * R / max(1e-9, P + R)
    out = {"split": split, "seg": seg, "det": det, "params": json.dumps(kw), "precision": P, "recall": R, "F1": F,
           "count_ratio": float((df.n_pred / df.n_gt).mean()),
           "frac_err_pts": float(np.nanmean(np.abs(df.pred_frac - df.true_frac)) * 100),
           "true_frac": float(df.true_frac.mean()), "pred_frac": float(df.pred_frac.mean()),
           "F1_se": _f1(df[df["style"] == "se"]), "F1_bse": _f1(df[df["style"] == "bse"]),
           "recall_all_gt": df.tp_all.sum() / max(1, df.n_gt_all.sum()),
           "sec": time.time() - t0}
    if not quiet:
        print(f"{split} seg={seg} det={det} {kw}: P={P:.3f} R={R:.3f} F1={F:.3f} "
              f"count_ratio={out['count_ratio']:.2f} frac_err={out['frac_err_pts']:.1f}pts "
              f"(true {out['true_frac']:.2f} pred {out['pred_frac']:.2f}) F1 se/bse={out['F1_se']:.3f}/{out['F1_bse']:.3f} "
              f"recall(all GT incl. edge parents)={out['recall_all_gt']:.3f}",
              flush=True)
    return out


def _f1(d):
    tp, npred, ngt = d.tp.sum(), d.n_pred.sum(), d.n_gt.sum()
    P, R = tp / max(1, npred), tp / max(1, ngt)
    return 2 * P * R / max(1e-9, P + R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune")
    ap.add_argument("--seg", default="watershed")
    ap.add_argument("--det", default="blob")
    ap.add_argument("params", nargs="*", help="detector params k=v")
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (p.split("=") for p in a.params)}
    run(a.split, a.seg, a.det, kw)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

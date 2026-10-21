"""Fit the satellite candidate scorer (logistic regression, numpy).

Training data: synthetic benchmark TUNING split (seeds 0-29) plus, with
--real-dev, candidates inside the hand-labelled crops of the real DEV images
(AA7075 Fig 2a, Ti-6Al-4V; tests/satellite_labels_real.json), weighted x REAL_W.
The probability cut-off is chosen on the real dev crops when used, else on
synthetic tuning. Real TEST images (Fig 2c, Ti64+15Cu) and synthetic seeds
30-59 are never used. Writes src/satellite_model.json.

  python -m src.train_satellites [--seg watershed] [--real-dev]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


REAL_W = 5.0


def fit_logreg(X, y, l2=1.0, iters=3000, lr=0.5, sample_w=None):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    pos = y.mean()
    # balance classes so the rare positives are not ignored
    sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    if sample_w is not None:
        sw = sw * sample_w / sample_w.mean()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        gr = (p - y) * sw
        w -= lr * (X.T @ gr / n + l2 * w / n)
        b -= lr * gr.mean()
    return w, b


def collect(seeds, seg, cache):
    from .matching import match
    from .satellites import candidates
    from .segment import segment
    from .validate_satellites import gt_tables, load
    out = []
    for s in seeds:
        img, meta = load(s)
        if (s, seg) not in cache:
            cache[(s, seg)] = segment(img, seg)[0]
        lab = cache[(s, seg)]
        gt, _ = gt_tables(meta)
        gt = gt[gt.measurable]  # satellites on parents the tool can measure
        c = candidates(img, lab)
        c["seed"] = s
        c["n_gt"] = len(gt)
        c["y"] = c["y"].astype(float)
        c["label"] = 0
        c["gt_idx"] = -1
        if len(c):
            mi, gi = match(c[["x", "y"]].to_numpy(float), gt[["x", "y"]].to_numpy(), 2 * gt.r.to_numpy(),
                           tol_frac=0.375, tol_min=3.0)
            c.loc[c.index[mi], "label"] = 1
            c.loc[c.index[mi], "gt_idx"] = gi
        out.append(c)
        print(f"seed {s}: {len(c)} candidates, {int(c.label.sum())} match GT ({len(gt)} GT)", flush=True)
    return pd.concat(out, ignore_index=True)


def collect_real(split="dev", seg="watershed"):
    """Candidates inside labelled real crops (outside ignore boxes), with labels."""
    from .imio import load_gray
    from .matching import match
    from .satellites import candidates
    from .scale import calibrate
    from .segment import segment
    L = json.loads((ROOT / "tests/satellite_labels_real.json").read_text())
    out = []
    for name in L["split"][split]:
        p = ROOT / L["images"][name]
        img = load_gray(p)
        y0, y1, x0, x1 = calibrate(p, img=img, use_meta=False).crop_box
        img = img[y0:y1, x0:x1]
        lab = segment(img, seg)[0]
        c_all = candidates(img, lab)
        for ck, (cy0, cy1, cx0, cx1) in L["crops"].items():
            key = f"{name}_{ck}"
            g = np.array(L["labels"][key]["sats"], float).reshape(-1, 3)
            inb = lambda x, y, b: (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])  # noqa: E731
            box = (cx0, cy0, cx1, cy1)
            kg = inb(g[:, 0], g[:, 1], box)
            kc = inb(c_all.x.to_numpy(), c_all.y.to_numpy(), box)
            for b in L["labels"][key]["ignore"]:
                kg &= ~inb(g[:, 0], g[:, 1], b)
                kc &= ~inb(c_all.x.to_numpy(), c_all.y.to_numpy(), b)
            g, c = g[kg], c_all[kc].copy()
            c["seed"] = key
            c["n_gt"] = len(g)
            c["label"] = 0
            c["gt_idx"] = -1
            if len(c) and len(g):
                mi, gi = match(c[["x", "y"]].to_numpy(float), g[:, :2], 2 * g[:, 2], tol_frac=0.5, tol_min=4.0)
                c.loc[c.index[mi], "label"] = 1
                c.loc[c.index[mi], "gt_idx"] = gi
            out.append(c)
            print(f"real {key}: {len(c)} candidates, {int(c.label.sum())} match labels ({len(g)} labels)", flush=True)
    return pd.concat(out, ignore_index=True)


def pipeline_f1(C, scores, p, gt_total):
    """F1 after threshold + dedupe, as the detector would run."""
    from .matching import match  # noqa: F401  (same matching as validation)
    from .satellites import dedupe
    tp = npred = 0
    for s, d in C.assign(score=scores).groupby("seed"):
        d = d[d.score >= p].sort_values("score", ascending=False)
        d = dedupe(d)
        npred += len(d)
        tp += int((d.gt_idx >= 0).sum() - d[d.gt_idx >= 0].gt_idx.duplicated().sum())
    P, R = tp / max(1, npred), tp / max(1, gt_total)
    return 2 * P * R / max(1e-9, P + R), P, R


def main():
    from .satellites import FEATURES, MODEL_PATH
    from .validate_satellites import SPLITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="watershed")
    ap.add_argument("--real-dev", action="store_true")
    a = ap.parse_args()
    C = collect(SPLITS["tune"], a.seg, {})
    C["w"] = 1.0
    parts = [C]
    if a.real_dev:
        R = collect_real("dev", a.seg)
        R["w"] = REAL_W
        parts.append(R)
    A = pd.concat(parts, ignore_index=True)
    X = A[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = fit_logreg(Z, A.label.to_numpy(float), sample_w=A.w.to_numpy())
    sig = lambda D: 1 / (1 + np.exp(-(((D[FEATURES].to_numpy(float) - mu) / sd) @ w + b)))  # noqa: E731
    grid = np.concatenate([np.linspace(0.3, 0.95, 14), np.linspace(0.955, 0.995, 9)])
    sel = R if a.real_dev else C
    gt_sel = sel.groupby("seed").n_gt.first().sum()
    best = max((pipeline_f1(sel, sig(sel), p, gt_sel) + (p,) for p in grid), key=lambda t: t[0])
    syn = pipeline_f1(C, sig(C), best[3], C.groupby("seed").n_gt.first().sum())
    print("coef:", {k: round(float(v), 3) for k, v in zip(FEATURES, w)}, "intercept", round(b, 3))
    print(f"cut-off p={best[3]:.2f} chosen on {'real dev' if a.real_dev else 'synthetic tune'}: "
          f"F1={best[0]:.3f} P={best[1]:.3f} R={best[2]:.3f} | synthetic tune F1={syn[0]:.3f}")
    model = {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(), "coef": w.tolist(),
             "intercept": float(b), "threshold": float(best[3]),
             "trained_on": "synthetic satellite benchmark seeds 0-29" + (" + real dev crops (Fig 2a, Ti-6Al-4V) x%g" % REAL_W if a.real_dev else "") + ", seg=" + a.seg}
    MODEL_PATH.write_text(json.dumps(model, indent=1))
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

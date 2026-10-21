"""Score satellite detection on hand-labelled crops of real SEM images.

  python -m src.validate_satellites_real [--split dev|test|all] [--det model|blob|seg_touch|none]

Labels: tests/satellite_labels_real.json (see its 'about'). Whole image is
segmented and searched; only detections and labels inside the crop and outside
ignore boxes are scored. Tolerance max(4 px, r_label).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "tests/satellite_labels_real.json"


def _inside(x, y, box):
    x0, y0, x1, y1 = box
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def run(split="dev", det="model", seg="watershed", kw=None, cache=None, quiet=False):
    from .imio import load_gray
    from .matching import match
    from .satellites import detect
    from .scale import calibrate
    from .segment import segment
    kw = kw or {}
    cache = {} if cache is None else cache
    L = json.loads(LABELS.read_text())
    names = L["split"]["dev"] + L["split"]["test"] if split == "all" else L["split"][split]
    rows = []
    for name in names:
        if (name, seg) not in cache:
            p = ROOT / L["images"][name]
            img = load_gray(p)
            y0, y1, x0, x1 = calibrate(p, img=img, use_meta=False).crop_box
            img = img[y0:y1, x0:x1]
            cache[(name, seg)] = (img, segment(img, seg)[0])
        img, lab = cache[(name, seg)]
        pred = detect(img, lab, det, **kw)
        for c, (cy0, cy1, cx0, cx1) in L["crops"].items():
            key = f"{name}_{c}"
            if key not in L["labels"]:
                continue
            g = np.array(L["labels"][key]["sats"], float).reshape(-1, 3)
            ign = L["labels"][key]["ignore"]
            box = (cx0, cy0, cx1, cy1)
            keep_g = _inside(g[:, 0], g[:, 1], box)
            for b in ign:
                keep_g &= ~_inside(g[:, 0], g[:, 1], b)
            g = g[keep_g]
            pr = pred.copy()
            if len(pr):
                k = _inside(pr.x.to_numpy(), pr.y.to_numpy(), box)
                for b in ign:
                    k &= ~_inside(pr.x.to_numpy(), pr.y.to_numpy(), b)
                pr = pr[k]
            if len(pr) and len(g):
                mi, gi = match(pr[["x", "y"]].to_numpy(float), g[:, :2], 2 * g[:, 2], tol_frac=0.5, tol_min=4.0)
            else:
                mi, gi = [], []
            rows.append({"crop": key, "n_gt": len(g), "n_pred": len(pr), "tp": len(gi)})
    df = pd.DataFrame(rows)
    tp, npred, ngt = df.tp.sum(), df.n_pred.sum(), df.n_gt.sum()
    P, R = tp / max(1, npred), tp / max(1, ngt)
    F = 2 * P * R / max(1e-9, P + R)
    if not quiet:
        print(f"real {split} det={det} {kw}: P={P:.3f} R={R:.3f} F1={F:.3f} (tp {tp}, pred {npred}, gt {ngt}) | "
              + " ".join(f"{r.crop}:{r.tp}/{r.n_pred}/{r.n_gt}" for r in df.itertuples()))
    return {"split": split, "det": det, "P": P, "R": R, "F1": F, "tp": int(tp), "pred": int(npred), "gt": int(ngt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--det", default="model")
    ap.add_argument("params", nargs="*")
    a = ap.parse_args()
    run(a.split, a.det, kw={k: float(v) for k, v in (p.split("=") for p in a.params)})


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

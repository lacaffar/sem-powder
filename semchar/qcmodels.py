"""Stage 'qcmodels': train and benchmark particle + satellite models on the public labeled
SEM powder data (AMPIS, SALAS), pick the best of each, then run QC on the real samples.

Contenders
  particles : cpsam (zero-shot), cp_particles (Cellpose-SAM fine-tuned on AMPIS particles),
              mrcnn_particles (Mask R-CNN, the papers' method)
  satellites: cp_satellites (Cellpose-SAM fine-tuned on AMPIS + SALAS satellites),
              mrcnn_satellites (Mask R-CNN)
Scoring on the published validation images (no edge removal, like Cohn 2021):
  per-instance precision / recall / F1 at IoU >= 0.5 (Cohn's metric),
  per-pixel precision / recall (Price's metric), per source (AMPIS / SALAS),
  % satellited particles on the AMPIS image that has both particle and satellite labels.
Published references: Cohn particles P 0.938 R 0.799, satellites P 0.692 R 0.545 (5-fold CV,
AMPIS); Price satellites pixel P 0.796 R 0.746 across 5 powders (SALAS).
Satellite Mask R-CNN tuning: inference scale x score threshold swept on the validation images.
  Its reported score is leave-one-image-out (setting chosen on the other 7 images, scored on the
  left-out one), so the tuned row is a held-out estimate; the final setting uses all 8 images.
Trained models in models/ are reused on later runs (set SEMCHAR_RETRAIN=1 to retrain).
Caveat: validation images also pick the fine-tuning epoch, so those scores are slightly optimistic.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import morphology, segmentation

from .evaluate import iou_pairs
from .models import ROOT, make

MODELS = ROOT / "models"
PUBLISHED = {
    "particles": {"P@0.5": 0.938, "R@0.5": 0.799, "source": "Cohn 2021, AMPIS 5-fold CV, per instance"},
    "satellites": {"P@0.5": 0.692, "R@0.5": 0.545, "pixel_P": 0.796, "pixel_R": 0.746,
                   "source": "Cohn 2021 per instance (AMPIS); Price 2021 per pixel (SALAS)"},
}


def _clean(lab, min_area):
    lab = morphology.remove_small_objects(lab.astype(np.int32), min_area)
    return segmentation.relabel_sequential(lab)[0]


def score_items(seg, items, min_area, log=print):
    """Instance + pixel scores, pooled over items [(name, img, gt, source)]."""
    rows = []
    for name, img, gt, src in items:
        t = time.time()
        pr = _clean(seg.segment(img), min_area)
        gi, pi, iou = iou_pairs(gt, pr)
        tp = int((iou >= 0.5).sum())
        ng, npd = int(gt.max()), int(pr.max())
        pix_tp = int(((gt > 0) & (pr > 0)).sum())
        rows.append({"image": name, "source": src, "n_gt": ng, "n_pred": npd, "tp": tp,
                     "pix_tp": pix_tp, "pix_pred": int((pr > 0).sum()), "pix_gt": int((gt > 0).sum()),
                     "mean_iou": float(iou[iou >= 0.5].mean()) if tp else np.nan, "seconds": time.time() - t})
    df = pd.DataFrame(rows)
    return by_source(df), df


def agg(d):
    P = d.tp.sum() / max(1, d.n_pred.sum())
    R = d.tp.sum() / max(1, d.n_gt.sum())
    return {"n_images": len(d), "n_gt": int(d.n_gt.sum()), "P@0.5": P, "R@0.5": R,
            "F1@0.5": 2 * P * R / max(1e-9, P + R), "count_ratio": d.n_pred.sum() / max(1, d.n_gt.sum()),
            "mean_iou": float(d.mean_iou.mean()),
            "pixel_P": d.pix_tp.sum() / max(1, d.pix_pred.sum()), "pixel_R": d.pix_tp.sum() / max(1, d.pix_gt.sum()),
            "sec_per_image": float(d.seconds.mean())}


def by_source(df):
    out = {"all": agg(df)}
    for src, d in df.groupby("source"):
        out[src] = agg(d)
    return out


SWEEP_SCALES = (1.25, 1.5625, 2.0)
SWEEP_THRESH = tuple(round(0.3 + 0.05 * i, 2) for i in range(14))   # 0.30 .. 0.95


def _best(df):
    """(scale, thresh) with the highest pooled F1 over the rows in df, and that F1."""
    g = df.groupby(["scale", "thresh"])[["tp", "n_pred", "n_gt"]].sum()
    f1 = 2 * g.tp / (g.n_pred + g.n_gt).clip(lower=1)
    k = f1.idxmax()
    return float(k[0]), float(k[1]), float(f1.max()), f1


def tune_maskrcnn(spec, items, min_area, run_dir, log=print):
    """Sweep inference scale x score threshold on items. -> (scale, thresh, held-out per-image rows).
    Held-out: each image is scored at the setting that maximizes pooled F1 on the OTHER images."""
    from .maskrcnn import MaskRCNNSeg
    seg = MaskRCNNSeg(ROOT / spec["path"], name=spec["name"], min_area=min_area)
    rows = []
    try:
        for sc in SWEEP_SCALES:
            for name, img, gt, src in items:
                t = time.time()
                masks, scores = seg.predict(img, scale=sc, min_score=min(SWEEP_THRESH))
                secs = time.time() - t
                for th in SWEEP_THRESH:
                    pr = _clean(seg.to_labels(masks, scores, th, img.shape), min_area)
                    _, _, iou = iou_pairs(gt, pr)
                    tp = int((iou >= 0.5).sum())
                    rows.append({"scale": sc, "thresh": th, "image": name, "source": src, "n_gt": int(gt.max()),
                                 "n_pred": int(pr.max()), "tp": tp, "pix_tp": int(((gt > 0) & (pr > 0)).sum()),
                                 "pix_pred": int((pr > 0).sum()), "pix_gt": int((gt > 0).sum()),
                                 "mean_iou": float(iou[iou >= 0.5].mean()) if tp else np.nan, "seconds": secs})
                del masks, scores
    finally:
        seg.close()
    df = pd.DataFrame(rows)
    df.to_csv(Path(run_dir) / f"tune_{spec['name']}.csv", index=False)
    sc, th, best_f1, f1 = _best(df)
    held = []
    for name in df.image.unique():
        s_o, t_o, _, _ = _best(df[df.image != name])
        held.append(df[(df.image == name) & (df.scale == s_o) & (df.thresh == t_o)].assign(chosen=f"{s_o}/{t_o}"))
    held = pd.concat(held, ignore_index=True)
    log(f"  sweep: best on all val images: scale {sc}, score >= {th} (F1 {best_f1:.3f}); "
        f"default scale 1.5625, score >= 0.5: F1 {f1.get((1.5625, 0.5), np.nan):.3f}")
    log("  leave-one-image-out settings: " + ", ".join(f"{r.image} {r.chosen}" for r in held.itertuples()))
    return sc, th, held


def run(run_dir, log=print, quick=False):
    from . import maskrcnn as M
    from .finetune import train_cellpose
    from .public_data import datasets, fetch_public
    run_dir = Path(run_dir)
    log("--- public labeled data")
    fetch_public()
    D = datasets()
    for t, sp in D.items():
        log(f"  {t}: train {len(sp['train'])} images ({sum(int(l.max()) for _, _, l, _ in sp['train'])} outlines), "
            f"val {len(sp['val'])} images ({sum(int(l.max()) for _, _, l, _ in sp['val'])})")
    ep = 6 if quick else 100
    every = 3 if quick else 20
    iters = 60 if quick else 2500
    MODELS.mkdir(exist_ok=True)
    tag = "_quick" if quick else ""          # quick test models never overwrite or stand in for real ones
    retrain = os.environ.get("SEMCHAR_RETRAIN") == "1"
    trained = {}

    def as_lists(items):
        return ([i for _, i, _, _ in items], [l for _, _, l, _ in items], [n for n, _, _, _ in items])

    for task in ("particles", "satellites"):
        cp, mr = MODELS / f"cp_{task}{tag}", MODELS / f"mrcnn_{task}{tag}.pt"
        log(f"--- train Cellpose-SAM on {task}")
        try:
            if cp.exists() and cp.with_suffix(".json").exists() and not retrain:
                log(f"  using existing {cp.name} (SEMCHAR_RETRAIN=1 retrains)")
            else:
                train_cellpose(as_lists(D[task]["train"]), as_lists(D[task]["val"]), cp,
                               log=log, n_epochs=ep, save_every=every, nimg_per_epoch=16)
            trained[f"cp_{task}"] = {"kind": "cellpose", "path": f"models/{cp.name}", "name": f"cp_{task}"}
        except Exception as e:
            log(f"  FAILED: {e}\n{traceback.format_exc()}")
        log(f"--- train Mask R-CNN on {task}")
        try:
            if mr.exists() and not retrain:
                log(f"  using existing {mr.name} (SEMCHAR_RETRAIN=1 retrains)")
            else:
                M.train([(i, l) for _, i, l, _ in D[task]["train"]], mr, iters=iters, log=log)
            trained[f"mrcnn_{task}"] = {"kind": "maskrcnn", "path": f"models/{mr.name}", "name": f"mrcnn_{task}"}
        except Exception as e:
            log(f"  FAILED: {e}\n{traceback.format_exc()}")

    log("--- benchmark on the published validation images")
    from .qc import _load
    contenders = {"particles": [("cpsam", {"kind": "zero", "name": "cpsam"})] +
                  [(k, v) for k, v in trained.items() if k.endswith("particles")],
                  "satellites": [(k, v) for k, v in trained.items() if k.endswith("satellites")]}
    rows = []
    for task, cands in contenders.items():
        min_area = 20 if task == "particles" else 8
        for name, spec in cands:
            try:
                seg = make("cpsam") if spec["kind"] == "zero" else _load(spec, satellites=task == "satellites")
                res, per = score_items(seg, D[task]["val"], min_area, log)
                seg.close()
                per.to_csv(run_dir / f"bench_{task}_{name}.csv", index=False)
                for src, r in res.items():
                    rows.append({"task": task, "model": name, "subset": src, **r})
                a = res["all"]
                log(f"  {task:10s} {name:17s} P={a['P@0.5']:.3f} R={a['R@0.5']:.3f} F1={a['F1@0.5']:.3f} "
                    f"pixelP={a['pixel_P']:.3f} pixelR={a['pixel_R']:.3f} {a['sec_per_image']:.1f}s/img")
            except Exception as e:
                log(f"  {task} {name} FAILED: {e}\n{traceback.format_exc()}")
    if "mrcnn_satellites" in trained:
        log("--- tune Mask R-CNN satellites (inference scale x score threshold)")
        try:
            spec = trained["mrcnn_satellites"]
            sc, th, held = tune_maskrcnn(spec, D["satellites"]["val"], 8, run_dir, log)
            name = "mrcnn_satellites_tuned"
            held.to_csv(run_dir / f"bench_satellites_{name}.csv", index=False)
            for src, r in by_source(held).items():
                rows.append({"task": "satellites", "model": name, "subset": src, **r})
            a = agg(held)
            log(f"  satellites {name:17s} P={a['P@0.5']:.3f} R={a['R@0.5']:.3f} F1={a['F1@0.5']:.3f} "
                f"count ratio {a['count_ratio']:.2f} (leave-one-image-out)")
            trained[name] = {**spec, "name": name, "score_thresh": th, "scale": sc}
        except Exception as e:
            log(f"  tuning FAILED: {e}\n{traceback.format_exc()}")
    bench = pd.DataFrame(rows)
    bench.to_csv(run_dir / "benchmark.csv", index=False)

    choice = {}
    for task in ("particles", "satellites"):
        b = bench[(bench.task == task) & (bench.subset == "all")] if len(bench) else bench
        b = b[b.model.isin(trained)] if task == "particles" else b   # QC needs a trained particle model
        if len(b) and task == "satellites":
            # QC reports satellite counts, so count bias matters: among models within 0.02 F1 of the
            # best, take the one whose predicted count is closest to the hand count
            near = b[b["F1@0.5"] >= b["F1@0.5"].max() - 0.02]
            best = near.loc[(np.log(near.count_ratio.clip(lower=1e-3))).abs().idxmin(), "model"]
            choice[task] = trained[best]
            log(f"  best {task} model: {best}")
        elif len(b):
            best = b.sort_values("F1@0.5", ascending=False).iloc[0]["model"]
            choice[task] = trained[best]
            log(f"  best {task} model: {best}")
    if len(choice) == 2:
        (MODELS / "qc_models.json").write_text(json.dumps(choice, indent=1))

    log("--- % satellited particles vs hand labels (AMPIS image with both label sets)")
    sat_check = pd.DataFrame()
    try:
        from .qc import image_particles
        both = sorted({n for n, *_ in D["particles"]["val"]} & {n for n, *_ in D["satellites"]["val"]})
        rows = []
        for n in both:
            _, img, gp, _ = next(x for x in D["particles"]["val"] if x[0] == n)
            _, _, gs, _ = next(x for x in D["satellites"]["val"] if x[0] == n)
            g = image_particles(img, gp, gs, None)
            rows.append({"image": n, "who": "hand labels", "particles": len(g),
                         "pct_satellited": 100 * g.satellited.mean(), "satellites_per_particle": g.n_satellites.mean()})
            if len(choice) == 2:
                ps = _load(choice["particles"])
                part = ps.segment(img)
                ps.close()
                sats = [choice["satellites"]]
                if choice["satellites"]["name"].endswith("_tuned") and "mrcnn_satellites" in trained:
                    sats.append(trained["mrcnn_satellites"])
                for sspec in sats:
                    ss = _load(sspec, satellites=True)
                    p = image_particles(img, part, ss.segment(img), None)
                    ss.close()
                    rows.append({"image": n, "who": f"{choice['particles']['name']} + {sspec['name']}",
                                 "particles": len(p), "pct_satellited": 100 * p.satellited.mean(),
                                 "satellites_per_particle": p.n_satellites.mean()})
        sat_check = pd.DataFrame(rows)
        sat_check.to_csv(run_dir / "satellited_check.csv", index=False)
        log("\n" + sat_check.round(3).to_string(index=False))
    except Exception as e:
        log(f"  FAILED: {e}\n{traceback.format_exc()}")
    return bench, choice, sat_check


def run_real_qc(run_dir, log=print):
    """QC on the real test samples + an example comparison."""
    from .qc import compare, run as qc_run
    from .refs import DATA, REAL_SAMPLES
    items = [(k, DATA / v[0]) for k, v in REAL_SAMPLES.items() if (DATA / v[0]).exists()]
    samples, boots, points = qc_run(items, Path(run_dir) / "qc_real", log=log)
    ex = None
    if "AA7075_base" in boots and "AA7075_TiC_surface" in boots:
        ex = compare(boots["AA7075_base"], boots["AA7075_TiC_surface"], points["AA7075_base"], points["AA7075_TiC_surface"])
        ex.to_csv(Path(run_dir) / "qc_real" / "compare_example_AA7075_base_vs_TiC_surface.csv", index=False)
    return samples, ex

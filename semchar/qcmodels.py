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
Caveat: validation images also pick the fine-tuning epoch, so those scores are slightly optimistic.
"""
from __future__ import annotations

import json
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

    def agg(d):
        P = d.tp.sum() / max(1, d.n_pred.sum())
        R = d.tp.sum() / max(1, d.n_gt.sum())
        return {"n_images": len(d), "n_gt": int(d.n_gt.sum()), "P@0.5": P, "R@0.5": R,
                "F1@0.5": 2 * P * R / max(1e-9, P + R), "mean_iou": float(d.mean_iou.mean()),
                "pixel_P": d.pix_tp.sum() / max(1, d.pix_pred.sum()), "pixel_R": d.pix_tp.sum() / max(1, d.pix_gt.sum()),
                "sec_per_image": float(d.seconds.mean())}
    out = {"all": agg(df)}
    for s, d in df.groupby("source"):
        out[s] = agg(d)
    return out, df


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
    trained = {}

    def as_lists(items):
        return ([i for _, i, _, _ in items], [l for _, _, l, _ in items], [n for n, _, _, _ in items])

    for task in ("particles", "satellites"):
        log(f"--- train Cellpose-SAM on {task}")
        try:
            train_cellpose(as_lists(D[task]["train"]), as_lists(D[task]["val"]), MODELS / f"cp_{task}",
                           log=log, n_epochs=ep, save_every=every, nimg_per_epoch=16)
            trained[f"cp_{task}"] = {"kind": "cellpose", "path": f"models/cp_{task}", "name": f"cp_{task}"}
        except Exception as e:
            log(f"  FAILED: {e}\n{traceback.format_exc()}")
        log(f"--- train Mask R-CNN on {task}")
        try:
            M.train([(i, l) for _, i, l, _ in D[task]["train"]], MODELS / f"mrcnn_{task}.pt", iters=iters, log=log)
            trained[f"mrcnn_{task}"] = {"kind": "maskrcnn", "path": f"models/mrcnn_{task}.pt", "name": f"mrcnn_{task}"}
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
    bench = pd.DataFrame(rows)
    bench.to_csv(run_dir / "benchmark.csv", index=False)

    choice = {}
    for task in ("particles", "satellites"):
        b = bench[(bench.task == task) & (bench.subset == "all")] if len(bench) else bench
        b = b[b.model.isin(trained)] if task == "particles" else b   # QC needs a trained particle model
        if len(b):
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
                ps, ss = _load(choice["particles"]), _load(choice["satellites"], satellites=True)
                p = image_particles(img, ps.segment(img), ss.segment(img), None)
                ps.close()
                ss.close()
                rows.append({"image": n, "who": f"{choice['particles']['name']} + {choice['satellites']['name']}",
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

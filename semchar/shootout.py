"""Zero-shot model shoot-out.

For each model:
  1. DeCost & Holm renders (sparse spheres, exact visible-instance masks by ray casting)
  2. Dense monolayer renders (data/synth_dense; parents + fused satellites + fines, exact masks)
  3. All real samples characterized and compared with laser diffraction (D10/D50/D90, volume,
     edge-corrected). Laser and image sample different things (few hundred particles in one
     frame vs millions dispersed), so this is a sanity check, weighted low in the ranking.
Ranking score = 0.5 F1@0.5(dense) + 0.25 F1@0.5(DeCost) + 0.25 (1 - min(1, mean |D50 err| real)).

Outputs (out_dir): synth_<set>_<model>.csv per-image scores, shootout.csv pooled,
real/<model>/... (characterize outputs), best_model.txt.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import classic
from .characterize import run as characterize
from .evaluate import pool, score
from .imio import load_gray
from .models import make
from .refs import DATA, REAL_SAMPLES, reference

GT_CACHE = DATA / "_gt_cache"


def decost_items(n_per_psd):
    from .decost_gt import ground_truth
    items = []
    GT_CACHE.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in (DATA / "synthetic").iterdir() if p.is_dir() and not p.name.startswith("_")):
        for k in range(1, n_per_psd + 1):
            img = d / "renders" / f"particles{k}.png"
            js = d / "particles" / f"particles{k}.json"
            if not img.exists() or not js.exists():
                continue
            cache = GT_CACHE / f"decost_{d.name[0]}_{k:03d}.npy"
            if not cache.exists():
                try:
                    _, ids = ground_truth(js)
                except ValueError:   # a/particles37.json is truncated upstream
                    continue
                np.save(cache, classic.postprocess(np.where(ids >= 0, ids + 1, 0).astype(np.int32)))
            items.append((f"{d.name[0]}_{k:03d}", img, cache))
    return items


def dense_items(n):
    items = []
    for k in range(n):
        img = DATA / "synth_dense" / f"sat_{k:03d}.png"
        ids = DATA / "synth_dense" / f"sat_{k:03d}_ids.npy"
        if img.exists() and ids.exists():
            cache = GT_CACHE / f"dense_{k:03d}.npy"
            if not cache.exists():
                GT_CACHE.mkdir(parents=True, exist_ok=True)
                raw = np.load(ids)
                np.save(cache, classic.postprocess(np.where(raw >= 0, raw + 1, 0).astype(np.int32)))
            items.append((f"dense_{k:03d}", img, cache))
    return items


def eval_set(seg, items, budget_s, log):
    rows, t0 = [], time.time()
    for i, (key, img_p, gt_p) in enumerate(items):
        if time.time() - t0 > budget_s:
            log(f"    time budget reached after {i} images")
            break
        img = load_gray(img_p)
        pr = seg(img)
        s = score(np.load(gt_p), pr)
        s.update({"image": key, "seconds": seg.last_seconds})
        rows.append(s)
    return pd.DataFrame(rows)


def run(models, out_dir, n_decost=12, n_dense=40, budget_s=2400, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {"decost": decost_items(n_decost), "dense": dense_items(n_dense)}
    log(f"shoot-out: {len(sets['decost'])} DeCost + {len(sets['dense'])} dense synthetic images, "
        f"{len(REAL_SAMPLES)} real samples, models {models}")
    refs = {k: reference(k) for k in REAL_SAMPLES if (DATA / REAL_SAMPLES[k][0]).exists()}
    real_items = [(k, DATA / REAL_SAMPLES[k][0]) for k in refs]
    pooled = []
    prev = {}
    if (out / "shootout.csv").exists():   # resuming: keep models that finished cleanly
        try:
            for r in pd.read_csv(out / "shootout.csv").to_dict("records"):
                done = (r.get("error") != r.get("error") or not r.get("error")) and (out / "real" / r["model"] / "samples.csv").exists()
                if done:
                    prev[r["model"]] = r
        except Exception:
            prev = {}
    for name in models:
        if name in prev:
            log(f"== model {name}: already done in this run, reusing")
            pooled.append(prev[name])
            continue
        log(f"== model {name}")
        try:
            seg = make(name)
        except Exception as e:
            log(f"  could not load {name}: {e}")
            pooled.append({"model": name, "error": f"load: {e}"})
            continue
        row = {"model": name}
        try:
            for set_name, items in sets.items():
                if not items:
                    continue
                df = eval_set(seg, items, budget_s, log)
                df.to_csv(out / f"synth_{set_name}_{name}.csv", index=False)
                p = pool(df.to_dict("records"))
                row.update({f"{set_name}_{k}": v for k, v in p.items()})
                log(f"  {set_name}: F1@0.5={p['F1@0.5']:.3f} P={p['P@0.5']:.3f} R={p['R@0.5']:.3f} "
                    f"IoU={p['mean_iou_matched']:.3f} D50num err={p['D50_num_err_mean']:+.1%} "
                    f"vol={p['D50_vol_err_mean']:+.1%} {p.get('sec_per_image', float('nan')):.1f}s/img")
            samples, _ = characterize(real_items, seg, out / "real" / name, refs=refs,
                                      log=lambda m: log("  " + m))
            if len(samples) and "err_D50_vol_ec" in samples:
                row["real_D50_abs_err"] = float(samples.err_D50_vol_ec.abs().mean())
                row["real_D90_abs_err"] = float(samples.err_D90_vol_ec.abs().mean())
                log(f"  real vs laser: mean |D50 err| {row['real_D50_abs_err']:.1%}, |D90 err| {row['real_D90_abs_err']:.1%}")
        except Exception as e:
            log(f"  {name} FAILED: {e}\n{traceback.format_exc()}")
            row["error"] = str(e)
        finally:
            seg.close()
        pooled.append(row)
        pd.DataFrame(pooled).to_csv(out / "shootout.csv", index=False)  # save as we go
    df = pd.DataFrame(pooled)

    def comp(r):
        f_dense = r.get("dense_F1@0.5", np.nan)
        f_dec = r.get("decost_F1@0.5", np.nan)
        e = r.get("real_D50_abs_err", np.nan)
        if any(v != v for v in (f_dense, f_dec, e)):
            return np.nan
        return 0.5 * f_dense + 0.25 * f_dec + 0.25 * (1 - min(1.0, e))
    df["score"] = [comp(r) for r in df.to_dict("records")]
    df = df.sort_values("score", ascending=False)
    df.to_csv(out / "shootout.csv", index=False)
    ok = df.dropna(subset=["score"])
    best = ok.iloc[0]["model"] if len(ok) else "cpsam"
    (out / "best_model.txt").write_text(best)
    json.dump({"best": best, "ranking": ok[["model", "score"]].to_dict("records")},
              open(out / "best_model.json", "w"), indent=1)
    log(f"best zero-shot model: {best}")
    return df, best

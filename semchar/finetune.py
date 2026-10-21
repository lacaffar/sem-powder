"""Visit 2: fine-tune Cellpose-SAM on the corrected crops, pick the epoch on the
validation crops, score everything on the held-out test crops.

GPU memory (Quadro RTX 4000, 8 GB): full fine-tuning of the ViT-L encoder in
float32 with AdamW needs ~5 GB for weights + optimizer state. Attempts in order,
each falling back on out-of-memory:
  1. batch 2, all layers   2. batch 1, all layers
  3. batch 1, first 12 of 24 encoder blocks frozen   4. batch 1, first 20 frozen
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import classic
from .evaluate import pool, score
from .labeling import load_split
from .models import ROOT, CellposeSeg, make

MODELS = ROOT / "models"
FINAL = MODELS / "cpsam_powder"
ATTEMPTS = [(2, 0), (1, 0), (1, 12), (1, 20)]


def _freeze(net, n_blocks):
    enc = net.encoder
    for p in enc.parameters():
        p.requires_grad = True
    if n_blocks <= 0:
        return
    for mod in (getattr(enc, "patch_embed", None),):
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = False
    if getattr(enc, "pos_embed", None) is not None:
        enc.pos_embed.requires_grad = False
    for blk in enc.blocks[:n_blocks]:
        for p in blk.parameters():
            p.requires_grad = False


def eval_masks(seg, imgs, masks, names):
    rows = []
    for img, gt, n in zip(imgs, masks, names):
        pr = seg(img)
        s = score(classic.postprocess(gt), pr)
        s.update({"crop": n, "seconds": seg.last_seconds})
        rows.append(s)
    return rows


def train(log=print, n_epochs=None, save_every=None, lr=1e-5, attempts=None):
    import os
    import subprocess
    import sys
    n_epochs = n_epochs or int(os.environ.get("SEMCHAR_FT_EPOCHS", 200))        # env overrides: quick tests
    save_every = save_every or int(os.environ.get("SEMCHAR_FT_SAVE_EVERY", 25))
    from . import labeling as L
    tr_i, tr_m, tr_n = load_split("train")
    va_i, va_m, va_n = load_split("val")
    log(f"fine-tune: {len(tr_i)} train crops, {len(va_i)} val crops (reviewed only)")
    if len(tr_i) < 4:
        raise RuntimeError(f"only {len(tr_i)} reviewed training crops; review at least 4 (ideally all) first")
    run_dir = MODELS / "runs" / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    cfg = None
    for bs, frozen in (attempts or ATTEMPTS):
        log(f"  attempt: batch {bs}, {frozen} of 24 encoder blocks frozen")
        c = {"batch_size": bs, "frozen": frozen, "n_epochs": n_epochs, "save_every": save_every, "lr": lr,
             "run_dir": str(run_dir), "result": str(run_dir / f"result_b{bs}_f{frozen}.json"),
             "labels_dir": str(L.LABELS)}
        cp = run_dir / f"cfg_b{bs}_f{frozen}.json"
        cp.write_text(json.dumps(c))
        for old in (run_dir / "models").glob("cpsam_powder*"):   # partial files from a failed attempt
            old.unlink()
        free_gb = shutil.disk_usage(run_dir).free / 2**30
        need_gb = 1.3 * (n_epochs // save_every + 1)
        if free_gb < need_gb:
            log(f"  WARNING: {free_gb:.1f} GB free, checkpoints need ~{need_gb:.0f} GB")
        proc = subprocess.Popen([sys.executable, "-m", "semchar._ft_worker", str(cp)], cwd=str(ROOT),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        oom = False
        limit_s = float(os.environ.get("SEMCHAR_FT_TIMEOUT_H", 4)) * 3600   # watchdog per attempt
        import threading
        timer = threading.Timer(limit_s, proc.kill)
        timer.start()
        for line in proc.stdout:
            line = line.rstrip()
            if "OOM" in line or "out of memory" in line.lower():
                oom = True
            if line and "%|" not in line:
                log("    " + line[:200])
        proc.wait()
        timed_out = not timer.is_alive() and proc.returncode != 0
        timer.cancel()
        if timed_out:
            log(f"  attempt exceeded {limit_s / 3600:.1f} h and was stopped")
        res = Path(c["result"])
        if proc.returncode == 0 and res.exists():
            r = json.loads(res.read_text())
            log(f"  trained in {r['minutes']:.1f} min, peak GPU memory {r['peak_vram_gb']:.1f} GB, "
                f"{r['trainable_params_m']:.0f}M trainable parameters")
            cfg = {**c, **r, "train_crops": tr_n, "val_crops": va_n}
            break
        log(f"  attempt failed (exit {proc.returncode}{', out of GPU memory' if oom else ''}); trying a lighter setting")
    if cfg is None:
        raise RuntimeError("fine-tuning failed for every memory setting; see log")
    # choose the checkpoint on the validation crops
    ckpts = sorted((run_dir / "models").glob("cpsam_powder_epoch_*")) + [run_dir / "models" / "cpsam_powder"]
    eval_i, eval_m, eval_n = (va_i, va_m, va_n) if va_i else (tr_i, tr_m, tr_n)
    rows = []
    for ck in ckpts:
        seg = CellposeSeg(ck, "native", name=ck.name)
        p = pool(eval_masks(seg, eval_i, eval_m, eval_n))
        seg.close()
        rows.append({"checkpoint": ck.name, **p})
        log(f"  {ck.name}: {'val' if va_i else 'train'} F1@0.5={p['F1@0.5']:.3f} IoU={p['mean_iou_matched']:.3f}")
    sel = pd.DataFrame(rows).sort_values("F1@0.5", ascending=False)
    sel.to_csv(run_dir / "checkpoint_selection.csv", index=False)
    best = run_dir / "models" / sel.iloc[0]["checkpoint"]
    MODELS.mkdir(exist_ok=True)
    shutil.copy(best, FINAL)
    cfg["chosen_checkpoint"] = best.name
    (MODELS / "cpsam_powder.json").write_text(json.dumps(cfg, indent=1))
    for ck in ckpts:   # 1.2 GB each; the chosen one is copied to models/cpsam_powder
        ck.unlink(missing_ok=True)
    log(f"  chosen {best.name} -> {FINAL}")
    return sel


def test_table(models_to_test, log=print):
    """Score each model on the held-out test crops (user-corrected)."""
    te_i, te_m, te_n = load_split("test")
    if not te_i:
        log("no reviewed test crops; skipping test scoring")
        return pd.DataFrame()
    rows = []
    for name in models_to_test:
        try:
            seg = make(name)
        except Exception as e:
            log(f"  {name}: could not load ({e})")
            continue
        r = eval_masks(seg, te_i, te_m, te_n)
        seg.close()
        p = pool(r)
        rows.append({"model": name, **p})
        log(f"  test {name}: F1@0.5={p['F1@0.5']:.3f} P={p['P@0.5']:.3f} R={p['R@0.5']:.3f} "
            f"IoU={p['mean_iou_matched']:.3f} D50num err={p['D50_num_err_mean']:+.1%} vol={p['D50_vol_err_mean']:+.1%}")
    return pd.DataFrame(rows)

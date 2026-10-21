"""Label set for fine-tuning: crops of the real images with draft outlines the user corrects.

Crops: each real image (bar removed) split into 4 quadrants -> labels/<split>/<sample>_q<k>.png
with a Cellpose-format draft labels/<split>/<sample>_q<k>_seg.npy (open in the Cellpose
GUI, fix, save with Ctrl+S). Drafts come from the best Cellpose model run on the FULL
image (more context than a crop), then cut into crops together with its flow fields.

Split by source image so no test image is seen in training:
  test  : AA7075_TiC_surface (Fig 2c), Ti64_15Cu   (same held-out images as attempt 1)
  val   : AA7075_TiC_volume (Fig 2d), Ti64_20Cu
  train : everything else (Cu crops are dense, ~200-330 particles each: label them last)
User-added images (CAMAL) go to train unless named in configs/label_split.json.
labels/manifest.json records each crop's source, box, split and draft hash; a crop
counts as reviewed once its _seg.npy differs from the draft or is newer than it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .imio import load_gray
from .refs import DATA, REAL_SAMPLES
from .scale import calibrate

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels"
SPLIT = {"AA7075_TiC_surface": "test", "Ti64_15Cu": "test", "AA7075_TiC_volume": "val", "Ti64_20Cu": "val"}


def _hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label_sources():
    srcs = [(k, DATA / v[0]) for k, v in REAL_SAMPLES.items() if (DATA / v[0]).exists()]
    extra = DATA / "label_extra"   # optional: user images to label (e.g. CAMAL), any layout
    if extra.exists():
        from .characterize import gather
        srcs += [(f"{s}_{p.stem}", p) for s, p in gather(extra)]
    cfg = ROOT / "configs" / "label_split.json"
    split = dict(SPLIT)
    if cfg.exists():
        split.update(json.loads(cfg.read_text()))
    return srcs, split


def quadrants(h, w):
    hh, hw = h // 2, w // 2
    return [(0, hh, 0, hw), (0, hh, hw, w), (hh, h, 0, hw), (hh, h, hw, w)]


def make_drafts(seg, log=print):
    """seg: models.CellposeSeg (needs .model with flows)."""
    import cv2
    from cellpose import io as cpio
    srcs, split = label_sources()
    man_p = LABELS / "manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else {}
    for name, path in srcs:
        img = load_gray(path)
        y0, y1, x0, x1 = calibrate(path, img=img, use_meta=False).crop_box
        img = img[y0:y1, x0:x1]
        sp = split.get(name, "train")
        todo = []
        for k, (a, b, c, d) in enumerate(quadrants(*img.shape)):
            crop = f"{name}_q{k}"
            seg_p = LABELS / sp / f"{crop}_seg.npy"
            if seg_p.exists():   # never overwrite a draft the user may have corrected
                continue
            todo.append((k, crop, (a, b, c, d)))
        if not todo:
            continue
        masks, flows, _ = seg.model.eval(img, batch_size=seg.batch_size, diameter=seg.diameter_for(img))
        masks = np.asarray(masks)
        for k, crop, (a, b, c, d) in todo:
            (LABELS / sp).mkdir(parents=True, exist_ok=True)
            img_p = LABELS / sp / f"{crop}.png"
            cv2.imwrite(str(img_p), img[a:b, c:d])
            fl = [flows[0][a:b, c:d], flows[1][:, a:b, c:d], flows[2][a:b, c:d]]
            m = masks[a:b, c:d]
            from skimage.segmentation import relabel_sequential
            m = relabel_sequential(m)[0]
            cpio.masks_flows_to_seg(img[a:b, c:d], m, fl, str(img_p))
            seg_p = LABELS / sp / f"{crop}_seg.npy"
            manifest[crop] = {"source": name, "path": str(Path(path).relative_to(ROOT)) if ROOT in Path(path).parents else str(path),
                              "box_yx": [int(a + y0), int(b + y0), int(c + x0), int(d + x0)], "split": sp,
                              "draft_sha256": _hash(seg_p), "draft_mtime": os.path.getmtime(seg_p),
                              "draft_model": seg.name, "n_draft": int(m.max())}
            log(f"  draft {sp}/{crop}: {int(m.max())} particles")
    LABELS.mkdir(exist_ok=True)
    man_p.write_text(json.dumps(manifest, indent=1))
    return manifest


def status():
    """-> DataFrame: crop, split, reviewed, n_masks."""
    import pandas as pd
    man_p = LABELS / "manifest.json"
    if not man_p.exists():
        return pd.DataFrame(columns=["crop", "split", "reviewed", "n_masks"])
    man = json.loads(man_p.read_text())
    rows = []
    for crop, m in man.items():
        p = LABELS / m["split"] / f"{crop}_seg.npy"
        if not p.exists():
            continue
        reviewed = _hash(p) != m["draft_sha256"] or os.path.getmtime(p) > m["draft_mtime"] + 1
        n = int(np.load(p, allow_pickle=True).item()["masks"].max())
        rows.append({"crop": crop, "split": m["split"], "reviewed": reviewed, "n_masks": n})
    return pd.DataFrame(rows)


def load_split(split, reviewed_only=True):
    """-> (images, masks, names) for a split."""
    st = status()
    imgs, masks, names = [], [], []
    for r in st[st.split == split].itertuples():
        if reviewed_only and not r.reviewed:
            continue
        dat = np.load(LABELS / split / f"{r.crop}_seg.npy", allow_pickle=True).item()
        imgs.append(load_gray(LABELS / split / f"{r.crop}.png"))
        masks.append(np.asarray(dat["masks"]).astype(np.int32))
        names.append(r.crop)
    return imgs, masks, names


def open_gui():
    """Open the Cellpose GUI on the first crop not yet reviewed (train, then val, then test)."""
    st = status()
    if not len(st):
        print("No drafts yet. Run visit 1 first.")
        return
    order = {"train": 0, "val": 1, "test": 2}
    todo = st[~st.reviewed].assign(o=st.split.map(order), cu=st.crop.str.startswith("Cu_"))
    todo = todo.sort_values(["o", "cu", "crop"])
    done = int(st.reviewed.sum())
    print(f"Reviewed {done}/{len(st)} crops.")
    if not len(todo):
        print("All crops reviewed. Next: run visit 2.")
        return
    first = todo.iloc[0]
    img = LABELS / first.split / f"{first.crop}.png"
    print(f"Opening {img} ... fix outlines, Ctrl+S to save, then press D for the next crop.")
    from . import models  # noqa: F401  (model cache location, in case the GUI's own segmentation is used)
    from cellpose.gui import gui
    gui.run(image=str(img))

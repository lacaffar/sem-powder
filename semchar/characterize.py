"""Batch characterization: folder of SEM images -> per-image and per-sample results.

Input layout: each subfolder = one sample (all its images pooled, which is what
makes D90 trustworthy). Images directly in the input folder form one sample named
after the folder. Scale per image: --um-per-px > scales.csv next to the images >
SEM TIFF metadata > scale-bar OCR > uncalibrated (px, flagged).

Outputs in out_dir:
  samples.csv                 one row per sample (pooled)
  images.csv                  one row per image
  particles/<sample>/<image>.csv
  overlays/<sample>/<image>.jpg
  reports/<sample>.html       size distribution, shape histograms, overlays
  index.html
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import pandas as pd

from . import classic
from .imio import IMG_EXTS, load_gray
from .measure import particle_table, summarize
from .report import sample_report, simple_page, table_html
from .scale import calibrate


def gather(input_path):
    """-> list of (sample, image_path)."""
    p = Path(input_path)
    if p.is_file():
        return [(p.parent.name, p)]
    items = []
    for f in sorted(p.iterdir()):
        if f.is_dir():
            items += [(f.name, g) for g in sorted(f.iterdir()) if g.suffix.lower() in IMG_EXTS]
        elif f.suffix.lower() in IMG_EXTS:
            items.append((p.name, f))
    return items


def run(items, seg, out_dir, um_per_px=None, refs=None, log=print):
    """items: list of (sample, path). seg: models.Segmenter. refs: {sample: {D10,D50,D90}}."""
    out = Path(out_dir)
    for sub in ("particles", "overlays", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    img_rows, per_sample, overlays = [], {}, {}
    for i, (sample, path) in enumerate(items, 1):
        t0 = time.time()
        try:
            img = load_gray(path)
            sc = calibrate(path, img=img, um_per_px=um_per_px)
            y0, y1, x0, x1 = sc.crop_box
            img = img[y0:y1, x0:x1]
            lab = seg(img)
            df = particle_table(lab, sc.um_per_px)
        except Exception as e:  # one bad image must not stop a batch
            log(f"  [{i}/{len(items)}] {sample}/{path.name}: FAILED {e}")
            img_rows.append({"sample": sample, "image": path.name, "error": str(e)})
            continue
        df.insert(0, "image", path.name)
        (out / "particles" / sample).mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "particles" / sample / f"{path.stem}.csv", index=False)
        s = summarize(df, sc.um_per_px is not None)
        s.update({"sample": sample, "image": path.name, "um_per_px": sc.um_per_px, "scale_source": sc.source,
                  "model": seg.name, "seconds": round(time.time() - t0, 2)})
        img_rows.append(s)
        ov = classic.overlay(img, lab, title=f"{path.name}  {seg.name}  n={len(df)}")
        (out / "overlays" / sample).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "overlays" / sample / f"{path.stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
        per_sample.setdefault(sample, []).append((df, sc.um_per_px is not None))
        overlays.setdefault(sample, []).append((path.name, ov))
        log(f"  [{i}/{len(items)}] {sample}/{path.name}: n={len(df)} D50vol={s['D50_vol_ec']:.1f} "
            f"{s['size_unit']} scale={sc.source} {s['seconds']}s")
    images = pd.DataFrame(img_rows)
    images.to_csv(out / "images.csv", index=False)
    srows = []
    for sample, parts in per_sample.items():
        calibrated = all(c for _, c in parts)
        df = pd.concat([d for d, _ in parts], ignore_index=True)
        s = summarize(df, calibrated)
        if not calibrated and any(c for _, c in parts):
            s["flags"] = (s["flags"] + "; " if s["flags"] else "") + "mixed calibrated/uncalibrated images"
        s = {"sample": sample, "n_images": len(parts), "model": seg.name, **s}
        ref = (refs or {}).get(sample)
        if ref:
            for k in ("D10", "D50", "D90"):
                s[f"ref_{k}"] = ref[k]
                s[f"err_{k}_vol_ec"] = s[f"{k}_vol_ec"] / ref[k] - 1
        srows.append(s)
        sample_report(out / "reports" / f"{sample}.html", sample, s, df, overlays[sample], ref, seg.name)
    samples = pd.DataFrame(srows)
    samples.to_csv(out / "samples.csv", index=False)
    if len(samples):
        cols = [c for c in ["sample", "n_images", "count", "size_unit", "D10_vol_ec", "D50_vol_ec", "D90_vol_ec",
                            "ref_D50", "err_D50_vol_ec", "circularity_median", "feret_ar_median", "pct_spherical",
                            "pct_with_satellite_like", "flags"] if c in samples]
        links = "".join(f"<li><a href='reports/{s}.html'>{s}</a></li>" for s in samples["sample"])
        simple_page(out / "index.html", f"SEM powder characterization ({seg.name})",
                    [("Samples", table_html(samples[cols])), ("Reports", f"<ul>{links}</ul>")])
    return samples, images

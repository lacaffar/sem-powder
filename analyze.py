"""SEM powder characterization: particle size and shape from SEM images.

  python analyze.py <image_or_folder> [--method watershed|cellpose|cellpose_ws]
                    [--um-per-px X] [--scales scales.csv] [--out out/]

Outputs in --out:
  results.csv            one row per image (count, D10/D50/D90, shape means, scale source)
  particles/<image>.csv  one row per particle
  psd/<image>.png        histogram + number/volume CDF
  overlays/<image>.png   segmentation overlay
  scale_debug/<image>.png detected data bar / scale bar / OCR label
  psd_all.png            combined CDFs across images
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

COLS = ["image", "method", "count", "size_unit", "um_per_px", "scale_source", "scale_confidence",
        "D10_num", "D50_num", "D90_num", "D10_vol", "D50_vol", "D90_vol",
        "D10_vol_ec", "D50_vol_ec", "D90_vol_ec",
        "mean_circularity", "mean_aspect_ratio", "mean_solidity", "irregular_frac", "seconds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder of images (.tif .png .jpg .bmp)")
    ap.add_argument("--method", default="watershed", choices=["watershed", "cellpose", "cellpose_ws"])
    ap.add_argument("--um-per-px", type=float, default=None, help="force scale for all images (µm per pixel)")
    ap.add_argument("--scales", default=None, help="CSV with rows: filename,um_per_px (default: <folder>/scales.csv)")
    ap.add_argument("--out", default="out", help="output folder")
    a = ap.parse_args()

    import cv2
    import pandas as pd
    from src.imio import list_images
    from src.metrics import analyze_image, plot_psd
    from src.segment import overlay

    imgs = list_images(a.input)
    if not imgs:
        sys.exit(f"no images found in {a.input}")
    out = Path(a.out)
    for sub in ("particles", "psd", "overlays", "scale_debug"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows, dfs, names = [], [], []
    t0 = time.time()
    for i, p in enumerate(imgs, 1):
        try:
            df, s, lab, img, sc = analyze_image(p, a.method, um_per_px=a.um_per_px, sidecar=a.scales,
                                                scale_debug=out / "scale_debug" / f"{p.stem}.png")
        except Exception as e:  # keep going on bad files
            print(f"[{i}/{len(imgs)}] {p.name}: FAILED ({e})")
            rows.append({"image": p.name, "method": a.method, "count": 0, "scale_source": f"error: {e}"})
            continue
        unit = s["size_unit"]
        df.to_csv(out / "particles" / f"{p.stem}.csv", index=False)
        cv2.imwrite(str(out / "overlays" / f"{p.stem}.png"),
                    overlay(img, lab, title=f"{p.name} {a.method} n={s['count']}"))
        if len(df):
            plot_psd(df, out / "psd" / f"{p.stem}.png", unit=unit,
                     title=f"{p.name}: n={s['count']}, D50 vol {s['D50_vol']:.1f} {unit}")
            dfs.append(df)
            names.append(p.stem)
        rows.append(s)
        flag = "" if unit == "um" else "  [UNCALIBRATED: px]"
        print(f"[{i}/{len(imgs)}] {p.name}: n={s['count']} D50 num/vol = "
              f"{s['D50_num']:.1f}/{s['D50_vol']:.1f} {unit} (scale: {s['scale_source']}){flag}")

    res = pd.DataFrame(rows)
    res = res[[c for c in COLS if c in res.columns]]
    res.to_csv(out / "results.csv", index=False, float_format="%.4g")
    if dfs:
        units = set(res["size_unit"].dropna())
        unit = "um" if units == {"um"} else "px"
        plot_psd(dfs, out / "psd_all.png", unit=unit, labels=names, title="All images")
    print(f"done: {len(imgs)} images in {time.time() - t0:.0f} s -> {out.resolve()}")


if __name__ == "__main__":
    main()

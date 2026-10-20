"""Block 3 check: overlays for 2 synthetic + 2 real images, every method."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imio import load_gray  # noqa: E402
from src.scale import calibrate  # noqa: E402
from src.segment import segment, overlay  # noqa: E402

IMGS = [
    ROOT / "data/synthetic/a-lognormal-loc0.1-shape0.5/renders/particles1.png",
    ROOT / "data/synthetic/g-weibull_fit-loc0.12-shape0.5/renders/particles1.png",
    ROOT / "data/aa7075/Original_Image_for_Fig_2a.tif",
    ROOT / "data/aa7075/Original_Image_for_Fig_2d.tif",
]
METHODS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["watershed", "cellpose", "cellpose_ws"]
OUT = ROOT / "out/overlays"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in IMGS:
        img = load_gray(p)
        sc = calibrate(p, img=img)
        y0, y1, x0, x1 = sc.crop_box
        img = img[y0:y1, x0:x1]
        tag = p.parent.parent.name[:1] + "_" + p.stem if "synthetic" in str(p) else p.stem
        for m in METHODS:
            lab, info = segment(img, m)
            ecd = 2 * np.sqrt(np.bincount(lab.ravel())[1:] / np.pi)
            print(f"{tag:34s} {m:12s} n={info['n']:4d} medECD={np.median(ecd) if len(ecd) else 0:6.1f}px "
                  f"t={info['seconds']:.1f}s {({k: v for k, v in info.items() if k not in ('n', 'seconds')})}")
            cv2.imwrite(str(OUT / f"{tag}__{m}.png"), overlay(img, lab, title=f"{m} n={info['n']}"))


if __name__ == "__main__":
    main()

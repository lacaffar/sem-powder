"""figures/fig1_overlays.png: raw | watershed | Cellpose, synthetic row + real row."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imio import load_gray  # noqa: E402
from src.segment import overlay, segment  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [
        ("Synthetic (PSD e, render 1), 256x256 crop",
         load_gray(ROOT / "data/synthetic/e-lognormal-loc0.12-shape0.5/renders/particles1.png"), (0, 256, 0, 256)),
        ("AA7075 base powder (Fig 2a), 300x, 0.372 µm/px",
         load_gray(ROOT / "data/aa7075/Original_Image_for_Fig_2a.tif"), None),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.4))
    for r, (name, img, crop) in enumerate(rows):
        panels = [("raw", img, None)]
        for m in ("watershed", "cellpose"):
            lab, info = segment(img, m)
            panels.append((f"{m}: n={info['n']} (full frame), {info['seconds']:.1f} s CPU", img, lab))
        for c, (t, im, lab) in enumerate(panels):
            show = np.dstack([im] * 3) if lab is None else overlay(im, lab)[..., ::-1]
            if crop:
                y0, y1, x0, x1 = crop
                show = show[y0:y1, x0:x1]
            axs[r, c].imshow(show, cmap="gray", interpolation="nearest")
            axs[r, c].set_title(t if c else f"{name}", fontsize=9)
            axs[r, c].axis("off")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/fig1_overlays.png", dpi=130)
    print("saved figures/fig1_overlays.png")


if __name__ == "__main__":
    main()

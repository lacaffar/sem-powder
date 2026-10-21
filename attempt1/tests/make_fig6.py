"""figures/fig6_satellites.png: satellite detections on real images + accuracy table.

Needs: python analyze.py samples --out out/demo ; python analyze.py data/alloy2/Ti-6Al-4V.tif --out out/ti64_demo
"""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = cv2.imread(str(ROOT / "out/demo/overlays/aa7075_fig2a.png"))[..., ::-1][0:420, 0:560]
    b = cv2.imread(str(ROOT / "out/ti64_demo/overlays/Ti-6Al-4V.png"))[..., ::-1][280:700, 440:1000]
    fig = plt.figure(figsize=(12, 6.6))
    ax1 = fig.add_axes([0.0, 0.33, 0.5, 0.64])
    ax2 = fig.add_axes([0.5, 0.33, 0.5, 0.64])
    for ax, im, t in ((ax1, a, "AA7075 base (SE, Zeiss), crop"), (ax2, b, "Ti-6Al-4V (BSE, TESCAN), crop")):
        ax.imshow(im)
        ax.set_title(t + ": satellites = red circles", fontsize=9)
        ax.axis("off")
    ax3 = fig.add_axes([0.03, 0.0, 0.94, 0.3])
    ax3.axis("off")
    rows = [
        ["", "Real, held-out crops\n(Fig 2c, Ti64+15Cu; 36 sats)", "Synthetic, held-out\n(30 images; 1,508 sats)", "Share of particles with\nsatellites, error (synthetic)"],
        ["Before (solidity proxy)", "F1 0.00", "F1 0.00", "67 pts"],
        ["Segmentation split only", "F1 0.20 (P 0.33, R 0.14)", "F1 0.04", "62 pts"],
        ["Satellite detector (new)", "F1 0.48 (P 0.49, R 0.47)\ncount 35 vs 36 labelled", "F1 0.67 (P 0.82, R 0.57)", "11 pts"],
    ]
    tb = ax3.table(cellText=rows, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 2.3)
    for (r, c), cell in tb.get_celld().items():
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
    fig.savefig(ROOT / "figures/fig6_satellites.png", dpi=130)
    print("saved figures/fig6_satellites.png")


if __name__ == "__main__":
    main()

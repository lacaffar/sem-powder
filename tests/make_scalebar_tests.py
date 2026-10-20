"""Render SEM-style data bars onto a real AA7075 frame with known um/px.

The AA7075 TIFFs ship without a data bar, so this is the only way to test the
scale-bar OCR path end to end. Ground truth um/px = 0.3722 (Zeiss metadata) times
the resample factor. Writes data/scaletest/*.png and truth.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imio import load_gray  # noqa: E402

SRC = ROOT / "data/aa7075/Original_Image_for_Fig_2a.tif"
OUT = ROOT / "data/scaletest"
BASE_UMPX = 0.3722


def font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def make(name, scale, bar_h_frac, bg, fg, label_um, label_txt, style, extras):
    img = load_gray(SRC)
    if scale != 1.0:
        img = np.asarray(Image.fromarray(img).resize(
            (int(img.shape[1] * scale), int(img.shape[0] * scale)), Image.BILINEAR))
    umpx = BASE_UMPX / scale
    h, w = img.shape
    bh = int(h * bar_h_frac)
    canvas = Image.new("L", (w, h + bh), bg)
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    f = font(max(14, bh // 4))
    bar_px = int(round(label_um / umpx))
    y = h + bh // 2
    if style == "zeiss":
        x0 = int(w * 0.04)
        d.text((x0, h + bh * 0.12), label_txt, fill=fg, font=f)
        d.rectangle([x0, y + bh // 8, x0 + bar_px - 1, y + bh // 8 + max(3, bh // 18)], fill=fg)
        xs = int(w * 0.35)
    elif style == "fei":
        x0 = int(w * 0.70)
        d.rectangle([x0, y - bh // 5, x0 + bar_px - 1, y - bh // 5 + max(2, bh // 25)], fill=fg)
        d.rectangle([x0, y - bh // 5 - 6, x0 + 1, y - bh // 5 + 6], fill=fg)
        d.rectangle([x0 + bar_px - 2, y - bh // 5 - 6, x0 + bar_px - 1, y - bh // 5 + 6], fill=fg)
        d.text((x0 + bar_px // 3, y), label_txt, fill=fg, font=f)
        xs = int(w * 0.03)
    else:  # jeol: one line of text with bar in the middle
        x0 = int(w * 0.45)
        d.text((int(w * 0.03), y - bh // 6), "SEI   20kV   WD10mm   x300", fill=fg, font=f)
        d.rectangle([x0, y, x0 + bar_px - 1, y + max(3, bh // 14)], fill=fg)
        d.text((x0 + bar_px + 15, y - bh // 6), label_txt, fill=fg, font=f)
        xs = None
    if xs is not None:
        for i, e in enumerate(extras):
            d.text((xs + (i // 2) * int(w * 0.22), h + bh * (0.15 + 0.4 * (i % 2))), e, fill=fg, font=f)
    out = OUT / f"{name}.png"
    canvas.save(out)
    return out.name, umpx, bar_px


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        make("zeiss_black_100um", 1.0, 0.12, 0, 255, 100, "100 µm", "zeiss",
             ["EHT = 20.00 kV", "WD = 8.5 mm", "Mag = 300 X", "Signal A = SE2"]),
        make("zeiss_white_200um", 1.0, 0.10, 255, 0, 200, "200 µm", "zeiss",
             ["EHT = 15.00 kV", "WD = 10.1 mm", "Mag = 300 X", "Detector = InLens"]),
        make("fei_gray_50um_half", 2.0, 0.09, 40, 230, 50, "50 μm", "fei",
             ["HV 20.00 kV", "HFW 190 µm", "mag 600 x", "WD 9.8 mm"]),
        make("jeol_black_100um_small", 0.75, 0.08, 0, 255, 100, "100μm", "jeol", []),
        make("zeiss_black_0p1mm", 1.0, 0.12, 0, 255, 100, "0.1 mm", "zeiss",
             ["EHT = 20.00 kV", "WD = 8.5 mm", "Mag = 300 X", "Date: 1 Feb 2024"]),
    ]
    with open(OUT / "truth.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "um_per_px", "bar_px"])
        wr.writerows(rows)
    print("\n".join(f"{r[0]} {r[1]:.4f} {r[2]}" for r in rows))


if __name__ == "__main__":
    main()

"""Tune watershed h_frac on held-out synthetic renders 49-50 of each PSD only."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imio import load_gray  # noqa: E402
from src.matching import match  # noqa: E402
from src.metrics import particle_table  # noqa: E402
from src.segment import postprocess, preprocess, segment_watershed  # noqa: E402
from src.synthetic_gt import ground_truth  # noqa: E402

SYN = ROOT / "data/synthetic"
HS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.1, 0.15, 0.2, 0.3]

cache = []
for d in sorted(p for p in SYN.iterdir() if p.is_dir()):
    for k in (49, 50):
        gt, _ = ground_truth(d / f"particles/particles{k}.json")
        cache.append((d.name[0], load_gray(d / f"renders/particles{k}.png"), gt[gt.visible]))

for h in HS:
    rec, prec, e50 = [], [], []
    for psd, img, gv in cache:
        lab, _ = segment_watershed(img, preprocess(img), h_frac=h)
        df = particle_table(postprocess(lab), None)
        mi, gi = match(df[["centroid_x", "centroid_y"]].to_numpy(), gv[["u", "v"]].to_numpy(), gv.d_px.to_numpy())
        rec.append(len(gi) / len(gv))
        prec.append(len(mi) / max(1, len(df)))
        e50.append(np.median(df.ecd_px) / np.median(gv.d_px) - 1)
    print(f"h_frac={h:.2f} recall={np.mean(rec):.3f} precision={np.mean(prec):.3f} "
          f"D50num err mean={np.mean(e50):+.1%} |err|={np.mean(np.abs(e50)):.1%}")

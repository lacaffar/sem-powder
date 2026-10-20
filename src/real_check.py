"""Block 6: image-derived PSD on AA7075 SEM frames vs reported laser diffraction.

Laser diffraction reports a volume-weighted PSD, so it is compared with the
volume-weighted image stats (raw and Miles-Lantuejoul edge-corrected).

Image -> laser sample mapping (inferred, see PROGRESS.md):
  Fig 2a: base 7075            -> '7075'
  Fig 2c: TiC surface-modified -> BM rows (0.3 / 1.0 / 2.0 wt% TiC)
  Fig 2d: TiC volume-modified  -> VM rows (0.3 / 1.0 wt% TiC)

Usage: python -m src.real_check [--methods watershed,cellpose]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AA = ROOT / "data/aa7075"
IMAGES = {
    "Original_Image_for_Fig_2a.tif": ("base 7075", ["7075"]),
    "Original_Image_for_Fig_2c.tif": ("7075+TiC BM (surface)", ["7075 0.3wt%TiC BM", "7075 1.0wt%TiC BM", "7075 2.0wt%TiC BM"]),
    "Original_Image_for_Fig_2d.tif": ("7075+TiC VM (volume)", ["7075 0.3wt%TiC VM", "7075 1.0wt%TiC VM"]),
}


def load_laser():
    rows = list(csv.reader(open(AA / "PSD.CSV", encoding="latin-1"), delimiter=";"))
    h = rows[0]
    i = next(k for k, c in enumerate(h) if c.startswith("Result Between"))
    f = lambda x: float(x.replace(",", ".")) if x.strip() else np.nan  # noqa: E731
    edges = np.array([f(x) for x in rows[1][i:]])
    out = {}
    for r in rows[2:]:
        if not r[0].strip():
            continue
        vol = np.array([f(x) for x in r[i + 1:]])
        out[r[0].strip()] = {
            "D10": f(r[h.index("d (0.1)")]), "D50": f(r[h.index("d (0.5)")]), "D90": f(r[h.index("d (0.9)")]),
            "edges": edges, "cum": np.concatenate([[0], np.cumsum(vol) / np.nansum(vol)]),
        }
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .metrics import analyze_image
    from .segment import overlay

    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="watershed,cellpose")
    a = ap.parse_args()
    methods = a.methods.split(",")
    laser = load_laser()
    out = ROOT / "out/real"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for fn, (desc, refs) in IMAGES.items():
        p = AA / fn
        for m in methods:
            df, s, lab, img, sc = analyze_image(p, m)
            df.to_csv(out / f"{p.stem}__{m}.csv", index=False)
            cv2.imwrite(str(out / f"{p.stem}__{m}.png"), overlay(img, lab, title=f"{fn[-6:-4]} {m} n={s['count']}"))
            curves[(fn, m)] = df
            ref = {k: np.mean([laser[r][k] for r in refs]) for k in ("D10", "D50", "D90")}
            rng = {k: (min(laser[r][k] for r in refs), max(laser[r][k] for r in refs)) for k in ("D10", "D50", "D90")}
            row = {"image": fn, "powder": desc, "method": m, "count": s["count"],
                   "um_per_px": s["um_per_px"], "scale_source": s["scale_source"], "laser_ref": " | ".join(refs)}
            for k in ("D10", "D50", "D90"):
                row[f"laser_{k}"] = ref[k]
                row[f"laser_{k}_range"] = f"{rng[k][0]:.1f}-{rng[k][1]:.1f}"
                row[f"img_{k}_num"] = s[f"{k}_num"]
                row[f"img_{k}_vol"] = s[f"{k}_vol"]
                row[f"img_{k}_vol_ec"] = s[f"{k}_vol_ec"]
                row[f"err_{k}_vol"] = s[f"{k}_vol"] / ref[k] - 1
                row[f"err_{k}_vol_ec"] = s[f"{k}_vol_ec"] / ref[k] - 1
            row["mean_circularity"] = s["mean_circularity"]
            row["mean_aspect_ratio"] = s["mean_aspect_ratio"]
            row["irregular_frac"] = s["irregular_frac"]
            rows.append(row)
            print(f"{fn[-7:-4]} {m:10s} n={s['count']:4d} vol D10/50/90 = "
                  f"{s['D10_vol']:.1f}/{s['D50_vol']:.1f}/{s['D90_vol']:.1f}  edge-corr "
                  f"{s['D10_vol_ec']:.1f}/{s['D50_vol_ec']:.1f}/{s['D90_vol_ec']:.1f}  laser "
                  f"{ref['D10']:.1f}/{ref['D50']:.1f}/{ref['D90']:.1f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "out/aa7075_vs_laser.csv", index=False)

    # figure 4: volume CDFs, image vs laser
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, len(IMAGES), figsize=(4.3 * len(IMAGES), 3.7), sharey=True)
    col = {"watershed": "#1f77b4", "cellpose": "#d62728", "cellpose_ws": "#2ca02c"}
    for ax, (fn, (desc, refs)) in zip(axs, IMAGES.items()):
        for r in refs:
            L = laser[r]
            ax.plot(L["edges"], L["cum"], color="0.4", lw=1.2, ls="--" if r != refs[0] else "-",
                    label=f"laser {r.replace('7075 ', '').replace('wt%', '')}")
        for m in methods:
            d = curves[(fn, m)]
            v = d.ecd_um.to_numpy()
            o = np.argsort(v)
            w = (v ** 3 * d.edge_weight.to_numpy())[o]
            ax.plot(v[o], np.cumsum(w) / w.sum(), color=col[m], lw=1.8, label=f"image {m} (edge-corr.)")
            w0 = (v ** 3)[o]
            ax.plot(v[o], np.cumsum(w0) / w0.sum(), color=col[m], lw=0.8, ls=":", label=f"image {m} (raw)")
        ax.set_xscale("log")
        ax.set_xlim(2, 300)
        ax.set_title(f"Fig {fn.split('_')[-1][:-4]}: {desc}", fontsize=9)
        ax.set_xlabel("diameter (µm)")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("cumulative volume fraction")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "fig4_aa7075_vs_reported.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(res[["image", "method", "count", "laser_D50", "img_D50_vol", "err_D50_vol", "img_D50_vol_ec",
               "err_D50_vol_ec", "err_D10_vol_ec", "err_D90_vol_ec"]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

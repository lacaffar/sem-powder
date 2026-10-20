"""Block 5: validate segmentation + sizing on the DeCost & Holm synthetic renders.

Usage:
  python -m src.validate_synthetic run    --method watershed [--first 1 --last 48]
  python -m src.validate_synthetic report

`run` caches per-image results in out/synth_raw/ (resumable). `report` pools them:
  out/synthetic_validation.csv, figures/fig2_synthetic_psd.png, figures/fig3_d50_parity.png

Ground truth (see src/synthetic_gt.py for the camera derivation):
  generating : every sphere in the JSON (800 per render), diameter = 2 * size.
  visible    : spheres > 50% visible by ray-cast z-buffer and not touching the frame.
Sizes in Blender units. Measured px -> units with ONE nominal scale for the whole
set (median px/unit of visible GT spheres = what a calibrated SEM would give). The
perspective camera makes the true scale vary +-10% with depth; the exact per-sphere
px comparison is also reported (columns *_px).
Renders 49-50 were used for tuning and are excluded here.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data/synthetic"
RAW = ROOT / "out/synth_raw"
METHODS = ["watershed", "cellpose", "cellpose_ws"]
PCT = (10, 50, 90)
MIN_ECD_PX = 2 * (20 / 3.141592653589793) ** 0.5


def psd_dirs():
    return sorted(p for p in SYN.iterdir() if p.is_dir())


def run(method, first, last):
    from .imio import load_gray
    from .matching import match
    from .metrics import particle_table
    from .segment import segment
    from .synthetic_gt import ground_truth
    (RAW / "gt").mkdir(parents=True, exist_ok=True)
    (RAW / method).mkdir(parents=True, exist_ok=True)
    for d in psd_dirs():
        for k in range(first, last + 1):
            tag = f"{d.name[0]}_{k:03d}"
            out = RAW / method / f"{tag}.csv"
            if out.exists():
                continue
            gpath = RAW / "gt" / f"{tag}.csv"
            if gpath.exists():
                gt = pd.read_csv(gpath)
            else:
                try:
                    gt, _ = ground_truth(d / f"particles/particles{k}.json")
                except ValueError as e:  # a/particles37.json is truncated upstream
                    print(f"SKIP {tag}: bad ground-truth JSON ({e})", flush=True)
                    continue
                gt.to_csv(gpath, index=False)
            img = load_gray(d / f"renders/particles{k}.png")
            lab, info = segment(img, method)
            df = particle_table(lab, None)
            gv = gt[gt.visible]
            mi, gi = match(df[["centroid_x", "centroid_y"]].to_numpy(),
                           gv[["u", "v"]].to_numpy(), gv.d_px.to_numpy())
            df["gt_index"] = -1
            df.loc[df.index[mi], "gt_index"] = gv.index.to_numpy()[gi]
            df["seconds"] = info["seconds"]
            df.to_csv(out, index=False)
            print(f"{method} {tag} n={len(df)} vis_gt={len(gv)} matched={len(mi)} t={info['seconds']}s", flush=True)


def wpct(x, pcts=PCT, w=None):
    from .metrics import weighted_percentiles
    x = np.asarray(x, float)
    return weighted_percentiles(x, np.ones_like(x) if w is None else w, pcts)


def report(max_k=None, figures=True):
    """max_k: only renders <= max_k (same-subset method comparison)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from .synthetic_gt import load_particles

    gts = {p.stem: pd.read_csv(p) for p in sorted((RAW / "gt").glob("*.csv"))
           if max_k is None or int(p.stem[2:]) <= max_k}
    sfx = "" if max_k is None else f"_first{max_k}"
    nom = float(np.median(np.concatenate([g.loc[g.visible, "px_per_unit"].to_numpy() for g in gts.values()])))
    methods = [m for m in METHODS if (RAW / m).is_dir() and any((RAW / m).glob("*.csv"))]
    rows, cdfs = [], {}
    for d in psd_dirs():
        L = d.name[0]
        tags = sorted(t for t in gts if t.startswith(L + "_"))
        gen = np.concatenate([2 * load_particles(d / f"particles/particles{int(t[2:])}.json")[1] for t in tags])
        vis = pd.concat([gts[t][gts[t].visible] for t in tags])
        gen_p = wpct(gen)
        vis_p = wpct(vis.d_units)
        vis_pv = wpct(vis.d_units, w=vis.d_units ** 3)
        vis_px = wpct(vis.d_px)
        # resolvable = visible and above the 20 px area floor (ECD >= 5.05 px)
        res_p = wpct(vis.loc[vis.d_px >= MIN_ECD_PX, "d_units"])
        frac_unres = float((vis.d_px < MIN_ECD_PX).mean())
        cdfs[(L, "generating")] = gen
        cdfs[(L, "visible")] = vis.d_units.to_numpy()
        for m in methods:
            parts, n_match, n_vis, secs, ratio = [], 0, 0, [], []
            for t in tags:
                f = RAW / m / f"{t}.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                g = gts[t]
                parts.append(df)
                hit = df[df.gt_index >= 0]
                n_match += len(hit)
                n_vis += int(g.visible.sum())
                secs.append(df.seconds.iloc[0] if len(df) else np.nan)
                ratio.append(hit.ecd_px.to_numpy() / g.loc[hit.gt_index, "d_px"].to_numpy())
            if not parts:
                continue
            allm = pd.concat(parts)
            meas = allm.ecd_px.to_numpy() / nom
            mp = wpct(meas)
            mpv = wpct(meas, w=meas ** 3)
            mpx = wpct(allm.ecd_px)
            cdfs[(L, m)] = meas
            r = {"psd": L, "psd_name": d.name, "method": m, "n_images": len(parts),
                 "n_measured": len(allm), "n_visible_gt": n_vis,
                 "recall": n_match / max(1, n_vis), "precision": n_match / max(1, len(allm)),
                 "matched_size_ratio_median": float(np.median(np.concatenate(ratio))) if ratio else np.nan,
                 "sec_per_image": float(np.nanmean(secs))}
            for i, p in enumerate(PCT):
                r[f"gen_D{p}"] = gen_p[i]
                r[f"vis_D{p}"] = vis_p[i]
                r[f"meas_D{p}"] = mp[i]
                r[f"err_vs_vis_D{p}"] = mp[i] / vis_p[i] - 1
                r[f"err_vs_gen_D{p}"] = mp[i] / gen_p[i] - 1
                r[f"err_vs_vis_D{p}_px"] = mpx[i] / vis_px[i] - 1
            r["res_D50"] = res_p[1]
            r["err_vs_resolvable_D50"] = mp[1] / res_p[1] - 1
            r["err_vs_resolvable_D10"] = mp[0] / res_p[0] - 1
            r["frac_visible_below_floor"] = frac_unres
            r["vis_D50_vol"] = vis_pv[1]
            r["meas_D50_vol"] = mpv[1]
            r["err_vs_vis_D50_vol"] = mpv[1] / vis_pv[1] - 1
            rows.append(r)
    res = pd.DataFrame(rows)
    res["nominal_px_per_unit"] = nom
    rho = []
    for m in methods:
        s = res[res.method == m]
        rv = spearmanr(s.meas_D50, s.vis_D50).statistic
        rg = spearmanr(s.meas_D50, s.gen_D50).statistic
        rv90 = spearmanr(s.meas_D90, s.vis_D90).statistic
        res.loc[res.method == m, "spearman_D50_vs_vis"] = rv
        res.loc[res.method == m, "spearman_D50_vs_gen"] = rg
        res.loc[res.method == m, "spearman_D90_vs_vis"] = rv90
        rho.append((m, rv, rg, rv90))
    out = ROOT / f"out/synthetic_validation{sfx}.csv"
    res.to_csv(out, index=False)

    # ---------------- console summary
    pd.set_option("display.width", 200)
    print(f"nominal scale {nom:.2f} px/unit")
    cols = ["psd", "method", "n_images", "recall", "precision", "matched_size_ratio_median",
            "vis_D50", "gen_D50", "meas_D50", "err_vs_vis_D50", "err_vs_gen_D50",
            "err_vs_vis_D10", "err_vs_vis_D90", "err_vs_vis_D50_px", "err_vs_vis_D50_vol",
            "err_vs_resolvable_D50", "frac_visible_below_floor"]
    print(res[cols].round(3).to_string(index=False))
    agg = res.groupby("method").agg(
        mean_abs_err_D50_vis=("err_vs_vis_D50", lambda x: np.mean(np.abs(x))),
        mean_err_D50_vis=("err_vs_vis_D50", "mean"),
        mean_abs_err_D10_vis=("err_vs_vis_D10", lambda x: np.mean(np.abs(x))),
        mean_abs_err_D90_vis=("err_vs_vis_D90", lambda x: np.mean(np.abs(x))),
        mean_abs_err_D50_gen=("err_vs_gen_D50", lambda x: np.mean(np.abs(x))),
        mean_abs_err_D50_vol=("err_vs_vis_D50_vol", lambda x: np.mean(np.abs(x))),
        mean_abs_err_D50_resolvable=("err_vs_resolvable_D50", lambda x: np.mean(np.abs(x))),
        recall=("recall", "mean"), precision=("precision", "mean"),
        size_ratio=("matched_size_ratio_median", "mean"), sec=("sec_per_image", "mean"))
    print(agg.round(3).to_string())
    for m, rv, rg, rv90 in rho:
        print(f"Spearman {m}: D50 vs visible {rv:.3f}, D50 vs generating {rg:.3f}, D90 vs visible {rv90:.3f}")
    agg.to_csv(ROOT / f"out/synthetic_validation_summary{sfx}.csv")
    if not figures:
        return

    # ---------------- figures
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    show = [p for p in ("a", "c", "h") if (p, "visible") in cdfs]
    colors = {"generating": "0.6", "visible": "k", "watershed": "#1f77b4", "cellpose": "#d62728", "cellpose_ws": "#2ca02c"}
    fig, axs = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, L in zip(axs, show):
        for key in ["generating", "visible"] + methods:
            if (L, key) not in cdfs:
                continue
            v = np.sort(cdfs[(L, key)])
            ls = "--" if key in ("generating", "visible") else "-"
            lab = {"generating": "true (all, generating)", "visible": "true (visible)"}.get(key, key)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=colors[key], label=lab, lw=1.5)
        name = res.loc[res.psd == L, "psd_name"].iloc[0]
        ax.set_title(f"PSD {name}", fontsize=9)
        ax.set_xlabel("diameter (Blender units)")
        ax.set_xlim(0, np.percentile(cdfs[(L, "generating")], 99.5))
    axs[0].set_ylabel("cumulative number fraction")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "fig2_synthetic_psd.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    mk = {"watershed": "o", "cellpose": "s", "cellpose_ws": "^"}
    lo = min(res.vis_D50.min(), res.meas_D50.min()) * 0.9
    hi = max(res.vis_D50.max(), res.meas_D50.max()) * 1.1
    for m in methods:
        s = res[res.method == m]
        ax.scatter(s.vis_D50, s.meas_D50, marker=mk[m], color=colors[m],
                   label=f"{m} (mean |err| {np.mean(np.abs(s.err_vs_vis_D50)):.1%})")
        for _, r in s.iterrows():
            ax.annotate(r.psd, (r.vis_D50, r.meas_D50), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.plot([lo, hi], [lo * 1.1, hi * 1.1], ":", color="0.6", lw=0.8)
    ax.plot([lo, hi], [lo * 0.9, hi * 0.9], ":", color="0.6", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true D50, visible particles (units)")
    ax.set_ylabel("measured D50 (units)")
    ax.set_title("Synthetic D50 parity (8 PSDs, 48 images each)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "fig3_d50_parity.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--method", default="watershed")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=48)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.method, a.first, a.last)
    elif a.last < 48:
        report(max_k=a.last, figures=False)
    else:
        report()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

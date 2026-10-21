"""Plots and self-contained HTML reports (images embedded as base64)."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    _plt().close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_to_b64(bgr, max_w=900):
    import cv2
    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(b.tobytes()).decode()


def psd_figure(df, unit, ref=None, title=""):
    plt = _plt()
    col = "ecd_um" if unit == "um" else "ecd_px"
    d = df[col].to_numpy(float)
    w = df["edge_weight"].to_numpy(float)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    if len(d):
        bins = np.logspace(np.log10(max(d.min() * 0.9, 1e-3)), np.log10(d.max() * 1.1), 30)
        ax[0].hist(d, bins=bins, color="#4c78a8", alpha=0.85)
        ax[0].set_xscale("log")
        o = np.argsort(d)
        ax[1].plot(d[o], np.arange(1, len(d) + 1) / len(d), label="number")
        v = (d ** 3 * w)[o]
        ax[1].plot(d[o], np.cumsum(v) / v.sum(), label="volume (edge-corrected)")
        ax[1].set_xscale("log")
    if ref:
        ax[1].plot([ref["D10"], ref["D50"], ref["D90"]], [0.1, 0.5, 0.9], "ks", label="reference D10/50/90")
    ax[0].set_xlabel(f"ECD ({unit})")
    ax[0].set_ylabel("count")
    ax[1].set_xlabel(f"ECD ({unit})")
    ax[1].set_ylabel("cumulative fraction")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def shape_figure(df):
    plt = _plt()
    keys = [("circularity", "circularity 4πA/P²"), ("feret_ar", "Feret aspect ratio (min/max)"),
            ("convexity", "convexity"), ("solidity", "solidity")]
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.6))
    for ax, (k, lab) in zip(axs, keys):
        v = df[k].dropna()
        ax.hist(v, bins=np.linspace(0.3, 1.0, 29), color="#72b7b2")
        ax.set_xlabel(lab, fontsize=8)
        if len(v):
            ax.axvline(v.median(), color="k", lw=1)
    axs[0].set_ylabel("count")
    fig.tight_layout()
    return fig


CSS = """body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}
th{background:#f2f2f2}td:first-child,th:first-child{text-align:left}.flag{color:#b00020;font-weight:600}
img{max-width:100%}h2{border-bottom:1px solid #ddd;padding-bottom:4px}.small{color:#666;font-size:12px}"""


def table_html(df, floatfmt="{:.3g}"):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if v != v else floatfmt.format(v)
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{head}</tr>{body}</table>"


def sample_report(path, sample, summary, df, overlays, ref=None, model=""):
    unit = summary["size_unit"]
    keys = ["count", "D10_num", "D50_num", "D90_num", "D10_vol_ec", "D50_vol_ec", "D90_vol_ec", "D43", "D32",
            "span_vol_ec", "circularity_median", "feret_ar_median", "convexity_median", "solidity_median",
            "pct_spherical", "pct_touching", "pct_with_satellite_like"]
    rows = [(k, summary.get(k)) for k in keys]
    if ref:
        rows += [("reference D10/D50/D90", f"{ref['D10']:.1f} / {ref['D50']:.1f} / {ref['D90']:.1f}")]
    t = pd.DataFrame(rows, columns=["quantity", f"value ({unit} for sizes)"])
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(sample)}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(sample)}</h1><p class='small'>model: {html.escape(model)} · images: {len(overlays)}</p>"]
    if summary.get("flags"):
        parts.append(f"<p class='flag'>{html.escape(summary['flags'])}</p>")
    parts.append(table_html(t))
    parts.append(f"<h2>Size distribution</h2><img src='data:image/png;base64,{fig_to_b64(psd_figure(df, unit, ref, sample))}'>")
    parts.append(f"<h2>Shape</h2><img src='data:image/png;base64,{fig_to_b64(shape_figure(df))}'>")
    parts.append("<h2>Segmentation</h2>")
    for name, bgr in overlays:
        parts.append(f"<p class='small'>{html.escape(name)}</p><img src='data:image/jpeg;base64,{img_to_b64(bgr)}'>")
    parts.append("</body></html>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def simple_page(path, title, sections):
    """sections: list of (heading, html_fragment)."""
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>"]
    for h, frag in sections:
        parts.append(f"<h2>{html.escape(h)}</h2>{frag}")
    parts.append("</body></html>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")

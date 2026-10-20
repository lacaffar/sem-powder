"""Scale calibration for SEM images.

Steps:
1. Detect the SEM data bar (uniform strip at bottom or top) and return a crop box
   for the image area.
2. Inside the data bar, find the scale bar as the longest thin horizontal segment.
3. OCR the label nearest the scale bar (easyocr); parse number + unit -> um.
4. um_per_px = label_um / bar_len_px.

Resolution order in `calibrate()`:
  CLI --um-per-px  >  scales.csv sidecar  >  scale-bar OCR  >  SEM TIFF metadata
  (Zeiss CZ_SEM / FEI)  >  uncalibrated (sizes in px).
When both OCR and TIFF metadata exist, metadata wins (instrument value) and the
OCR value is kept for the debug record.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from .imio import load_gray

UNIT_TO_UM = {"nm": 1e-3, "um": 1.0, "µm": 1.0, "μm": 1.0, "mm": 1e3, "pm": 1e-6}
# OCR often reads µ as u, p, y, or 'pu'; allow those for 'um'
LABEL_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(nm|mm|[uµμpy]{1,2}m)(?![a-z])", re.IGNORECASE)
# data-bar fields that carry lengths but are not the scale bar label
SKIP_RE = re.compile(r"(WD|HFW|VFW|FWD|HV|EHT|MAG|SPOT|DET|PRESS|KV|DATE|TIME)", re.IGNORECASE)

_READER = None


def _reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=_has_cuda(), verbose=False)
    return _READER


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass
class ScaleResult:
    um_per_px: float | None
    source: str  # cli | sidecar | scalebar | tiff_meta | none
    confidence: float
    crop_box: tuple  # (y0, y1, x0, x1) of the image area to keep
    bar_px: float | None = None
    label: str | None = None
    label_um: float | None = None
    ocr_um_per_px: float | None = None
    meta_um_per_px: float | None = None
    notes: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["notes"] = "; ".join(self.notes)
        return d


# ---------------------------------------------------------------- data bar
def _row_uniformity(img):
    """Fraction of each row's pixels equal (+-2) to that row's modal value."""
    h, w = img.shape
    out = np.empty(h, np.float32)
    for y in range(h):
        hist = np.bincount(img[y], minlength=256)
        m = int(hist.argmax())
        out[y] = hist[max(0, m - 2): m + 3].sum() / w
    return out


def detect_databar(img, min_frac=0.04, uni_thresh=0.45):
    """Return (y0, y1) of a data bar touching the bottom or top edge, or None.

    A data-bar row is mostly one flat value (text and the scale bar are sparse).
    SEM image rows are noisy, so few pixels share the mode. We take the longest run
    of uniform rows (allowing short gaps for dense text rows) touching an edge.
    """
    h, w = img.shape
    u = _row_uniformity(img)
    is_bar = u > uni_thresh
    # close small gaps (text lines can reduce uniformity)
    k = max(3, h // 100) | 1  # odd: an even kernel shifts the bar edge by one row
    is_bar = cv2.morphologyEx(is_bar.astype(np.uint8)[:, None], cv2.MORPH_CLOSE,
                              np.ones((k, 1), np.uint8))[:, 0].astype(bool)
    best = None
    # bottom
    if is_bar[-1]:
        y = h - 1
        while y > 0 and is_bar[y - 1]:
            y -= 1
        if (h - y) >= min_frac * h and (h - y) < 0.4 * h:
            best = ("bottom", y, h)
    if best is None and is_bar[0]:
        y = 0
        while y < h - 1 and is_bar[y + 1]:
            y += 1
        if (y + 1) >= min_frac * h and (y + 1) < 0.4 * h:
            best = ("top", 0, y + 1)
    if best is None:
        # some instruments draw a thin separator line, then the bar; look for a
        # strong full-width edge in the lower 35% followed by a mostly uniform block
        lo = int(h * 0.65)
        sub = u[lo:]
        frac_uniform = np.array([sub[i:].mean() for i in range(len(sub))])
        cand = np.where(frac_uniform > uni_thresh + 0.1)[0]
        if len(cand) and (len(sub) - cand[0]) >= min_frac * h:
            best = ("bottom", lo + int(cand[0]), h)
    return best


# ---------------------------------------------------------------- scale bar
def find_scale_bar(region):
    """Longest thin horizontal segment in a data-bar region.

    Returns (x0, x1, y_center, length_px, score) or None.
    """
    h, w = region.shape
    vals, counts = np.unique(region, return_counts=True)
    bg = int(vals[counts.argmax()])
    fg = (np.abs(region.astype(np.int16) - bg) > 40).astype(np.uint8)
    kx = max(15, w // 60)
    horiz = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((1, kx), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if ww < kx or hh > max(25, h * 0.5):
            continue
        aspect = ww / max(hh, 1)
        if aspect < 4:
            continue
        # a filled box (text background highlight) is not a bar; require thin
        fill = area / (ww * hh)
        score = ww * (1.0 if hh <= h * 0.25 else 0.5) * (0.5 + 0.5 * fill)
        if best is None or score > best[4]:
            # measure length on the raw mask row through the middle (ignore ticks)
            yc = y + hh // 2
            row = fg[yc, x:x + ww]
            length = float(row.sum()) if row.sum() > 0.8 * ww else float(ww)
            best = (int(x), int(x + ww), int(yc), length, score)
    return best


def _fix_digits(text):
    """Map common OCR letter/digit confusions inside numeric runs (1OO -> 100)."""
    def fix(m):
        return m.group(0).translate(str.maketrans("OoIlS", "00115"))
    return re.sub(r"\d[\dOoIlS.,]*|[OoIl](?=[\d.,]*\d)", fix, text)


def _parse_label(text):
    t = _fix_digits(text).replace(",", ".")
    m = LABEL_RE.search(t)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit not in ("nm", "mm"):
        # 'pm'/'ym'/'pum' scale bars do not exist on powder SEM: misread of µm
        unit = "um"
    return num * UNIT_TO_UM[unit], f"{m.group(1)} {unit}"


def _ocr_tokens(res, up):
    """easyocr results -> list of (text, conf, cx, cy, x0, x1, h) in region coords."""
    toks = []
    for box, text, conf in res:
        xs = [b[0] / up for b in box]
        ys = [b[1] / up for b in box]
        toks.append((text, float(conf), np.mean(xs), np.mean(ys), min(xs), max(xs), max(ys) - min(ys)))
    return toks


def ocr_label(region, bar):
    """OCR the data bar, pick the number+unit token nearest the scale bar.

    Single tokens and pairs of horizontally adjacent tokens on the same line are
    both tried ('0.1' + 'mm').
    """
    up = 2 if region.shape[0] < 200 else 1
    reg = cv2.resize(region, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else region
    res = _reader().readtext(reg)
    toks = _ocr_tokens(res, up)
    groups = [(t[0], t[1], t[2], t[3]) for t in toks]
    for a in toks:
        for b in toks:
            if a is b:
                continue
            same_line = abs(a[3] - b[3]) < 0.5 * max(a[6], b[6])
            gap = b[4] - a[5]
            if same_line and -2 < gap < 1.5 * max(a[6], b[6]):
                groups.append((a[0] + " " + b[0], min(a[1], b[1]),
                               0.5 * (a[4] + b[5]), 0.5 * (a[3] + b[3])))
    cands = []
    for text, conf, cx, cy in groups:
        # skip 'WD = 8.5 mm', 'EHT = 20 kV', 'HFW 190 um' style fields
        if "=" in text or ":" in text or SKIP_RE.search(text):
            continue
        p = _parse_label(text)
        if p is None:
            continue
        if bar is not None:
            bx = 0.5 * (bar[0] + bar[1])
            dist = np.hypot((cx - bx) / max(1, bar[1] - bar[0]), (cy - bar[2]) / max(1, region.shape[0]))
        else:
            dist = 1.0
        cands.append((dist, conf, p[0], p[1], text))
    if not cands:
        return None, [r[1] for r in res]
    cands.sort(key=lambda c: (c[0], -c[1]))
    return cands[0], [r[1] for r in res]


# ---------------------------------------------------------------- metadata
def tiff_meta_um_per_px(path, width_px=None):
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            w = width_px or t.pages[0].shape[-1]
            m = t.sem_metadata
            if m:
                if "ap_width" in m and m["ap_width"][1]:
                    val, unit = m["ap_width"][1], str(m["ap_width"][2])
                    # unit text is often mangled (µ -> '?'); only nm/mm are unambiguous
                    f = 1e-3 if unit.endswith("nm") else 1e3 if unit.endswith("mm") else 1.0
                    return float(val) * f / w
                if "ap_image_pixel_size" in m:
                    val, unit = m["ap_image_pixel_size"][1], m["ap_image_pixel_size"][2]
                    return float(val) * {"nm": 1e-3, "pm": 1e-6, "mm": 1e3}.get(unit, 1.0)
            f = t.fei_metadata
            if f and "Scan" in f and "PixelWidth" in f["Scan"]:
                return float(f["Scan"]["PixelWidth"]) * 1e6
    except Exception:
        return None
    return None


def _sidecar(path, sidecar):
    cands = [Path(sidecar)] if sidecar else [Path(path).parent / "scales.csv"]
    for c in cands:
        if c.is_file():
            with open(c, newline="") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 2 and row[0].strip() == Path(path).name:
                        try:
                            return float(row[1])
                        except ValueError:
                            pass
    return None


# ---------------------------------------------------------------- main entry
def calibrate(path, img=None, um_per_px=None, sidecar=None, debug_path=None, use_meta=True):
    img = load_gray(path) if img is None else img
    h, w = img.shape
    crop = (0, h, 0, w)
    bar_info = detect_databar(img)
    notes = []
    res = ScaleResult(None, "none", 0.0, crop, notes=notes)

    if bar_info is not None:
        side, y0, y1 = bar_info
        res.crop_box = (0, y0, 0, w) if side == "bottom" else (y1, h, 0, w)
        region = img[y0:y1]
        bar = find_scale_bar(region)
        hit, texts = ocr_label(region, bar)
        notes.append(f"databar {side} rows {y0}-{y1}")
        notes.append("ocr: " + " | ".join(texts)[:200])
        if bar is not None and hit is not None:
            dist, conf, label_um, label, _ = hit
            res.bar_px = bar[3]
            res.label, res.label_um = label, label_um
            res.ocr_um_per_px = label_um / bar[3]
            # confidence: OCR conf, penalised if label far from bar
            res.confidence = float(conf) * (1.0 if dist < 1.0 else 0.6)
        elif bar is None:
            notes.append("no scale bar line found")
        else:
            notes.append("no number+unit label parsed")
    else:
        notes.append("no data bar detected")

    meta = tiff_meta_um_per_px(path, w) if use_meta else None
    res.meta_um_per_px = meta

    if um_per_px:
        res.um_per_px, res.source, res.confidence = float(um_per_px), "cli", 1.0
    elif (sc := _sidecar(path, sidecar)) is not None:
        res.um_per_px, res.source, res.confidence = sc, "sidecar", 1.0
    elif meta is not None:
        res.um_per_px, res.source = meta, "tiff_meta"
        res.confidence = 1.0
        if res.ocr_um_per_px:
            err = res.ocr_um_per_px / meta - 1
            notes.append(f"ocr vs meta {err:+.1%}")
    elif res.ocr_um_per_px and res.confidence >= 0.3:
        res.um_per_px, res.source = res.ocr_um_per_px, "scalebar"
    else:
        notes.append("UNCALIBRATED: sizes reported in px")

    if debug_path is not None:
        save_debug(img, res, bar_info, debug_path)
    return res


def save_debug(img, res, bar_info, out):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    y0, y1, x0, x1 = res.crop_box
    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 0), 2)
    if bar_info is not None:
        _, by0, by1 = bar_info
        region = img[by0:by1]
        bar = find_scale_bar(region)
        cv2.rectangle(vis, (0, by0), (img.shape[1] - 1, by1 - 1), (255, 128, 0), 2)
        if bar is not None:
            cv2.line(vis, (bar[0], by0 + bar[2]), (bar[1], by0 + bar[2]), (0, 0, 255), 3)
    txt = f"{res.source}: " + (f"{res.um_per_px:.4f} um/px" if res.um_per_px else "uncalibrated")
    if res.label:
        txt += f" | label '{res.label}' bar {res.bar_px:.0f}px"
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)

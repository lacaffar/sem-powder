"""Public labeled SEM powder data (MIT licensed), used to train and benchmark the
particle and satellite models.

AMPIS  (Cohn et al. 2021, JOM 73:2159; github.com/rccohn/AMPIS @ fd822b1):
  gas-atomized Ni superalloy, 1024x1536 SEM tiles.
  particles : 4 train images (1122 outlines), 1 val image (238)   -> Cohn's split
  satellites: 8 train images (836), 2 val images (193)
SALAS  (Price et al. 2021, IMMI 10:661; github.com/Data-Driven-Materials-Science/SALAS @ 1ced8f9):
  Al 5056, IN718, SS 17-4, Ti-Nb-Zr, Ti-6Al-4V at 250-3000x, 768x1024.
  satellites: 20 train images (1412), 6 val images (445)          -> Price's split

Labels are VIA polygons, converted to instance label images (later polygons are
painted on top). Particle outlines in AMPIS include their satellites; satellite
outlines are separate (they overlap the particle), as in Cohn's method.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from skimage.draw import ellipse as draw_ellipse
from skimage.draw import polygon as draw_polygon

from .download import fetch

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "public"
AMPIS_SHA = "fd822b1b4cb82aa54d8d90f537a5742eef8b12f8"
SALAS_SHA = "1ced8f996cb9eca53b4cd382d1054cf69cf734fc"
AMPIS_RAW = f"https://raw.githubusercontent.com/rccohn/AMPIS/{AMPIS_SHA}/examples/powder/data/"
AMPIS_IMGS = ["Sc1Tile_001-001-000_0-000", "Sc1Tile_001-002-000_0-000", "Sc1Tile_001-005-000_0-000",
              "Sc1Tile_008-006-000_0-000", "Sc2Tile_001-001-000_0-000", "Sc2Tile_003-001-000_0-000",
              "Sc2Tile_010-006-000_0-000", "Sc3Tile_008-007-000_0-000", "Sc3Tile_010-001-000_0-000",
              "Sc4Tile_002-008-000_0-000", "Sc4Tile_008-004-000_0-000"]
AMPIS_JSON = ["via_powder_particle_masks_training.json", "via_powder_particle_masks_validation.json",
              "via_powder_satellite_masks_training.json", "via_powder_satellite_masks_validation.json"]
SALAS_ZIP = f"https://codeload.github.com/Data-Driven-Materials-Science/SALAS/zip/{SALAS_SHA}"


def log(m):
    print(f"[public] {m}", flush=True)


def fetch_public():
    a = PUB / "ampis"
    a.mkdir(parents=True, exist_ok=True)
    for n in AMPIS_IMGS:
        if not (a / f"{n}.png").exists():
            fetch(AMPIS_RAW + f"images_png/{n}.png", a / f"{n}.png")
    for n in AMPIS_JSON:
        if not (a / n).exists():
            fetch(AMPIS_RAW + f"via_2.0.8/{n}", a / n)
    log(f"AMPIS: {len(list(a.glob('*.png')))} images")
    s = PUB / "salas"
    if not (s / "VIA" / "satellite_training.json").exists():
        tmp = PUB / "_dl" / "salas.zip"
        log("SALAS: downloading repository zip (~77 MB)")
        fetch(SALAS_ZIP, tmp)
        with zipfile.ZipFile(tmp) as z:
            for m in z.namelist():
                rel = m.split("/", 1)[1] if "/" in m else m
                if (rel.startswith("data/SEM/") or rel.startswith("data/VIA/")) and not m.endswith("/"):
                    out = s / rel[len("data/"):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(m))
        tmp.unlink()
    log(f"SALAS: {len(list((s / 'SEM').glob('*.png')))} images")


def via_labels(json_path, img_dir):
    """-> list of (name, gray uint8 image, int32 label image)."""
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    d = d.get("_via_img_metadata", d)
    out = []
    for v in d.values():
        p = Path(img_dir) / v["filename"]
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        lab = np.zeros((H, W), np.int32)
        k = 0
        for r in v.get("regions", []):
            sa = r["shape_attributes"]
            t = sa.get("name")
            if t in ("polygon", "polyline"):
                rr, cc = draw_polygon(np.array(sa["all_points_y"]), np.array(sa["all_points_x"]), (H, W))
            elif t == "rect":
                rr, cc = draw_polygon(np.array([sa["y"], sa["y"], sa["y"] + sa["height"], sa["y"] + sa["height"]]),
                                      np.array([sa["x"], sa["x"] + sa["width"], sa["x"] + sa["width"], sa["x"]]), (H, W))
            elif t in ("circle", "ellipse"):
                ry, rx = (sa["r"], sa["r"]) if t == "circle" else (sa["ry"], sa["rx"])
                rr, cc = draw_ellipse(sa["cy"], sa["cx"], ry, rx, (H, W))
            else:
                continue
            if len(rr) < 3:
                continue
            k += 1
            lab[rr, cc] = k
        out.append((Path(v["filename"]).stem, img, lab))
    return out


def datasets():
    """{'particles': {'train': [...], 'val': [...]}, 'satellites': {...}}; items (name, img, lab, source)."""
    a, s = PUB / "ampis", PUB / "salas"
    tag = lambda items, src: [(n, i, l, src) for n, i, l in items]  # noqa: E731
    return {
        "particles": {
            "train": tag(via_labels(a / AMPIS_JSON[0], a), "AMPIS"),
            "val": tag(via_labels(a / AMPIS_JSON[1], a), "AMPIS"),
        },
        "satellites": {
            "train": tag(via_labels(a / AMPIS_JSON[2], a), "AMPIS") + tag(via_labels(s / "VIA" / "satellite_training.json", s / "SEM"), "SALAS"),
            "val": tag(via_labels(a / AMPIS_JSON[3], a), "AMPIS") + tag(via_labels(s / "VIA" / "satellite_validation.json", s / "SEM"), "SALAS"),
        },
    }

"""Trained QC models published as a GitHub Release asset, so a new machine needs no training.

sem_models.zip (made on the Hunt PC after `qcmodels`) holds:
  qc_models.json        which models QC uses + tuned satellite settings
  cp_particles(.json)   Cellpose-SAM fine-tuned on AMPIS particle outlines
  mrcnn_satellites.pt   Mask R-CNN trained on AMPIS + SALAS satellite outlines
Pinned URL + SHA-256. Override for a newer set: SEMCHAR_MODELS_URL and SEMCHAR_MODELS_SHA256.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from .download import fetch
from .models import ROOT

MODELS = ROOT / "models"
MODELS_URL = "https://github.com/lacaffar/sem-powder/releases/download/models-v1/sem_models.zip"
MODELS_SHA256 = "41cd8035e5dfd5528c4b802bbf0135836ca93d544bcfe0c99bbf3f0cdf58ef26"


def have_models():
    cfg_p = MODELS / "qc_models.json"
    if not cfg_p.exists():
        return False
    cfg = json.loads(cfg_p.read_text())
    return all((ROOT / v["path"]).exists() for v in cfg.values())


def unpack(zip_path, log=print):
    """Extract into models/ (flat names only; nothing outside models/ is written)."""
    MODELS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = Path(m.filename.replace("\\", "/")).name
            if m.is_dir() or not name or name.startswith("."):
                continue
            log(f"  {name} ({m.file_size / 2**20:.0f} MB)")
            with z.open(m) as src, open(MODELS / name, "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)


def ensure_models(log=print):
    """-> True when the QC models are present (downloading them if needed)."""
    if have_models():
        return True
    url = os.environ.get("SEMCHAR_MODELS_URL", MODELS_URL)
    sha = os.environ.get("SEMCHAR_MODELS_SHA256", MODELS_SHA256 if url == MODELS_URL else None)
    tmp = MODELS / "_download" / "sem_models.zip"
    try:
        log(f"--- downloading trained QC models (~1.4 GB) from {url}")
        fetch(url, tmp, sha256=sha)
        log("  checksum OK" if sha else "  (no checksum given)")
        unpack(tmp, log)
    except Exception as e:
        log(f"  model download failed: {e}")
        return False
    finally:
        tmp.unlink(missing_ok=True)
        if tmp.parent.exists() and not any(tmp.parent.iterdir()):
            tmp.parent.rmdir()
    ok = have_models()
    log("  models ready" if ok else "  zip did not contain every model listed in qc_models.json")
    return ok

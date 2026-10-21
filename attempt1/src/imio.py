"""Image loading shared by all modules. Returns 2D uint8 grayscale."""
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_gray(path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(str(path))
    else:
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise IOError(f"cannot read {path}")
        if a.ndim == 3:
            a = a[..., ::-1]  # BGR(A) -> RGB(A)
    a = np.asarray(a)
    while a.ndim > 3:  # multi-page / extra axes: take first plane
        a = a[0]
    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            a = a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        else:  # (pages, H, W)
            a = a[0]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo + 1e-12) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def list_images(folder):
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)

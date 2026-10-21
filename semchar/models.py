"""Segmenters with one interface: seg.segment(img_u8) -> int32 label image.

Registry names (used on the command line and in reports):
  watershed      classical baseline from attempt 1 (CPU, no learning)
  cpsam          Cellpose-SAM v2 (cellpose 4.2 'cpsam_v2'), native image scale
  cpsam_auto     same model, image rescaled so the typical particle is ~30-60 px
  cpsam_ft       Cellpose-SAM fine-tuned on corrected SEM labels (visit 2)
  sam_amg        Segment Anything ViT-L 'segment everything' + particle filter

All outputs go through classic.postprocess (drop edge-touching and < 20 px).
GPU is used when available. bfloat16 only on GPUs with native support; the
Quadro RTX 4000 (Turing) has none, so it runs float32 (bf16 emulation on Turing
and on CPU is ~10x slower).
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np

from . import classic

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SEMCHAR_CACHE", ROOT / "cache"))
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(CACHE / "cellpose"))


def torch_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def native_bf16():
    import torch
    if os.environ.get("SEMCHAR_FP32") == "1" or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def typical_ecd_px(img_u8):
    """Typical particle diameter from the classical foreground (cheap, scale-free)."""
    fg = classic.foreground(classic.preprocess(img_u8))
    return 2.0 * classic.estimate_radius(fg)


class Segmenter:
    name = "base"

    def segment(self, img_u8: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        free_gpu()

    def __call__(self, img_u8):
        t = time.time()
        lab = classic.postprocess(self.segment(img_u8))
        self.last_seconds = time.time() - t
        return lab


class Watershed(Segmenter):
    name = "watershed"

    def segment(self, img_u8):
        return classic.segment_watershed(img_u8)[0]


class CellposeSeg(Segmenter):
    """Cellpose 4 (Cellpose-SAM). scale: 'native' or 'auto'."""

    def __init__(self, pretrained="cpsam_v2", scale="native", name=None, batch_size=None,
                 flow_threshold=0.4, cellprob_threshold=0.0):
        import torch
        from cellpose import models
        self.name = name or ("cpsam" if scale == "native" else "cpsam_auto")
        self.scale = scale
        gpu = torch.cuda.is_available()
        self.model = models.CellposeModel(gpu=gpu, pretrained_model=str(pretrained),
                                          use_bfloat16=native_bf16())
        if batch_size is None:
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if gpu else 0
            batch_size = 16 if vram >= 16 else 8 if vram >= 7 else 4
        self.batch_size = batch_size
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

    def diameter_for(self, img_u8):
        if self.scale != "auto":
            return None
        d = typical_ecd_px(img_u8)
        if d < 15:        # tiny particles: enlarge so typical ~30 px
            return max(d, 4.0)
        if d > 90:        # huge particles: shrink so typical ~60 px
            return d / 2.0
        return None

    def segment(self, img_u8):
        d = self.diameter_for(img_u8)
        masks, _, _ = self.model.eval(img_u8, batch_size=self.batch_size, diameter=d,
                                      flow_threshold=self.flow_threshold,
                                      cellprob_threshold=self.cellprob_threshold)
        return np.asarray(masks, dtype=np.int32)

    def close(self):
        del self.model
        free_gpu()


def sam_checkpoint():
    p = CACHE / "sam" / Path(SAM_URL).name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        from .download import fetch
        fetch(SAM_URL, p)
    return p


class SamAMG(Segmenter):
    """Segment Anything (ViT-L) automatic mask generator, kept only where a mask is
    mostly foreground (bright particle, not substrate)."""
    name = "sam_amg"

    def __init__(self, points_per_side=32, crop_n_layers=1):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        dev = torch_device()
        sam = sam_model_registry["vit_l"](checkpoint=str(sam_checkpoint())).to(dev)
        self.gen = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, points_per_batch=32 if dev.type == "cuda" else 16,
            pred_iou_thresh=0.86, stability_score_thresh=0.90, crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=2, min_mask_region_area=classic.MIN_AREA_PX)

    def segment(self, img_u8):
        rgb = np.repeat(img_u8[..., None], 3, axis=2)
        anns = self.gen.generate(rgb)
        H, W = img_u8.shape
        fg = classic.foreground(classic.preprocess(img_u8))
        lab = np.zeros((H, W), np.int32)
        k = 0
        # paint large first so smaller (more specific) masks win overlaps
        for a in sorted(anns, key=lambda a: -a["area"]):
            m = a["segmentation"]
            if a["area"] > 0.25 * H * W:
                continue
            if fg[m].mean() < 0.6:   # mostly substrate
                continue
            k += 1
            lab[m] = k
        return lab

    def close(self):
        del self.gen
        free_gpu()


def make(name, **kw):
    if name == "watershed":
        return Watershed()
    if name == "cpsam":
        return CellposeSeg("cpsam_v2", "native", **kw)
    if name == "cpsam_auto":
        return CellposeSeg("cpsam_v2", "auto", **kw)
    if name == "cpsam_ft":
        p = kw.pop("path", None) or (ROOT / "models" / "cpsam_powder")
        if not Path(p).exists():
            raise FileNotFoundError(f"fine-tuned model not found: {p} (run visit 2 first)")
        return CellposeSeg(p, kw.pop("scale", "native"), name="cpsam_ft", **kw)
    if name == "sam_amg":
        return SamAMG(**kw)
    raise ValueError(f"unknown model {name}")


ALL_ZERO_SHOT = ["watershed", "cpsam", "cpsam_auto", "sam_amg"]

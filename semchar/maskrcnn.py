"""Mask R-CNN (the method of Cohn 2021, Price 2021, Bakas 2022) via torchvision, for comparison
with Cellpose-SAM. torchvision's implementation needs no Detectron2 build, so it runs on Windows.

Model: maskrcnn_resnet50_fpn_v2, COCO-pretrained, 1 object class (particle OR satellite: one
model per task, as in Cohn 2021). Training: random 512x512 crops (flips, 90 degree rotations),
upscaled x1.5625 by the model transform (512 -> 800); SGD, mixed precision on GPU.
Inference: whole image upscaled by the same factor, masks > 0.5, detections with score >= 0.5.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import classic
from .models import CACHE, Segmenter, free_gpu, torch_device

os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
CROP = 512
SCALE = 800 / CROP


def build(num_classes=2, pretrained=True):
    import torchvision
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, num_classes)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    m.roi_heads.detections_per_img = 1000   # dense powder images hold hundreds of particles
    m.rpn._pre_nms_top_n = dict(training=4000, testing=6000)
    m.rpn._post_nms_top_n = dict(training=2000, testing=3000)
    m.transform.min_size = (800,)
    m.transform.max_size = 1333
    _ = torchvision
    return m


def _sample(items, rng):
    """Random crop with >= 1 instance -> (3xHxW float tensor, target dict)."""
    import torch
    for _ in range(50):
        img, lab = items[rng.integers(len(items))]
        H, W = lab.shape
        y0 = rng.integers(0, max(1, H - CROP + 1))
        x0 = rng.integers(0, max(1, W - CROP + 1))
        im = np.zeros((CROP, CROP), np.uint8)
        lb = np.zeros((CROP, CROP), np.int32)
        ci = img[y0:y0 + CROP, x0:x0 + CROP]
        cl = lab[y0:y0 + CROP, x0:x0 + CROP]
        im[:ci.shape[0], :ci.shape[1]] = ci
        lb[:cl.shape[0], :cl.shape[1]] = cl
        k = rng.integers(4)
        im, lb = np.rot90(im, k), np.rot90(lb, k)
        if rng.random() < 0.5:
            im, lb = im[:, ::-1], lb[:, ::-1]
        ids = [i for i in np.unique(lb) if i > 0]
        masks, boxes = [], []
        for i in ids:
            m = lb == i
            if m.sum() < 6:
                continue
            ys, xs = np.nonzero(m)
            if xs.max() <= xs.min() or ys.max() <= ys.min():
                continue
            masks.append(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        if not masks:
            continue
        t = torch.from_numpy(np.ascontiguousarray(im)).float().div(255).expand(3, CROP, CROP).clone()
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.ones(len(masks), dtype=torch.int64),
                  "masks": torch.from_numpy(np.stack(masks).astype(np.uint8))}
        return t, target
    raise RuntimeError("could not sample a crop containing an instance")


def train(train_items, out_path, iters=None, batch=2, lr=0.01, log=print, seed=0):
    """train_items: list of (gray uint8 image, int32 label image)."""
    import torch
    iters = iters or int(os.environ.get("SEMCHAR_MRCNN_ITERS", 3000))
    dev = torch_device()
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("SEMCHAR_VRAM_FRACTION", 0.92)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for bs in (batch, 1):
        try:
            model = build().to(dev)
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.75), int(iters * 0.9)], 0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
            t0 = time.time()
            for it in range(iters):
                batch_items = [_sample(train_items, rng) for _ in range(bs)]
                imgs = [b[0].to(dev) for b in batch_items]
                tgts = [{k: v.to(dev) for k, v in b[1].items()} for b in batch_items]
                warm = min(1.0, (it + 1) / 200)
                for g in opt.param_groups:
                    g["lr"] = g.get("initial_lr", lr) * warm if it < 200 else g["lr"]
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    losses = model(imgs, tgts)
                    loss = sum(losses.values())
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                if (it + 1) % 250 == 0 or it == 0:
                    log(f"    iter {it + 1}/{iters} loss {loss.item():.3f} ({(time.time() - t0) / (it + 1):.2f} s/iter)")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
            log(f"  saved {out_path} ({(time.time() - t0) / 60:.1f} min, batch {bs}, peak GPU {peak:.1f} GB)")
            del model, opt
            free_gpu()
            return {"minutes": (time.time() - t0) / 60, "batch": bs, "iters": iters, "peak_vram_gb": peak}
        except torch.cuda.OutOfMemoryError:
            log(f"  out of GPU memory at batch {bs}; retrying smaller")
            free_gpu()
    raise RuntimeError("Mask R-CNN training did not fit in GPU memory")


class MaskRCNNSeg(Segmenter):
    def __init__(self, path, name="mrcnn", score_thresh=0.5, min_area=classic.MIN_AREA_PX):
        import torch
        self.name = name
        self.dev = torch_device()
        self.model = build(pretrained=False)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.to(self.dev).eval()
        self.score_thresh = score_thresh
        self.min_area = min_area

    def segment(self, img_u8):
        import torch
        H, W = img_u8.shape
        self.model.transform.min_size = (int(round(min(H, W) * SCALE)),)
        self.model.transform.max_size = int(round(max(H, W) * SCALE)) + 1
        t = torch.from_numpy(img_u8).float().div(255).expand(3, H, W).clone().to(self.dev)
        with torch.no_grad(), torch.autocast(device_type=self.dev.type, enabled=self.dev.type == "cuda"):
            out = self.model([t])[0]
        scores = out["scores"].float().cpu().numpy()
        masks = (out["masks"][:, 0].float().cpu().numpy() > 0.5)
        lab = np.zeros((H, W), np.int32)
        k = 0
        for i in np.argsort(scores):          # low score first; higher scores paint over
            if scores[i] < self.score_thresh:
                continue
            k += 1
            lab[masks[i]] = k
        return lab

    def close(self):
        del self.model
        free_gpu()

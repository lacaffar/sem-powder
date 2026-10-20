"""Ground truth for the DeCost & Holm synthetic powder renders.

Geometry, read from data/synthetic/blender_powder.py and blend.sh:
- Particle JSON: 'size' (sphere radius, passed to primitive_uv_sphere_add(size=...),
  which is the radius in Blender 2.7x) and x, y, z in [0, 1].
- World position = [11, 11, 2] * (x, y, z) - [5.5, 5.5, 1.0]  ->  x,y in [-5.5, 5.5], z in [-1, 1].
- Camera at (0, 0, 10), rotation (0, 0, 0): looks straight down -z, image up = +y.
- Camera intrinsics are not set in the script, so Blender 2.7x defaults apply:
  perspective, focal length 35 mm, sensor width 32 mm, sensor fit AUTO.
- Render resolution 1024 x 1024 at the default 50% resolution scale -> 512 x 512 PNG
  (matches the files).
- Focal length in px: f = 35 / 32 * 512 = 560 px.
- Scale at depth z: px_per_unit(z) = f / (10 - z). z = 0 -> 56.0 px/unit; z = 1 -> 62.2.
  Image half-width covers 4.57 units at z = 0, so the 11 x 11 box overfills the frame.

Visibility: each pixel's ray is intersected with every sphere (z-buffer), giving an
exact ID map of which sphere is seen at each pixel. A sphere's visible fraction =
visible px / its full projected disk px. Visible GT = spheres with visible fraction
> 0.5 whose projected disk does not touch the image border.

Sizes are in Blender units. Projected diameter in px is converted with the
per-sphere depth, so measured-vs-true comparisons in px are exact for the visible set.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

W = H = 512
F_PX = 35.0 / 32.0 * 512  # 560
CAM_Z = 10.0
BOX = np.array([11.0, 11.0, 2.0])


def load_particles(json_path):
    with open(json_path) as fh:
        d = json.load(fh)
    p = d["particles"]
    xyz = np.array([[q["x"], q["y"], q["z"]] for q in p]) * BOX - BOX / 2
    r = np.array([q["size"] for q in p])
    return xyz, r, d


@lru_cache(maxsize=1)
def _rays():
    u = np.arange(W) + 0.5 - W / 2
    v = np.arange(H) + 0.5 - H / 2
    uu, vv = np.meshgrid(u, v)
    d = np.stack([uu / F_PX, -vv / F_PX, -np.ones_like(uu)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project(xyz, r):
    """Projected centre (u, v) in px and approx projected radius in px."""
    depth = CAM_Z - xyz[:, 2]
    u = W / 2 + F_PX * xyz[:, 0] / depth
    v = H / 2 - F_PX * xyz[:, 1] / depth
    rp = F_PX * r / np.sqrt(np.maximum(depth ** 2 - r ** 2, 1e-9))
    return u, v, rp, depth


def render_ids(xyz, r):
    """Ray-cast z-buffer. Returns (id_map, full_px) where id_map[y, x] is the index of
    the visible sphere (-1 background) and full_px[i] the sphere's unoccluded disk area."""
    rays = _rays()
    O = np.array([0.0, 0.0, CAM_Z])
    zbuf = np.full((H, W), np.inf)
    ids = np.full((H, W), -1, np.int32)
    full = np.zeros(len(r), np.int64)
    u, v, rp, _ = project(xyz, r)
    for i in range(len(r)):
        x0, x1 = int(np.floor(u[i] - rp[i] - 2)), int(np.ceil(u[i] + rp[i] + 2))
        y0, y1 = int(np.floor(v[i] - rp[i] - 2)), int(np.ceil(v[i] + rp[i] + 2))
        x0c, x1c, y0c, y1c = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        if x0c >= x1c or y0c >= y1c:
            continue
        d = rays[y0c:y1c, x0c:x1c]
        oc = O - xyz[i]
        b = d @ oc
        c = oc @ oc - r[i] ** 2
        disc = b * b - c
        hit = disc > 0
        if not hit.any():
            continue
        t = -b - np.sqrt(np.where(hit, disc, 0))
        full[i] = int(hit.sum())
        zb = zbuf[y0c:y1c, x0c:x1c]
        closer = hit & (t < zb)
        zb[closer] = t[closer]
        ids[y0c:y1c, x0c:x1c][closer] = i
    return ids, full


def ground_truth(json_path, vis_thresh=0.5):
    """Per-sphere table for one render, with visibility flags."""
    import pandas as pd
    xyz, r, meta = load_particles(json_path)
    ids, full = render_ids(xyz, r)
    vis = np.bincount(ids[ids >= 0], minlength=len(r))
    u, v, rp, depth = project(xyz, r)
    frac = np.where(full > 0, vis / np.maximum(full, 1), 0.0)
    in_frame = full > 0
    edge = (u - rp < 1) | (u + rp > W - 1) | (v - rp < 1) | (v + rp > H - 1)
    df = pd.DataFrame({
        "r_units": r, "d_units": 2 * r, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "u": u, "v": v, "rp_px": rp, "d_px": 2 * rp, "depth": depth,
        "px_per_unit": F_PX / depth, "full_px": full, "vis_px": vis, "vis_frac": frac,
        "in_frame": in_frame, "touches_edge": edge,
    })
    df["visible"] = in_frame & (frac > vis_thresh) & ~edge
    return df, ids

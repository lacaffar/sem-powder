"""Synthetic satellite benchmark: SEM-like renders of powder with known satellites.

No public SEM powder set labels satellites, so this renders scenes where every
satellite is known:
- Parents: spheres resting on the substrate (monolayer), lognormal radii.
- Satellites: small spheres fused onto a parent's surface (centre at distance
  R + f*r from the parent centre, f in [0.2, 0.8] = partly embedded), radius
  6-35% of the parent radius, anywhere on the visible hemisphere including the rim.
- Fines: free small spheres on the substrate (NOT satellites; negatives).
- Orthographic camera (SEM at low mag is near-orthographic), z-buffer visibility.
- Two shading styles with randomised parameters per image:
    se : Lambertian from a tilted detector direction + edge brightening +
         small-particle brightening (small spheres look bright in SE, as in AA7075).
    bse: flatter albedo, mild top lighting, darker rims (as in TESCAN Ti-6Al-4V).
  Plus per-particle surface texture, blur, and noise.
GT satellites for scoring: >= 40% of the projected disk visible and r >= 2 px.

Usage: python -m src.synth_satellites --n 60 --out data/synth_sat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

H, W = 512, 640


def make_scene(rng):
    parents = []
    med_r = rng.uniform(22, 40)
    tries = 0
    while tries < 4000 and len(parents) < 60:
        tries += 1
        r = float(np.clip(rng.lognormal(np.log(med_r), 0.3), 10, 90))
        x, y = rng.uniform(-r, W + r), rng.uniform(-r, H + r)
        if all(np.hypot(x - p[0], y - p[1]) > 0.92 * (r + p[3]) for p in parents):
            parents.append((x, y, r, r))  # (x, y, z=r on substrate, r)
    spheres = [dict(x=p[0], y=p[1], z=p[2], r=p[3], kind="parent", parent=-1) for p in parents]
    lam = rng.uniform(0.4, 2.5)  # mean satellites per parent for this image
    for pi, p in enumerate(parents):
        px, py, pz, R = p
        for _ in range(rng.poisson(lam)):
            r = R * rng.uniform(0.06, 0.35)
            if r < 1.5:
                continue
            # direction: visible hemisphere incl. rim (n_z > -0.15)
            while True:
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                if d[2] > -0.15:
                    break
            dist = R + rng.uniform(0.2, 0.8) * r
            c = np.array([px, py, pz]) + d * dist
            if c[2] - r < 0:  # would sink into the substrate
                continue
            spheres.append(dict(x=c[0], y=c[1], z=c[2], r=r, kind="satellite", parent=pi))
    # free fines on the substrate (negatives)
    for _ in range(rng.integers(10, 50)):
        r = rng.uniform(2.0, 0.3 * med_r)
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        spheres.append(dict(x=x, y=y, z=r, r=r, kind="fine", parent=-1))
    return spheres


def zbuffer(spheres):
    """Top surface height, sphere id and unit normal for every pixel (orthographic)."""
    zb = np.full((H, W), -np.inf)
    ids = np.full((H, W), -1, np.int32)
    full = np.zeros(len(spheres), np.int64)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    for i, s in enumerate(spheres):
        x0, x1 = int(max(0, s["x"] - s["r"] - 1)), int(min(W, s["x"] + s["r"] + 2))
        y0, y1 = int(max(0, s["y"] - s["r"] - 1)), int(min(H, s["y"] + s["r"] + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        dx = xx[y0:y1, x0:x1] - s["x"]
        dy = yy[y0:y1, x0:x1] - s["y"]
        q = s["r"] ** 2 - dx * dx - dy * dy
        hit = q > 0
        full[i] = int(hit.sum())
        z = s["z"] + np.sqrt(np.where(hit, q, 0))
        zr = zb[y0:y1, x0:x1]
        top = hit & (z > zr)
        zr[top] = z[top]
        ids[y0:y1, x0:x1][top] = i
    return zb, ids, full


def shade(spheres, zb, ids, rng, style):
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    m = ids >= 0
    sx = np.array([s["x"] for s in spheres])
    sy = np.array([s["y"] for s in spheres])
    sz = np.array([s["z"] for s in spheres])
    sr = np.array([s["r"] for s in spheres])
    nx = np.zeros((H, W))
    ny = np.zeros((H, W))
    nz = np.zeros((H, W))
    i = ids[m]
    nx[m] = (xx[m] - sx[i]) / sr[i]
    ny[m] = (yy[m] - sy[i]) / sr[i]
    nz[m] = (zb[m] - sz[i]) / sr[i]
    az = rng.uniform(0, 2 * np.pi)
    el = np.deg2rad(rng.uniform(35, 75))
    L = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    lam = np.clip(nx * L[0] - ny * L[1] + nz * L[2], 0, 1)  # image y points down
    rim = 1 - np.clip(nz, 0, 1)
    img = np.full((H, W), rng.uniform(0.05, 0.15))
    if style == "se":
        small = np.zeros((H, W))
        small[m] = np.clip(rng.uniform(4, 10) / sr[i], 0, 1)  # small spheres bright
        val = (rng.uniform(0.25, 0.4) + rng.uniform(0.35, 0.55) * lam
               + rng.uniform(0.15, 0.35) * rim ** 2 + rng.uniform(0.1, 0.3) * small)
    else:
        val = rng.uniform(0.55, 0.7) + rng.uniform(0.1, 0.2) * lam - rng.uniform(0.15, 0.3) * rim ** 3
    # per-particle albedo jitter and surface texture
    alb = rng.uniform(0.9, 1.1, size=len(spheres))
    tex = cv2.GaussianBlur(rng.normal(size=(H, W)), (0, 0), rng.uniform(1.0, 2.5))
    tex = tex / (tex.std() + 1e-9) * rng.uniform(0.01, 0.035)
    img[m] = val[m] * alb[i] + tex[m]
    # substrate: low-frequency mottling + contact darkening near particles
    bg = cv2.GaussianBlur(rng.normal(size=(H, W)), (0, 0), 20)
    bg = bg / (bg.std() + 1e-9) * 0.02
    dist_to_p = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
    shadow = np.exp(-dist_to_p / rng.uniform(2, 6)) * rng.uniform(0.0, 0.06)
    img[~m] = img[~m] + bg[~m] - shadow[~m]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.0))
    img = img + rng.normal(scale=rng.uniform(0.01, 0.04), size=img.shape)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def render(seed):
    rng = np.random.default_rng(seed)
    spheres = make_scene(rng)
    zb, ids, full = zbuffer(spheres)
    style = "se" if seed % 2 == 0 else "bse"
    img = shade(spheres, zb, ids, rng, style)
    vis = np.bincount(ids[ids >= 0], minlength=len(spheres))
    for k, s in enumerate(spheres):
        s["full_px"] = int(full[k])
        s["vis_px"] = int(vis[k])
        s["vis_frac"] = float(vis[k] / full[k]) if full[k] else 0.0
        s["gt_satellite"] = bool(s["kind"] == "satellite" and s["vis_frac"] >= 0.4 and s["r"] >= 2.0
                                 and 0 <= s["x"] < W and 0 <= s["y"] < H)
    return img, ids, spheres, style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/synth_sat")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(a.n):
        img, ids, spheres, style = render(seed)
        cv2.imwrite(str(out / f"sat_{seed:03d}.png"), img)
        np.save(out / f"sat_{seed:03d}_ids.npy", ids.astype(np.int16))
        json.dump({"seed": seed, "style": style, "spheres": spheres}, open(out / f"sat_{seed:03d}.json", "w"))
        n_sat = sum(s["gt_satellite"] for s in spheres)
        print(f"sat_{seed:03d} {style} parents={sum(s['kind'] == 'parent' for s in spheres)} gt_satellites={n_sat}")


if __name__ == "__main__":
    main()

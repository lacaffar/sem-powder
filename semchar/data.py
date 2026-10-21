"""Fetch every dataset the pipeline uses into data/ (idempotent: skips what exists).

  data/aa7075/     Mendeley 42gzbgk5hb v3 (CC BY 4.0): Fig 2a/2c/2d SEM TIFFs + PSD.CSV (laser)
  data/alloy2/     Zenodo 17037765 (CC BY-SA 4.0): Ti-6Al-4V, Cu, Ti64+10/15/20Cu TESCAN images + PSD workbook
  data/synthetic/  DeCost & Holm, Mendeley tj4syyj9mr v1 (CC BY 4.0): renders 1-24 per PSD + JSON ground truth
  data/synth_dense/ dense monolayer renders with exact instance masks (src: semchar/synth_render.py)

  python -m semchar data
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path

from .download import fetch, get_json, sha256_of

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DECOST_URL = ("https://data.mendeley.com/public-files/datasets/tj4syyj9mr/files/"
              "ffe2eb7c-ba3b-491f-9c51-298bb40c647f/file_downloaded")
DECOST_SHA = "ef4d4a15a4596566d759b7964a145dba45edff44d86145e1d056e5eeccf7f63d"
DECOST_KEEP = 24
ZENODO = "https://zenodo.org/records/17037765/files/{}?download=1"


def log(msg):
    print(f"[data] {msg}", flush=True)


def mendeley_files(ds, ver):
    folders = get_json(f"https://data.mendeley.com/public-api/datasets/{ds}/folders/{ver}")
    fmap = {f["id"]: f for f in folders}

    def path(fid):
        p = []
        while fid in fmap:
            p.append(fmap[fid]["name"])
            fid = fmap[fid].get("parent_id")
        return "/".join(reversed(p))

    out = []
    for fid in ["root"] + list(fmap):
        for f in get_json(f"https://data.mendeley.com/public-api/datasets/{ds}/files?folder_id={fid}&version={ver}"):
            out.append({"path": (path(fid) + "/" if fid != "root" else "") + f["filename"],
                        "url": f["content_details"]["download_url"], "size": f["size"]})
    return out


def get_aa7075():
    d = DATA / "aa7075"
    want = ["Original_Image_for_Fig_2a.tif", "Original_Image_for_Fig_2c.tif",
            "Original_Image_for_Fig_2d.tif", "PSD.CSV", "ReadMe_PSD.txt"]
    if all((d / w).exists() for w in want):
        log("aa7075 present")
    else:
        log("aa7075: listing Mendeley 42gzbgk5hb v3")
        files = mendeley_files("42gzbgk5hb", 3)
        for f in files:
            name = f["path"].split("/")[-1]
            if name in want and not (d / name).exists():
                log(f"  {name} ({f['size'] / 1e6:.1f} MB)")
                fetch(f["url"], d / name)
    shutil.copy(ROOT / "configs" / "scales_aa7075.csv", d / "scales.csv")  # Fig 2c has no SEM metadata


def get_alloy2():
    d = DATA / "alloy2"
    tifs = ["Ti-6Al-4V.tif", "Cu.tif", "Ti64+10Cu.tif", "Ti64+15Cu.tif", "Ti64+20Cu.tif"]
    if all((d / t).exists() for t in tifs) and (d / "Table2_ParticleSizeDistribution.xlsx").exists():
        log("alloy2 present")
        return
    d.mkdir(parents=True, exist_ok=True)
    for z, keep in [("Particle_shape.zip", ".tif"), ("Particle_size_distribution.zip", ".xlsx")]:
        log(f"alloy2: {z}")
        tmp = DATA / "_dl" / z
        fetch(ZENODO.format(z), tmp)
        with zipfile.ZipFile(tmp) as zf:
            for n in zf.namelist():
                if n.endswith(keep):
                    (d / Path(n).name).write_bytes(zf.read(n))
        tmp.unlink()


def get_decost():
    d = DATA / "synthetic"
    have = list(d.glob("*/renders/particles*.png"))
    if len(have) >= 8 * DECOST_KEEP:
        log("synthetic (DeCost) present")
        return
    tmp = DATA / "_dl" / "decost.zip"
    if not tmp.exists() or sha256_of(tmp) != DECOST_SHA:
        log("synthetic: downloading DeCost & Holm zip (585 MB)")
        fetch(DECOST_URL, tmp, sha256=DECOST_SHA)
    log(f"synthetic: extracting renders 1-{DECOST_KEEP} per PSD + JSON")
    with zipfile.ZipFile(tmp) as zf:
        for n in zf.namelist():
            if n.startswith("__MACOSX") or n.endswith("/") or ".DS_Store" in n:
                continue
            rel = n.split("/", 1)[1]
            m = re.search(r"(renders|particles)/\D*(\d+)\.(png|json)$", rel)
            if m and int(m.group(2)) > DECOST_KEEP:
                continue
            out = d / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(n))
    tmp.unlink()


def get_synth_dense(n=40):
    d = DATA / "synth_dense"
    if len(list(d.glob("sat_*_ids.npy"))) >= n:
        log("synth_dense present")
        return
    log(f"synth_dense: rendering {n} dense monolayer images with exact masks")
    import cv2
    import json
    import numpy as np
    from .synth_render import render
    d.mkdir(parents=True, exist_ok=True)
    for seed in range(n):
        img, ids, spheres, style = render(1000 + seed)  # seeds disjoint from attempt 1's satellite set
        cv2.imwrite(str(d / f"sat_{seed:03d}.png"), img)
        np.save(d / f"sat_{seed:03d}_ids.npy", ids.astype(np.int32))
        json.dump({"seed": 1000 + seed, "style": style, "spheres": spheres}, open(d / f"sat_{seed:03d}.json", "w"))


def main():
    DATA.mkdir(exist_ok=True)
    for f in (get_aa7075, get_alloy2, get_decost, get_synth_dense):
        try:
            f()
        except Exception as e:  # keep going; the run reports what is missing
            log(f"FAILED {f.__name__}: {e}")
    shutil.rmtree(DATA / "_dl", ignore_errors=True)
    log("done")


if __name__ == "__main__":
    main()

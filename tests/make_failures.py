"""Save annotated failure examples to figures/failures/."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imio import load_gray  # noqa: E402
from src.segment import overlay, segment  # noqa: E402
from src.synthetic_gt import ground_truth  # noqa: E402

OUT = ROOT / "figures/failures"


def caption(img, lines, scale=2):
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    bar = np.full((22 * len(lines) + 10, img.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    aa = ROOT / "data/aa7075"
    i2d = load_gray(aa / "Original_Image_for_Fig_2d.tif")
    lab, _ = segment(i2d, "watershed")
    ov = overlay(i2d, lab)
    cv2.imwrite(str(OUT / "fail1_2d_watershed_shaded_flanks_edge_loss.png"), caption(ov, [
        "Fig 2d, watershed: shaded lower flanks fall below the foreground threshold -> crescent-cut masks (undersize).",
        "Large spheres touching the frame are excluded (grey, unlabelled): only 29 particles kept, D90 -41% vs laser.",
    ], scale=1))

    lab, _ = segment(i2d, "cellpose")
    ov = overlay(i2d, lab)[0:260, 600:920]
    cv2.imwrite(str(OUT / "fail2_2d_cellpose_background_false_positive.png"), caption(ov, [
        "Fig 2d, Cellpose: straight-edged mask on substrate",
        "between spheres (false positive).",
    ]))

    i2a = load_gray(aa / "Original_Image_for_Fig_2a.tif")
    lab, _ = segment(i2a, "watershed")
    ov = overlay(i2a, lab)[100:330, 40:330]
    cv2.imwrite(str(OUT / "fail3_2a_watershed_agglomerate_merge.png"), caption(ov, [
        "Fig 2a, watershed: satellite cluster / agglomerate",
        "kept as one object (oversize, low solidity).",
    ]))

    syn = ROOT / "data/synthetic/h-weibull_fit-loc0.12-shape0.6"
    img = load_gray(syn / "renders/particles1.png")
    lab, _ = segment(img, "watershed")
    gt, ids = ground_truth(syn / "particles/particles1.json")
    ov = overlay(img, lab, alpha=0.25)
    b = cv2.morphologyEx((ids + 1).astype(np.uint16), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ov[b] = (0, 0, 255)
    cv2.imwrite(str(OUT / "fail4_synthetic_overlap_merge.png"), caption(ov[0:256, 0:256], [
        "Synthetic PSD h, watershed (fill) vs ray-cast GT (red):",
        "overlapping spheres merged into one label -> missed",
        "small particles, D50 biased high (+19% on PSD h).",
    ], scale=3))
    a2 = ROOT / "data/alloy2"
    if (a2 / "Ti-6Al-4V.tif").exists():
        from src.scale import detect_databar, find_scale_bar
        img = load_gray(a2 / "Ti-6Al-4V.tif")
        _, y0, y1 = detect_databar(img)
        bar = find_scale_bar(img[y0:y1])
        vis = cv2.cvtColor(img[y0 - 20:y1], cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (bar[0], 20 + bar[2]), (bar[1], 20 + bar[2]), (0, 0, 255), 2)
        cv2.imwrite(str(OUT / "fail5_tescan_ocr_um_read_as_Hm.png"), caption(vis, [
            "Real TESCAN data bar (Ti-6Al-4V, Zenodo 17037765): bar found (red, 187 px) but OCR read the label as '100 Hm'.",
            "Before the fix every image in the set was left uncalibrated (sizes in px). Fixed: any 1-2 letters + 'm' (not n/m) = um.",
            "Remaining scale error -1.1% vs 'View field: 553.5 um' (bar length includes the end ticks).",
        ], scale=1))
        img = load_gray(a2 / "Cu.tif")
        lab, _ = segment(img[:768], "watershed")
        ov = overlay(img[:768], lab)[0:330, 0:420]
        cv2.imwrite(str(OUT / "fail6_cu_fines_clusters.png"), caption(ov, [
            "Cu powder, watershed: clumps of fines kept as one",
            "irregular particle; isolated fines < 5 px dropped.",
            "D10 +41% vs reported PSD.",
        ]))
    print("saved", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

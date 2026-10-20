# PROGRESS

## Summary (read this first)
1. Blocks 1-2 done. Env works (`check_env.py` prints ok). Both datasets downloaded automatically.
2. Scale: AA7075 TIFFs have NO data bar; µm/px read from Zeiss TIFF metadata (0.3722 µm/px). OCR scale-bar path tested on 5 rendered data bars: max error 0.25%.
3. (pending) Segmentation, metrics, synthetic validation, real-image check, CLI, one-pager.
4. Manual items for you: put Ti-6Al-4V / 316L images in `data/alloy2/` if you want Block 6 alloy-2 run.
5. Disk: 12 GB free at start. Data on disk ~270 MB.

## Block 1: Env + data (DONE)
- Python 3.13.2, torch 2.14.1+cpu (already present), cellpose 3.0.9, easyocr 1.7.2, skimage 0.26, opencv-python-headless 5.0. Pinned in `requirements.txt`.
- Decision: `--only-binary=:all:` needed. A plain install tried to build an old numpy from source (no compiler). Binary-only install resolved cleanly.
- Decision: opencv-python-headless instead of opencv-python (easyocr requires headless; avoids two cv2 builds).
- Gotcha: easyocr's first-run model download crashes on a cp1252 Windows console (progress bar uses U+2588). Fix: `PYTHONIOENCODING=utf-8` on first run. Models now cached in `~/.EasyOCR`, `~/.cellpose`.
- Mendeley API: Python `requests` gets a Cloudflare 403; `curl` works. Downloads done with curl.
- Synthetic (DeCost & Holm): zip 585 MB, sha256 verified, extracted README + 3 scripts + all 2048 JSON + renders 1-50 per PSD (400 PNG). Zip deleted. 266 MB on disk.
- AA7075 (Mendeley 42gzbgk5hb v3): `Original_Image_for_Fig_2{a,b,c,d}.tif` + `PSD.CSV` + ReadMe.
  - 2a = base 7075 (Zeiss 300x, 0.3722 µm/px).
  - 2b = nano TiC (100kx). Skipped per plan.
  - 2c = TiC-modified (looks like surface/"BM" modification: base morphology + satellites). Photoshop re-save, no SEM metadata, cropped to 691 rows. ASSUMPTION: same 300x / 0.3722 µm/px as 2a/2d (same frame width, similar particle sizes). Stored in `data/aa7075/scales.csv`.
  - 2d = TiC-modified, coarser wrinkled spheres (looks like the "VM" volume-modified powders, which have larger reported D50). Zeiss 300x, 0.3722 µm/px.
  - Fig 3 TIFFs are single-particle EDS context images. Downloaded, inspected, deleted (no PSD value).
- Reported laser-diffraction PSD (volume-weighted, Malvern Hydro 2000S, µm):

| Sample | D10 | D50 | D90 |
|---|---|---|---|
| 7075 | 26.0 | 44.4 | 72.9 |
| 7075 0.3wt%TiC BM | 22.1 | 38.8 | 65.0 |
| 7075 1.0wt%TiC BM | 22.0 | 38.7 | 64.6 |
| 7075 2.0wt%TiC BM | 24.3 | 41.6 | 68.1 |
| 7075 0.3wt%TiC VM | 42.8 | 69.2 | 110.9 |
| 7075 1.0wt%TiC VM | 47.1 | 73.7 | 126.4 |

- `data/alloy2/` is empty. Second alloy will be skipped unless images are added.

## Block 2: Scale calibration (DONE)
- `src/scale.py`: data-bar detection (rows dominated by one flat value, touching top/bottom edge), scale bar = longest thin horizontal component after horizontal morphological opening, easyocr label nearest the bar, regex number+unit with OCR-confusion fixes (O->0, l->1, 'pum'/'pm' -> µm), merges split tokens ("0.1" + "mm"), ignores WD/HFW/EHT/Mag fields.
- Resolution order: CLI `--um-per-px` > `scales.csv` sidecar > SEM TIFF metadata (Zeiss CZ_SEM `ap_width`/`ap_image_pixel_size`, FEI `PixelWidth`) > scale-bar OCR > uncalibrated (px).
- Decision: added TIFF-metadata step (not in plan). The real AA7075 images have no data bar, so without it they would be uncalibrated. When both OCR and metadata exist, metadata wins and the OCR disagreement is logged in notes.
- Test (`tests/make_scalebar_tests.py`, `tests/test_scale.py`): real Fig 2a frame + rendered Zeiss/FEI/JEOL-style data bars, black/white/gray backgrounds, labels 100 µm, 200 µm, 50 μm, 100μm, 0.1 mm, 0.5x-2x resampling.

| Test image | true µm/px | OCR µm/px | error |
|---|---|---|---|
| zeiss_black_100um | 0.3722 | 0.3717 | -0.12% |
| zeiss_white_200um | 0.3722 | 0.3724 | +0.06% |
| fei_gray_50um_half | 0.1861 | 0.1866 | +0.25% |
| jeol_black_100um_small | 0.4963 | 0.4950 | -0.25% |
| zeiss_black_0p1mm | 0.3722 | 0.3717 | -0.12% |

- Limitation: OCR path not yet tested on a real instrument data bar (none in the available data). Debug images in `out/scale_debug/`.
- Sanity check vs. PSD: a typical large particle in Fig 2a is ~130 px = 48 µm; reported D50 = 44 µm. Consistent.

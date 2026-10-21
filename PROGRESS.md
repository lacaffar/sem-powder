# PROGRESS

## Summary (read this first)
1. All 8 blocks done. Demo: `python analyze.py samples --out out/demo` (watershed, ~1 s/image). One-pager: `ONE_PAGER.md`. Figures 1-4 + failures in `figures/`.
2. Synthetic accuracy (exact 3D GT, 8 PSDs): watershed number D50 +12% (range +7 to +19%, always high), volume D50 +12%; Cellpose volume D50 6% abs. D90 ranks all 8 PSDs correctly (ρ = 1.0).
3. Real AA7075 vs laser diffraction (volume, edge-corrected, watershed): D50 -14% (base), -8% (surface-mod TiC), -25% (volume-mod TiC, only 29 particles). D90 -17 to -42%: one 300x frame is too small a sample for D90.
4. Alloy 2 (Ti-6Al-4V, public Zenodo set, TESCAN SEM, real data bar), unchanged settings: D50 +8%, D90 -1% vs reported PSD. First real scale bar broke OCR ("µm" read as "Hm"); fixed, now -1.1% vs the bar's view field.
5. Satellites: new detector, held-out real F1 0.48 (was 0: not detected), synthetic F1 0.67, share-with-satellites error 67 -> 11 pts. Skipped: Rettenberger model (needs Detectron2 compile, no compiler), 316L (no public SEM + PSD set found). Cellpose validated on 16 img/PSD, not 50 (CPU time). Keep the project at a short path on the lab PC (Windows path-length limit breaks the torch install).

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

- `data/alloy2/` was empty; filled later from Zenodo 17037765 (see Alloy 2 at the end).

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

- Real instrument bar: tested later on TESCAN Vega (see Alloy 2 at the end). Debug images in `out/scale_debug/`.
- Sanity check vs. PSD: a typical large particle in Fig 2a is ~130 px = 48 µm; reported D50 = 44 µm. Consistent.

## Block 3: Segmentation (DONE)
- `src/segment.py`. Preprocess: grayscale, CLAHE (tile ~ image/128), percentile-normalize to [0,1].
- Foreground: lower threshold of 3-class multi-Otsu (substrate | shaded flanks | bright tops). Plain Otsu cut spheres in half (shaded lower flanks lost). Synthetic foreground IoU vs ray-cast GT: 0.94-0.96 (multi-Otsu) vs 0.85-0.89 (Otsu).
- Watershed: markers = h-maxima of (distance transform / r_est + brightness smoothed at 0.25 r_est), h = 0.05. r_est = median max-inscribed radius of foreground components (image-derived, no alloy constants).
  - Distance-only markers merged overlapping spheres: recall 0.61-0.69 on tuning renders.
  - Brightness unsmoothed: synthetic recall 0.83-0.85, but shattered textured real spheres (659 fragments on Fig 2d). Smoothing 0.25 r_est: synthetic recall 0.77, precision 0.99, real images clean. Chose the setting that works on real powder.
  - Tuning used only synthetic renders 49-50 per PSD; validation (Block 5) uses renders 1-48.
- Cellpose cyto3 (cellpose 3.0.9), `channels=[0,0]`, on the preprocessed image. Two variants: `cellpose` (diameter=None, auto) and `cellpose_ws` (diameter = watershed median ECD). CPU time: 6-18 s (synthetic 512^2), 10-36 s (real 1024x768). `cellpose_ws` is ~2x faster (skips the size model).
- Post: `clear_border` + remove < 20 px area.
- Visual: Cellpose captures shaded flanks and separates satellites on AA7075; watershed leaves partial crescents on Fig 2d shaded sides.
- Rettenberger Mask R-CNN: SKIPPED. Needs Detectron2, which has no Windows wheels and must be compiled; no C/C++ compiler on this machine (numpy source build failed for the same reason). Not worth risking the venv.
- Overlays: `out/overlays/` (2 synthetic + 2 real, 3 methods).
- `src/synthetic_gt.py` (ray-cast ground truth) written early to tune against; see Block 5.

## Block 4: Metrics (DONE)
- `src/metrics.py`: per particle area_px, area_um2, ECD (px, µm), perimeter, circularity 4πA/P² (clipped at 1), aspect ratio (major/minor), solidity, centroid.
- Per image: count, D10/D50/D90 number- and volume-weighted (w ∝ ECD³, midpoint-CDF interpolation), mean circularity/aspect/solidity.
- Satellite proxy: `irregular_frac` = fraction with solidity < 0.90 (coarse: satellites attached or agglomerates).
- `analyze_image(path, method, um_per_px=None) -> (df, summary, labels, img, scale)` and `plot_psd()`.
- First real number, Fig 2a cellpose: D10/D50/D90 vol = 17.9/32.7/53.3 µm vs laser 26.0/44.4/72.9 (D50 -26%). See Block 6.

## Block 5: Synthetic validation (DONE)
- `src/synthetic_gt.py`, `src/validate_synthetic.py` -> `out/synthetic_validation.csv` (+ `_summary.csv`, `_first16.csv`), `figures/fig2_synthetic_psd.png`, `figures/fig3_d50_parity.png`.
- Camera geometry (from `blender_powder.py` + `blend.sh`; intrinsics are Blender 2.7x defaults since the script sets none): perspective camera at (0,0,10) looking down -z; 35 mm lens, 32 mm sensor; 1024^2 render at default 50% -> 512^2 PNG; f = 35/32 x 512 = 560 px; px/unit = 560 / (10 - z) = 56.0 at z=0, 50.9-62.2 over the particle box (z in [-1, 1]). Sphere `size` = radius. World xyz = [11,11,2] x json - [5.5,5.5,1].
- Check: ray-cast foreground of the JSON spheres vs thresholded render, IoU 0.92; outlines sit on the rendered spheres (visual).
- Visible GT: per-pixel ray-sphere z-buffer gives exact visible-pixel counts; keep spheres > 50% visible and not touching the frame. Sparse scenes: visible D50 = generating D50 within 1%.
- Measured px -> units with ONE nominal scale (median px/unit of visible GT spheres = 56.0, what a calibrated SEM gives). Exact per-sphere px comparison also in CSV (`*_px` columns): same numbers within 1 pt.
- Matching: Hungarian on centroids, tolerance max(3 px, 0.5 x GT diameter).
- Images: watershed renders 1-48 per PSD (383; `a/particles37.json` is truncated upstream, skipped). Cellpose and cellpose_ws renders 1-16 per PSD (127 each). Deviation from ~50/PSD: Cellpose ran 20-60 s/image on this CPU (two parallel runs throttled each other), full set would have taken ~6 h. Renders 49-50 were the tuning set and are excluded.
- Results, watershed (48 img/PSD) and Cellpose auto-diameter (16 img/PSD), errors vs visible GT:

| PSD | true D50 vis / gen | WS D50 | WS err vis / gen | WS D10 / D90 err | WS vol D50 err | WS recall | CP D50 err | CP vol D50 err | CP recall |
|---|---|---|---|---|---|---|---|---|---|
| a | 0.200 / 0.200 | 0.223 | +11.6% / +11.5% | +19.1% / +11.4% | +11.3% | 0.80 | +19.4% | +3.4% | 0.86 |
| b | 0.199 / 0.199 | 0.230 | +15.2% / +15.7% | +33.3% / +12.3% | +9.8% | 0.74 | +23.7% | -3.1% | 0.79 |
| c | 0.201 / 0.200 | 0.215 | +7.3% / +7.4% | +10.4% / +12.0% | +11.0% | 0.89 | +13.6% | +11.4% | 0.95 |
| d | 0.201 / 0.201 | 0.216 | +7.5% / +7.7% | +6.0% / +11.4% | +11.9% | 0.89 | +14.7% | +11.6% | 0.96 |
| e | 0.242 / 0.241 | 0.264 | +9.1% / +9.1% | +15.1% / +11.1% | +12.5% | 0.77 | +14.9% | +4.3% | 0.82 |
| f | 0.241 / 0.240 | 0.274 | +13.6% / +14.2% | +24.3% / +14.1% | +10.7% | 0.72 | +20.7% | -8.6% | 0.77 |
| g | 0.254 / 0.254 | 0.289 | +13.7% / +14.1% | +40.1% / +11.1% | +11.5% | 0.74 | +15.3% | +5.1% | 0.79 |
| h | 0.259 / 0.259 | 0.308 | +19.0% / +18.9% | +59.4% / +13.8% | +13.7% | 0.69 | +25.4% | +2.3% | 0.73 |

- Means over 8 PSDs (|error| vs visible GT):

| Method | num D50 | num D10 | num D90 | vol D50 | recall | precision | matched size ratio | s/img |
|---|---|---|---|---|---|---|---|---|
| watershed | 12.1% | 25.9% | 12.2% | 11.5% | 0.78 | 0.99 | 1.02 | 0.3 |
| cellpose | 18.5% | 44.9% | 10.6% | 6.2% | 0.83 | 0.97 | 1.10 | 20 |
| cellpose_ws | 18.1% | 46.1% | 10.4% | 6.9% | 0.84 | 0.97 | 1.10 | 14 |

- Same-subset check (renders 1-16, all methods): watershed num D50 12.2%, vol D50 11.5%; cellpose 18.5% / 5.8%. Conclusions unchanged.
- Ranking (Spearman ρ, 8 PSDs): number D50 0.76 vs visible (0.81 vs generating) for all methods; volume D50 1.00 (watershed), 0.98 (Cellpose); D90 1.00 for all. The D50 ρ < 1 is only within ties: PSDs a-d have identical true D50 (0.200) and e-h near-identical (0.24-0.26). Every method separates the two groups with no overlap (watershed: max a-d 0.230 < min e-h 0.264).
- Bias: number D50 always reads high (+7 to +19%). Cause: overlapping spheres merge (small ones lost; recall 0.69-0.89) and 0-10% of visible spheres are below the 5 px ECD floor. Against resolvable GT (ECD >= 5 px) watershed D50 error drops to 8.0%. Matched particles are sized accurately by watershed (ratio 1.02); Cellpose masks are ~10% oversized on these ~11 px particles.
- Decision: keep both. Watershed default (best number-weighted, 70x faster). Cellpose better for volume-weighted D50 on synthetic. No synthetic-derived correction factor applied (would not transfer to real images).

## Block 6: Real-image check (DONE)
- `src/real_check.py` -> `out/aa7075_vs_laser.csv`, `figures/fig4_aa7075_vs_reported.png`, overlays in `out/real/`.
- Image -> laser mapping (inferred from morphology and size; dataset does not label Fig 2 panels in the files): 2a = base 7075; 2c = TiC surface-modified ("BM", compared to mean of 0.3/1.0/2.0 wt% BM); 2d = TiC volume-modified ("VM", mean of 0.3/1.0 wt% VM).
- Added Miles-Lantuéjoul edge correction (`*_vol_ec`): edge-touching particles are excluded, which under-counts large ones; each kept particle weighted by frame area / area where it fits without touching the edge. Standard stereology, not tuned.
- Volume-weighted, edge-corrected vs laser (µm):

| Image | Method | n | D10 img / laser | D50 img / laser | D90 img / laser | D50 err |
|---|---|---|---|---|---|---|
| 2a base | watershed | 176 | 21.8 / 26.0 | 38.4 / 44.4 | 60.3 / 72.9 | -13.5% |
| 2a base | cellpose | 200 | 18.5 / 26.0 | 33.7 / 44.4 | 53.9 / 72.9 | -24.1% |
| 2c BM | watershed | 271 | 17.8 / 22.8 | 36.5 / 39.7 | 46.3 / 65.9 | -8.1% |
| 2c BM | cellpose | 302 | 15.9 / 22.8 | 33.7 / 39.7 | 45.3 / 65.9 | -15.1% |
| 2d VM | watershed | 29 | 43.7 / 44.9 | 53.3 / 71.4 | 69.3 / 118.6 | -25.4% |
| 2d VM | cellpose | 22 | 40.8 / 44.9 | 55.2 / 71.4 | 70.0 / 118.6 | -22.8% |

- Systematic bias: image under-reads laser, worst at D90 (-17 to -41%). Causes, in order of size:
  1. Field of view. One 381 x 286 µm frame per powder. The image CDF ends at 60-70 µm; laser extends past 100 µm. 2d holds only 22-29 whole particles; large ones touch the border and are dropped. Edge correction recovers only a few %, since there are almost no large particles left to up-weight.
  2. Laser diffraction reads high on non-spherical / agglomerated powder (equivalent-sphere model); the 1.0 VM sample has ~6% volume at 1-3 mm (agglomerates) that no 300x frame can contain.
  3. Partly buried particles (2a background layer) read small; satellite clusters read as one (watershed).
- Watershed is closer to laser than Cellpose on 2a and 2c; Cellpose masks are slightly tighter on real particles and split satellite clusters more, which lowers volume-weighted sizes.
- Alloy 2: done after hand-off, see Alloy 2 at the end. 2c/2d (TiC-modified) also ran unchanged.
- Failure examples in `figures/failures/`: (1) 2d watershed loses shaded flanks + edge loss of large spheres; (2) 2d Cellpose false-positive mask on substrate; (3) 2a watershed merges an agglomerate; (4) synthetic overlapping spheres merged.

## Block 7: CLI tool (DONE)
- `python analyze.py <folder|image> [--method watershed|cellpose|cellpose_ws] [--um-per-px X] [--scales csv] [--out dir]`. Writes results.csv, particles/, psd/, overlays/, scale_debug/, psd_all.png. Continues past unreadable files.
- Default method: watershed (best synthetic D50, closer to laser on 2 of 3 real frames, ~1 s vs 15-60 s per image on CPU).
- Bug found and fixed: even-sized morphology kernel left one data-bar row in the crop, which shielded bottom-edge particles from `clear_border` (185 vs 176 particles). Kernel now odd; data-bar and metadata versions of Fig 2a now give identical results.
- `samples/`: Fig 2a TIFF (metadata scale) + same frame with a rendered data bar (OCR scale) + `expected/results.csv`.
- Fresh clone + fresh venv + `pip install --only-binary=:all: -r requirements.txt` + `analyze.py samples`: results.csv matches expected exactly (timing column excluded). Gotcha found: install fails under a long path (Windows 260-char limit, torch headers). README says to keep the project at a short path.

## Block 8: Figures + one-pager (DONE)
- `figures/fig1_overlays.png` (raw | watershed | Cellpose; synthetic PSD e crop + AA7075 Fig 2a), `fig2_synthetic_psd.png` (CDFs, PSDs a/c/h), `fig3_d50_parity.png` (number and volume D50 parity, all PSDs, 3 methods), `fig4_aa7075_vs_reported.png` (image vs laser volume CDFs), `failures/` (4 annotated examples).
- `ONE_PAGER.md`: accuracy tables first, inputs/outputs, limits, ask (20-30 CAMAL powder SEM images + laser PSD of the same lot if available).
- Regenerate: `python -m src.validate_synthetic report`, `python -m src.real_check`, `python tests/make_fig1.py`, `python tests/make_failures.py`.

## Alloy 2: Ti-6Al-4V (DONE, after hand-off)
- Search: no public 316L powder SEM + PSD dataset found (Zenodo/Mendeley API + web search; only paper figures). Ti-6Al-4V found: Zenodo 17037765 (Strauch, Altmann, Tönjes, IWT Bremen; CC BY-SA 4.0). Gas-atomized Ti-6Al-4V (LPW) plus Cu and Ti64+10/15/20 wt% Cu blends. One TESCAN Vega II SEM image each (BSE, 15 kV, 1024 x 864 with data bar) + PSD workbook (3 runs + mean, d10/d50/d90). Downloaded only `Particle_shape.zip` (3.5 MB) and `Particle_size_distribution.zip`. Not committed (data/ is gitignored; CC BY-SA).
- PSD method not stated in the record. Log bins from 0.012 µm suggest laser diffraction; compared with volume-weighted, edge-corrected image stats.
- Run 1, fully unchanged: data bar found and cropped (rows 768-864), scale bar found (187 px), but OCR read the label as "100 Hm". Every image left uncalibrated (px). First real instrument data bar the tool has seen.
- Fix (general, not alloy-specific): unit regex accepts any 1-2 letters other than n/m before "m" as µm (SEM bars never use km/cm). Rendered-bar tests still pass (max 0.25%).
- Scale after fix: 0.5348 µm/px on all 5 images vs 553.5 µm view field / 1024 px = 0.5405. Error -1.1% (measured bar 187 px vs expected 185; end ticks included).
- Segmentation unchanged (no parameter edits). `src/alloy2_check.py` -> `out/alloy2_vs_reported.csv`, `figures/fig5_ti64_vs_reported.png`, overlays in `out/alloy2/`.

| Image | Method | n | Image D10/D50/D90 (µm) | Reported | Error D10 / D50 / D90 |
|---|---|---|---|---|---|
| Ti-6Al-4V | watershed | 227 | 22.0 / 32.5 / 41.7 | 20.6 / 30.1 / 42.3 | +6.9% / +8.2% / -1.4% |
| Ti-6Al-4V | cellpose | 289 | 22.7 / 34.1 / 43.8 | 20.6 / 30.1 / 42.3 | +10.5% / +13.5% / +3.6% |
| Ti64+10Cu | watershed | 230 | 23.8 / 39.3 / 49.3 | 20.5 / 32.3 / 52.1 | +15.8% / +21.7% / -5.4% |
| Ti64+15Cu | watershed | 238 | 21.8 / 34.6 / 40.3 | 20.5 / 31.2 / 46.0 | +6.5% / +11.0% / -12.4% |
| Ti64+20Cu | watershed | 340 | 19.4 / 33.2 / 49.9 | 20.1 / 31.8 / 48.6 | -3.6% / +4.4% / +2.6% |
| Cu | watershed | 578 | 20.8 / 39.9 / 47.7 | 14.7 / 34.4 / 48.7 | +41.4% / +15.8% / -2.0% |

- Ti-6Al-4V within 8% at D50 and 1% at D90 with zero tuning. Opposite sign to AA7075 (which read low): these frames hold ~230 particles of ~56 px in a 553 µm field, so D90 is sampled well. Supports field of view as the main AA7075 error.
- Bias left: image D50 reads high (+4 to +22%), as on synthetic. The -1.1% scale error would push true sizes ~1% higher still.
- Cu: clumps of fines kept as one irregular particle and isolated fines < 5 px dropped: D10 +41%.
- Failure examples added: `figures/failures/fail5_tescan_ocr_um_read_as_Hm.png`, `fail6_cu_fines_clusters.png`.
- Possible next step: also parse "View field" / "HFW" from the data bar as a second scale source (field width / image width). Not done.

## Satellites (DONE, after hand-off)
- Before: no satellite detection. Only `irregular_frac` (solidity < 0.90). Watershed deliberately merges bonded satellites into the parent (right for the size distribution, wrong for satellite counting).
- Benchmark 1, synthetic (`src/synth_satellites.py`, 60 renders, `data/synth_sat/`, regenerable): orthographic z-buffer scenes of parent spheres on a substrate, satellites fused on the visible hemisphere incl. rims (r = 0.06-0.35 R, 20-80% embedded), free fines as negatives, SE- and BSE-like shading with randomised lighting/texture/noise. Seeds 0-29 tune, 30-59 test. GT = satellites >= 40% visible, r >= 2 px, on parents not cut by the frame (the tool drops those parents; 21% of satellites sit on them, reported separately as recall_all_gt).
- Benchmark 2, real (`tests/satellite_labels_real.json`): 8 fixed crops (positions chosen before looking), 4 images. Dev: AA7075 Fig 2a, Ti-6Al-4V (47 satellites). Test: AA7075 Fig 2c, Ti64+15Cu (36). Labelled by eye on 3x gridded crops by Claude; satellite = r <= ~0.3 host; 0.3-0.4 and dense agglomerates are ignore boxes. Tolerance max(4 px, r).
- Detector (`src/satellites.py`, method `model`): LoG blob candidates on a white top-hat image + the plain image (merged), restricted to particles (or within r of one), host radius >= 2.5 x blob radius; satellites split off by segmentation re-attached to the largest touching particle if <= 0.3 x its radius. 11 features per candidate; logistic regression (numpy, class-balanced, real rows x5) fit on synthetic tune + real dev; cut-off chosen on real dev (p = 0.97). `src/train_satellites.py`, coefficients `src/satellite_model.json`.
- Iterations (F1): blob threshold only 0.24-0.51 (synthetic tune) -> logistic on top-hat candidates 0.64 (synthetic tune) / 0.51 (real dev) -> + measurable-parent GT, split-off re-attachment 0.72 synthetic test -> + real dev training, host-centre feature 0.62 real dev -> + plain-image candidates, own-size rule for split-off particles 0.65 real dev.
- Held-out results:

| Test set | Before (solidity proxy) | Segmentation split only | Satellite detector |
|---|---|---|---|
| Real test crops (36) | F1 0.00 | F1 0.20 (P 0.33, R 0.14) | F1 0.48 (P 0.49, R 0.47), 35 found vs 36 |
| Synthetic test (1,508) | F1 0.00 | F1 0.04 | F1 0.67 (P 0.82, R 0.57), count 0.72x |
| Share of particles with satellites, synthetic test | 67 pts off | 62 pts off | 11 pts off |

- Trade-off: a synthetic-only model scored better on synthetic test (F1 0.72, count 0.99x, share error 5 pts) but worse on real dev (0.51). Kept the real-tuned model; real images are the target.
- By image (real test): Ti64+15Cu good (12 of 22 found, 12 of 17 detections right); TiC-modified AA7075 Fig 2c weak (5 of 14 found). Dev F1 kept rising (0.62 -> 0.65) while test stayed flat (0.474 -> 0.479): stopped tuning there to avoid fitting the labels.
- Real-image caveats: 83 labels from one non-expert annotator; ambiguity between a fused satellite and a small particle merely touching a larger one in 2D; test set is small (F1 uncertainty roughly +-0.1).
- Tool: on by default (~8 s/image), `--no-satellites` to skip. New outputs: `satellites/<image>.csv`; columns `satellite_count`, `satellites_per_particle`, `frac_with_satellites`, `satellite_D50` (results.csv) and `n_satellites` (particles). Size distribution unchanged. `figures/fig6_satellites.png`.
- Fig 2a: 122 satellites, 37% of particles carry one. Ti-6Al-4V: 94, 28%.
- Next levers if more accuracy is needed: more real labels (ideally by a metallographer, on CAMAL images), Cellpose host masks, or a small CNN trained on the synthetic renders + real labels.

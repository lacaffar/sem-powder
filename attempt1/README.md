# SEM Powder Characterization

Particle size and shape statistics from SEM images of metal AM powder. Any alloy, image only. CPU only.

## Install (Windows PowerShell, Python 3.11 to 3.13)
```
python -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: -r requirements.txt
python check_env.py   # prints "ok"; first run downloads Cellpose and EasyOCR models (~100 MB)
```
Keep the project at a short path (e.g. `C:\sem-powder`): torch has deeply nested files and pip fails with `OSError: [Errno 2]` past Windows' 260-character path limit unless long paths are enabled. (Linux/macOS: `source .venv/bin/activate`. On first run, if easyocr crashes with `UnicodeEncodeError`, set `PYTHONIOENCODING=utf-8`.)

## Run
```
python analyze.py <image_or_folder> [--method watershed|cellpose|cellpose_ws] [--um-per-px X] [--scales scales.csv] [--out out/]
```
Example: `python analyze.py samples --out out/demo`. Compare with `samples/expected/results.csv`.

Default method `watershed` (~1 s/image). `cellpose` (Cellpose 3 cyto3, zero-shot) takes 15 to 60 s/image on CPU; `cellpose_ws` seeds Cellpose with the watershed size estimate and is ~2x faster than `cellpose`.

## Scale (µm per pixel)
Resolved per image, first hit wins:
1. `--um-per-px X` (all images).
2. `scales.csv` next to the images (or `--scales`): rows `filename,um_per_px`.
3. SEM TIFF metadata (Zeiss `CZ_SEM`, FEI/Thermo `PixelWidth`).
4. Scale bar: data bar detected at bottom/top edge, scale bar = longest horizontal line in it, label read by EasyOCR (`nm`, `µm`, `um`, `mm`).
5. Otherwise the image is flagged uncalibrated and sizes are in px (`size_unit = px`).

The data bar is cropped off before segmentation. `scale_debug/<image>.png` shows what was detected.

## Outputs
| File | Content |
|---|---|
| `results.csv` | one row per image |
| `particles/<image>.csv` | one row per particle |
| `psd/<image>.png` | ECD histogram + number and volume CDFs |
| `overlays/<image>.png` | segmentation overlay, satellites as red circles (check this first) |
| `satellites/<image>.csv` | one row per detected satellite: centre x/y (px), radius r_px, d_um, host particle id, score |
| `scale_debug/<image>.png` | data bar / scale bar / OCR result |
| `psd_all.png` | CDFs of all images |

### `results.csv` columns
| Column | Meaning |
|---|---|
| `count` | particles measured (edge-touching and < 20 px area excluded) |
| `size_unit` | `um`, or `px` if uncalibrated |
| `um_per_px`, `scale_source`, `scale_confidence` | scale used; source = cli / sidecar / tiff_meta / scalebar / none |
| `D10_num`, `D50_num`, `D90_num` | number-weighted percentiles of ECD |
| `D10_vol`, `D50_vol`, `D90_vol` | volume-weighted percentiles (weight ∝ ECD³). Compare these with laser diffraction |
| `D10_vol_ec` ... `D90_vol_ec` | volume-weighted, Miles-Lantuéjoul edge-corrected (large particles touch the frame more often and are dropped; each kept particle is weighted by frame area / area where it could sit without touching the edge) |
| `mean_circularity` | mean 4πA/P² (1 = circle) |
| `mean_aspect_ratio` | mean major/minor axis of fitted ellipse |
| `mean_solidity` | mean area / convex hull area |
| `irregular_frac` | fraction with solidity < 0.90 (non-convex outline: agglomerates, attached particles) |
| `satellite_count` | satellites detected on measured particles |
| `satellites_per_particle` | satellite_count / particles with ECD >= 10 px (the smallest that can host a detectable satellite) |
| `frac_with_satellites` | share of those particles carrying >= 1 satellite |
| `satellite_D50` | median satellite diameter (µm, or px if uncalibrated) |
| `seconds` | segmentation time |

### `particles/<image>.csv` columns
`area_px`, `area_um2`, `ecd_px`, `ecd_um` (equivalent circle diameter, 2√(A/π)), `perimeter_px`, `circularity`, `aspect_ratio`, `solidity`, `centroid_x/y` (px), `edge_weight` (edge-correction weight), `n_satellites`.

`--no-satellites` skips satellite detection (~8 s per image faster). Particle sizes are the same either way: bonded satellites stay part of their parent in the size distribution, as in laser diffraction.

## Accuracy
See `ONE_PAGER.md` and `PROGRESS.md`. Synthetic renders with exact 3D ground truth (DeCost & Holm 2016): `src/validate_synthetic.py`. Real AA7075 vs laser diffraction: `src/real_check.py`.

## Satellites
Satellite = small particle fused to a larger one (radius up to ~0.4x the host's), on top or at the rim. Detector (`src/satellites.py`): blob candidates on a white top-hat image and the plain image, inside or touching segmented particles, each scored by a logistic regression on 11 features (blob strength, local contrast, circular edge support, size and position relative to the host). Coefficients in `src/satellite_model.json`, fit on synthetic renders (`src/synth_satellites.py`, seeds 0-29) plus hand-labelled real crops of AA7075 Fig 2a and Ti-6Al-4V (`tests/satellite_labels_real.json`).

| Held-out test | Before | Satellite detector |
|---|---|---|
| Real crops, AA7075 Fig 2c + Ti64+15Cu (36 labelled satellites) | F1 0.00 | F1 0.48 (precision 0.49, recall 0.47); 35 found vs 36 labelled |
| Synthetic renders, seeds 30-59 (1,508 satellites) | F1 0.00 | F1 0.67 (precision 0.82, recall 0.57) |
| Share of particles with satellites, synthetic | 67 pts off | 11 pts off |

Re-run: `python -m src.validate_satellites_real --split test`, `python -m src.validate_satellites --split test`. Retrain: `python -m src.train_satellites --real-dev`.

## Known limits
- 2D projection: ECD of a sphere's silhouette equals its diameter, but partly hidden particles read small and overlapping particles merge (read large, small ones lost). Synthetic D50 reads ~8 to 20% high from this.
- Resolution: particles under ~5 px ECD (20 px area) are dropped. Image at a magnification where the smallest particles of interest are > 10 px across.
- Field of view: D90 needs many large particles. Frames with < 100 particles, or where large particles touch the edge, under-read D90 (AA7075 at 300x: D90 -17 to -42% vs laser). Use several fields or lower magnification.
- Agglomerates and satellite clusters are measured as one particle (as laser diffraction also does).
- Dark substrate assumed (particles brighter than background, SE/BSE on carbon tape). Bright or textured substrates break the foreground step.
- Scale-bar OCR tested on one real instrument data bar (TESCAN Vega, -1.1% vs view field) and on rendered Zeiss/FEI/JEOL-style bars (max error 0.25%). Other fonts may misread. Always check `scale_debug/` or pass `--um-per-px`.
- Clumps of fine particles are measured as one particle (Cu powder: D10 +41%).
- Satellites: about half of the detections on real images are right and about half of the real satellites are found (held-out F1 0.48); counts are close because misses and false alarms roughly cancel. Weakest on irregular powders (TiC-modified AA7075 Fig 2c). Common false alarms: the bright top of a sphere, small whole particles touching a larger one. Real labels are one annotator's (Claude's), 83 satellites on 8 crops.

## Layout
`src/imio.py` loading, `src/scale.py` calibration, `src/segment.py` segmentation, `src/metrics.py` metrics, `src/synthetic_gt.py` + `src/validate_synthetic.py` synthetic validation, `src/real_check.py` AA7075 check, `tests/` checks and figure scripts.

## Data credits
- Synthetic: B. DeCost, E. Holm, Mendeley Data, doi:10.17632/tj4syyj9mr.1 (CC BY 4.0).
- AA7075 + TiC: Mendeley Data 42gzbgk5hb v3 (CC BY 4.0). `samples/aa7075_fig2a.tif` is their Fig. 2a original; `samples/aa7075_fig2a_databar.png` is the same frame with a rendered data bar for testing the OCR path.
- Ti-6Al-4V / Cu / blends: A. Strauch, M. L. Altmann, A. Tönjes (IWT Bremen), Zenodo doi:10.5281/zenodo.17037765 (CC BY-SA 4.0). Not redistributed here; `src/alloy2_check.py` expects `Particle_shape/*.tif` and `Table2_ParticleSizeDistribution.xlsx` in `data/alloy2/`.

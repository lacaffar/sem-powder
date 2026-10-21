# SEM Powder Characterization Tool: Build Plan

## Goal
A tool that takes SEM images of metal AM powder (any alloy, image only, no metadata) and outputs particle size and shape stats. Deliverable for Ben Labiner (CAMAL, NC State) tomorrow: working demo + accuracy number + one-pager.

## Environment (read first)
- Windows laptop, PowerShell. Project root: `C:\Users\lacaf\sem-powder`. Venv at `.venv` (activate: `.venv\Scripts\Activate.ps1`). Always use `python -m pip`.
- CPU only. No CUDA. Design everything CPU-first; auto-use GPU only if `torch.cuda.is_available()`.
- Limited disk. Do not keep the full synthetic dataset. Do not install conda, CUDA torch, or Detectron2 unless the Rettenberger step below works cleanly.
- The tool will be handed off to someone on a normal lab PC. Keep deps light and pinned in `requirements.txt`.
- Work autonomously. Do not stop to ask questions. When blocked, pick the most reasonable option, log it in `PROGRESS.md`, and move to the next block. Only stop if every remaining block is blocked.
- Test every step on a small subset (1 to 5 images) before batch runs.

## Folder layout
```
data/synthetic/   data/aa7075/   data/alloy2/
src/  out/  figures/
analyze.py  requirements.txt  README.md  PROGRESS.md  ONE_PAGER.md
```
`.gitignore`: `.venv/ data/ out/ __pycache__/`. Run `git init` and commit after each block.

## Data
1. **Synthetic validation set**: DeCost & Holm, Mendeley Data DOI 10.17632/tj4syyj9mr.1 (https://data.mendeley.com/datasets/tj4syyj9mr/1). 2048 Blender-rendered 512x512 PNGs, 8 PSDs (a to h) x 256 images, plus JSON 3D ground truth (sphere positions + radii) and the generating Blender/Python scripts. No scale bar.
   - Try to download programmatically. If that fails, log it and check whether the user has already put the files in `data/synthetic/`.
   - Keep the JSON and scripts. Keep about 50 images per PSD and delete the rest.
2. **Real images**: AA7075 + TiC powder for LPBF, Mendeley Data https://data.mendeley.com/datasets/42gzbgk5hb. Full-frame TIFFs of powder morphology (base 7075, TiC nano, TiC-modified) + measured PSD tables (CSV). Use base 7075 and TiC-modified powder images. Skip the nano-TiC images.
   - Same rule: try to download. If that fails, use whatever is in `data/aa7075/`.
3. **Second alloy**: `data/alloy2/` (Ti-6Al-4V or 316L, user-supplied). If empty, skip this and note it in `PROGRESS.md`.

## Blocks (in order)

### Block 1: Env + data
- Install: `torch` (CPU wheel: `--index-url https://download.pytorch.org/whl/cpu`), `cellpose<4`, `easyocr`, `scikit-image`, `opencv-python`, `pandas`, `matplotlib`, `tifffile`.
- `python check_env.py` must print `ok`.
- Done when: imports work, data on disk (or absence logged).

### Block 2: Scale calibration (`src/scale.py`)
- Detect the SEM data bar (the uniform strip, usually at the bottom). Crop it off before segmentation.
- Find the scale bar: the longest horizontal line segment in the data bar (threshold + morphology or HoughLinesP). Its length in px.
- OCR the label with easyocr. Regex for a number + unit (`nm`, `um`, `µm`, `μm`, `mm`). Convert to µm.
- Return `um_per_px`, the detection confidence, and the crop box.
- Fallbacks in this order: CLI `--um-per-px`, then a sidecar `scales.csv` (filename, um_per_px), then flag the image as uncalibrated and report sizes in px.
- Save a debug image showing the detected bar + OCR text.
- Done when: correct µm/px on the AA7075 images (sanity check: particle sizes land near the dataset's reported PSD order of magnitude).

### Block 3: Segmentation (`src/segment.py`)
Preprocess: grayscale, CLAHE, normalize to [0,1]. No alloy-specific thresholds.
- **Watershed baseline**: Otsu foreground, distance transform, `peak_local_max` markers (min_distance tied to estimated particle radius), `skimage.segmentation.watershed`.
- **Cellpose v3 `cyto3`**, zero-shot, `channels=[0,0]`. Estimate diameter automatically (`diameter=None`) and also try a value from the watershed median. Keep whichever validates better in Block 5.
- **Optional, hard 20-min timebox**: Rettenberger et al. 2024 uncertainty-aware Mask R-CNN (github.com/lrettenberger/Uncertainty-Aware-Particle-Segmentation-for-SEM). Only if it installs on CPU without Detectron2 build pain or a torch downgrade. Otherwise log "skipped" and move on. Do not let it break the venv. If it needs conflicting deps, use a separate venv `.venv-rett` or skip it.
- Post: remove edge-touching particles (`clear_border`) and particles under ~20 px area. Save colored overlays to `out/overlays/`.
- Done when: overlays exist for 2 synthetic + 2 real images, all methods.

### Block 4: Metrics (`src/metrics.py`)
- Per particle: area_px, area_um2, ECD (µm), perimeter, circularity = 4πA/P², aspect ratio (major/minor axis), solidity, centroid.
- Per image summary: count, D10/D50/D90 both number-weighted and volume-weighted (weights ∝ ECD³), mean circularity, mean aspect ratio, mean solidity.
- Satellite fraction: LOWEST PRIORITY. Only if time allows (small particles attached to large ones / low-solidity particles flag).
- One function: `analyze_image(path, method, um_per_px=None) -> (particles_df, summary_dict)`, plus a PSD histogram plot.

### Block 5: Synthetic validation (`src/validate_synthetic.py`). NEVER CUT THIS.
- Read the generating scripts/JSON to get the camera geometry, and work out the true px-per-unit at 512x512 (orthographic scale or camera FOV + distance). Document how it was derived.
- Ground truth issue: only the top layer is visible and partially occluded. Compute and report BOTH:
  1. The generating PSD D50 (all particles in the JSON).
  2. The "visible" ground truth: particles whose projected disk is mostly unoccluded from the camera (project spheres, depth sort, keep those with more than ~50% of the projected area visible) and not touching the image edge. This is the fair comparison.
- Run each method on ~50 images per PSD (a to h).
- Report per PSD and method: measured D50 vs. visible-true D50 (% error), vs. generating D50 (% error), the D10/D90 errors, and the count recall if feasible (matched particles / visible GT, using centroid matching).
- Also report: does the measured D50 rank the 8 PSDs in the same order as the truth? (Spearman ρ)
- Output: `out/synthetic_validation.csv` + `figures/fig2_synthetic_psd.png` (measured vs. true CDFs for 2 to 3 PSDs) + `figures/fig3_d50_parity.png` (measured vs. true D50, all PSDs, both methods).

### Block 6: Real-image check
- AA7075: run the best method. Compare the image-derived D10/D50/D90 against the dataset's measured PSD table. Note that laser diffraction PSD is volume-weighted, so compare against the volume-weighted image stats. Note any systematic bias (2D projection, edge exclusion, magnification).
- Alloy 2: run unchanged, with no parameter tweaks. Record where it breaks.
- Save 2 to 3 failure examples (overlays) to `figures/failures/`.
- Output `figures/fig4_aa7075_vs_reported.png`.

### Block 7: Wrap as a tool (`analyze.py`)
- `python analyze.py <image_folder> [--method cellpose|watershed] [--um-per-px X] [--out out/]`
- Outputs: `results.csv` (one row per image summary), `particles/<image>.csv`, `psd/<image>.png`, `overlays/<image>.png`, plus a combined PSD plot across all images.
- README: 3-line install, 1-line run, what each output column means, known limits.
- Include 2 sample images + expected `results.csv` in `samples/` (small files only).
- Finally, test from a fresh clone in a temp dir with a fresh venv, if disk allows.

### Block 8: Figures + one-pager
- `figures/fig1_overlays.png`: raw | watershed | Cellpose (| Rettenberger if used), one synthetic row + one real row.
- `ONE_PAGER.md`: what the tool does, inputs/outputs, accuracy (synthetic D50 error numbers, AA7075 comparison), known limits (2D projection of 3D spheres, magnification/resolution bias for small particles, overlapping/agglomerated particles, scale-bar OCR failures), and the ask: 20 to 30 real CAMAL SEM powder images from any run, no metadata needed.
- Writing style: succinct, technical, numbers first. No em dashes. No filler or narrative prose.

## Cut order if behind
1. Satellite fraction, 2. Rettenberger model, 3. Second alloy, 4. Fresh-clone test, 5. CLI polish.
Synthetic validation (Block 5) is never cut.

## Logging
Keep `PROGRESS.md` updated after each block: status, key numbers, decisions/assumptions, anything the user must do manually (e.g. "download X from browser into data/aa7075"). Put a 5-line summary at the top that the user can read on return.

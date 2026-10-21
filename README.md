# SEM Powder Characterizer (attempt 2)

Particle size and shape statistics from SEM images of metal powder, built on Cellpose-SAM
(a 2025 foundation segmentation model), fine-tuned on corrected SEM outlines, run in batches on a GPU.
Attempt 1 (classical watershed + zero-shot Cellpose 3, CPU) is archived in [`attempt1/`](attempt1/ATTEMPT1.md).

## Powder QC: sphere fraction and satellites before/after a jet change

Models: Cellpose-SAM fine-tuned on AMPIS particle outlines (F1 0.93 on the published validation image; Cohn 2021
Mask R-CNN: P 0.94, R 0.80) and Mask R-CNN for satellites (AMPIS + SALAS labels). Results in [PROGRESS.md](PROGRESS.md).

On the Hunt PC, in Windows PowerShell:

```
$env:SEM_STAGE = "qcmodels"
irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex
```

trains (first time, ~75 min; reused afterwards, `SEMCHAR_RETRAIN=1` retrains) and benchmarks the models. Then, in
`%USERPROFILE%\sem-powder`:

```
.\run.cmd qc C:\path\to\images
.\run.cmd compare C:\path\to\before C:\path\to\after
```

`qc`: per sample (one subfolder per sample) % spherical (minor/major axis >= 0.90), median sphericity, % satellited
particles, satellites per particle, D10/D50/D90, each with a 95% bootstrap CI (1000 resamples; whole images when a
sample has 3 or more, else particles). `compare`: B - A for every metric with 95% CI, p-value and CHANGED / no
detectable change. Use 5 or more images per condition at the same magnification so the CI covers field-to-field
variation. The scale bar is read from the image; otherwise pass `--um-per-px`.

## On the Hunt GPU PC (original size/shape workflow)

Open **Windows PowerShell** and paste:

```
irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex
```

That one command:
1. installs uv (Python manager) and portable Git into `%USERPROFILE%\sem-tools` (no admin, pinned versions, checksums verified);
2. clones the repository into `%USERPROFILE%\sem-powder`, or updates it on later visits;
3. installs Python 3.12, the CUDA build of PyTorch that matches the GPU driver, Cellpose 4 and the rest (first time ~10-20 min, ~6 GB);
4. downloads the datasets (~600 MB, from Mendeley Data and Zenodo);
5. runs the next stage and leaves results in `%USERPROFILE%\sem-powder\runs\`.

Use the same command on every visit. It picks the stage itself:

| Visit | What runs (unattended) | Time (rehearsed on a 6 GB laptop GPU; Hunt similar) |
|---|---|---|
| 1 | First time only: tools + Python + packages (~15-20 min). GPU check; model shoot-out on synthetic images with exact masks; all real samples measured and compared with laser diffraction; draft outlines for labeling | ~2-2.5 h |
| (labeling) | You correct the draft outlines: `run.cmd label` (see [LABELING.md](LABELING.md)) | ~4-6 h total, any sittings |
| 2 | Fine-tunes Cellpose-SAM on your corrections (200 epochs; falls back to lighter settings if GPU memory runs out); scores every model on the held-out test crops; final measurements with the fine-tuned model | ~1-2 h |

Keep the PowerShell window open and stay logged in while it runs (the program stops Windows from sleeping, but a
logout ends it). If a visit-1 run is interrupted, running the command again resumes it.

Afterwards, in `%USERPROFILE%\sem-powder`:

```
run.cmd analyze C:\path\to\images
```

measures any folder of images (one subfolder per sample). The other commands are `run.cmd label`, `run.cmd status` and `run.cmd visit1` / `visit2`, or double-click `run.cmd` to run the next stage.

**Bringing results back:** each run writes `runs\<stage>_<time>\` (open `index.html`) and a zip of it next to it. Copy the zip to OneDrive or a USB stick.

## What it measures

Per particle: ECD, area, perimeter, Feret max/min, circularity (4πA/P²), Feret aspect ratio, ellipse aspect ratio, roundness, convexity, solidity, number of touching particles, number of attached smaller ("satellite-like") particles.

Per sample (all images of the sample pooled): count; D10/D50/D90 by number, by volume, and by volume with edge correction (compare this one with laser diffraction); D[4,3], D[3,2], span; median and P10/P90 of every shape descriptor; % spherical (circularity >= 0.90 and Feret ratio >= 0.80); % touching; % with satellite-like neighbours; warnings (uncalibrated scale, fewer than 300 particles).

Scale per image: `--um-per-px` > `scales.csv` next to the images > SEM TIFF metadata (Zeiss, FEI) > scale-bar OCR > uncalibrated (flagged).

## Models compared

| Name | What it is |
|---|---|
| `watershed` | attempt 1's classical baseline (no learning) |
| `cpsam` | Cellpose-SAM v2, zero-shot |
| `cpsam_auto` | same, image rescaled so typical particles are 30-60 px |
| `sam_amg` | Segment Anything ViT-L "segment everything" + particle filter |
| `cpsam_ft` | Cellpose-SAM fine-tuned on corrected SEM outlines (visit 2) |

## Layout

`semchar/` package (`python -m semchar <stage>`), `scripts/setup_and_run.ps1` environment + runner, `hunt.ps1` bootstrap, `run.cmd` launcher, `configs/` small settings, `attempt1/` archive. Not in git: `data/`, `runs/`, `labels/`, `models/`, `cache/`, `.venv/`.

## Data credits

AA7075 + TiC: Mendeley Data 42gzbgk5hb v3 (CC BY 4.0). Ti-6Al-4V / Cu: Strauch, Altmann, Tönjes (IWT Bremen), Zenodo 10.5281/zenodo.17037765 (CC BY-SA 4.0). Synthetic renders: DeCost & Holm, Mendeley Data 10.17632/tj4syyj9mr.1 (CC BY 4.0). Cellpose-SAM: Pachitariu, Rariden, Stringer (BSD-3). Segment Anything: Meta AI (Apache-2.0).

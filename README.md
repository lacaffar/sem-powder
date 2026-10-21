# SEM Powder Characterizer (attempt 2)

Particle size and shape statistics from SEM images of metal powder, built on Cellpose-SAM
(a 2025 foundation segmentation model), fine-tuned on corrected SEM outlines, run in batches on a GPU.
Attempt 1 (classical watershed + zero-shot Cellpose 3, CPU) is archived in [`attempt1/`](attempt1/ATTEMPT1.md).

## On the Hunt GPU PC

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

| Visit | What runs (unattended) | Time on a Quadro RTX 4000 (estimate) |
|---|---|---|
| 1 | GPU check; model shoot-out on synthetic images with exact masks; all real samples measured and compared with laser diffraction; draft outlines made for labeling | 1-3 h |
| (labeling) | You correct the draft outlines: `run.cmd label` (see [LABELING.md](LABELING.md)) | ~4-6 h total, any sittings |
| 2 | Fine-tunes Cellpose-SAM on your corrections; scores every model on the held-out test crops; final measurements with the fine-tuned model | 1-2 h |

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

# Size/shape workflow (attempt 2, before the QC direction)

Full size and shape statistics with Cellpose-SAM fine-tuned on your own corrected outlines. Run each stage
explicitly: `$env:SEM_STAGE = "visit1"` (then `visit2`) before the one-liner, or `.\run.cmd <stage>`.

| Stage | What runs (unattended) | Time (6 GB laptop GPU; Hunt similar) |
|---|---|---|
| `visit1` | GPU check; model shoot-out on synthetic images with exact masks; real samples vs laser diffraction; draft outlines for labeling (downloads ~600 MB of data) | ~2-2.5 h |
| `label` | You correct the drafts in the Cellpose GUI (see [LABELING.md](../LABELING.md)) | ~4-6 h, any sittings |
| `visit2` | Fine-tunes Cellpose-SAM on the corrections (falls back to lighter settings on GPU memory errors); scores all models on held-out test crops; final measurements | ~1-2 h |

`.\run.cmd analyze C:\path\to\images [--model NAME] [--um-per-px X]` measures any folder (one subfolder per sample).
`.\run.cmd status` shows labeling progress. An interrupted `visit1` resumes when run again.

## What it measures

Per particle: ECD, area, perimeter, Feret max/min, circularity (4πA/P²), Feret aspect ratio, ellipse aspect ratio,
roundness, convexity, solidity, touching neighbours, attached smaller ("satellite-like") particles.

Per sample: count; D10/D50/D90 by number, by volume, and by volume with edge correction (compare this one with laser
diffraction); D[4,3], D[3,2], span; median and P10/P90 of every shape descriptor; % spherical (circularity >= 0.90
and Feret ratio >= 0.80); % touching; % with satellite-like neighbours; warnings (uncalibrated, < 300 particles).

## Models compared

| Name | What it is |
|---|---|
| `watershed` | attempt 1's classical baseline (no learning) |
| `cpsam` | Cellpose-SAM v2, zero-shot |
| `cpsam_auto` | same, image rescaled so typical particles are 30-60 px |
| `sam_amg` | Segment Anything ViT-L "segment everything" + particle filter |
| `cpsam_ft` | Cellpose-SAM fine-tuned on corrected SEM outlines (visit 2) |

# SEM Powder Characterizer, attempt 2

## Goal
Best-performing characterizer for SEM images of metal AM powder (any alloy, image only): particle outlines from a
foundation segmentation model, fine-tuned on corrected SEM outlines, then a general set of size and shape
characteristics per sample. Runs unattended in batches on a GPU. Attempt 1 (CPU, watershed + Cellpose 3) is archived in
`attempt1/` (git tag `attempt1`); reuse its validated pieces, do not edit it.

## Machines
- **Hunt Library GPU PC** (where batch runs happen): Windows 11, Dell Precision 7820, Xeon Silver 4214R, NVIDIA Quadro
  RTX 4000 (8 GB, Turing sm_75: no native bfloat16, so models run float32), 64 GB RAM, 2 TB disk. Files persist between
  sessions. No admin rights assumed. The user runs ONE command there:
  `irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex`
- **Dev laptop** (this repo): Windows 11, NVIDIA RTX A1000 6 GB (driver 596), Python 3.12 venv in `.venv` with CUDA 12.6
  torch, made by uv (`%USERPROFILE%\sem-tools\uv\uv.exe`). Disk is nearly full: check `df` before downloads, never keep
  more than one copy of model weights or datasets. Set `SEMCHAR_FP32=1` to mimic the Hunt GPU's float32 path.

## Workflow
- Visit 1 (unattended): GPU check, data download, zero-shot shoot-out (watershed, cpsam, cpsam_auto, sam_amg) on
  synthetic images with exact masks + real samples vs laser diffraction, labeling drafts in `labels/`.
- User corrects drafts in the Cellpose GUI (`run.cmd label`, see LABELING.md).
- Visit 2 (unattended): fine-tune Cellpose-SAM (attempts in child processes, falling back on GPU OOM), pick epoch on
  val crops, score all models on the held-out test crops, final characterization.
- Results: `runs/<stage>_<time>/` + zip. The user brings zips back by OneDrive/USB; Hunt cannot push to GitHub.

## Rules
- Everything the Hunt run needs must be in git and work from a fresh clone with only the one command. Data and model
  weights are downloaded by the code (pinned URLs, checksums where available); never commit data, runs, labels, models.
- Test on small subsets first. Test GPU code on the laptop GPU (6 GB is a stricter memory test than Hunt's 8 GB).
- A long unattended run must never die on one bad image or one failing model: catch, log, continue, report.
- Pin versions (requirements.txt, torch build tags like `2.14.1+cu126`, tool releases at least two weeks old).
- Honest metrics: held-out splits fixed before looking (test = Fig 2c + Ti64+15Cu crops), report what failed.
- Writing style for docs and reports: succinct, technical, numbers first, no em dashes, no filler.
- Commit and push after each meaningful step (repo is public: github.com/lacaffar/sem-powder).

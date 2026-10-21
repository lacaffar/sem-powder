# SEM Powder QC

Measures sphericity and satellites in SEM images of metal AM powder, and tells you whether they changed between
two batches (for example before and after an atomizer jet change). Runs on any Windows PC; a GPU makes it faster.

## 1. Set up (once per computer, ~15-25 min)

Open **Windows PowerShell** (no admin needed) and paste:

```
irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex
```

This installs everything into your user folder (`%USERPROFILE%\sem-powder`) and downloads the trained models
(1.4 GB). Run the same line again later to update.

## 2. Take the images

- 5-10 random fields per batch (300+ particles per batch). Do not pick nice-looking areas.
- Same magnification, detector (SE or BSE), kV and sample prep for every batch you compare.
- A thin single layer of powder on carbon tape; heavy piles and overlaps reduce accuracy.
- Save the original `.tif` from the SEM. Sizes come from the TIFF metadata or the scale bar; screenshots and
  cropped JPEGs lose that (sizes then stay in pixels, sphericity and satellites still work).

Put each batch in its own folder:

```
C:\SEM\jet_old\   all "before" images
C:\SEM\jet_new\   all "after" images
```

## 3. Run

In PowerShell:

```
cd $env:USERPROFILE\sem-powder
.\run.cmd compare C:\SEM\jet_old C:\SEM\jet_new
```

To measure batches without comparing, use one folder with one subfolder per batch:

```
.\run.cmd qc C:\SEM\all_batches
```

Time per image: ~5-10 s with an NVIDIA GPU, ~1.5-3 min without one.

## 4. Read the results

Results go to `runs\compare_<date-time>\` (or `runs\qc_...`), with a zip of the folder next to it.

- `index.html`: the main table. For each metric: before, after, the difference with its 95% range, a p-value, and
  **CHANGED** or **no detectable change**.
- `overlays\`: every image with particle outlines and satellites circled in red. Flip through these to catch a bad
  image or a bad outline.
- CSV files with every particle, for your own plots.

| Metric | Meaning |
|---|---|
| `pct_spherical` | % of particles with sphericity >= 0.90 (sphericity = short axis / long axis of the outline) |
| `sphericity_median` | median sphericity |
| `pct_satellited` | % of particles carrying at least one satellite |
| `satellites_per_particle` | mean number of satellites per particle |
| `D10_num`, `D50_num`, `D90_num` | particle size (equivalent circle diameter, µm) at 10/50/90% by count |
| `D50_vol` | median size by volume |

Particles cut by the image edge are left out. The 95% ranges come from resampling the images 1000 times, so they
include field-to-field variation (with fewer than 3 images per batch, particles are resampled instead).

## How accurate is it

Scored against published hand-labeled SEM powder images:

| | Precision | Recall | Published method (Cohn 2021) |
|---|---|---|---|
| Particles (Cellpose-SAM, fine-tuned) | 0.94 | 0.93 | 0.94 / 0.80 |
| Satellites (Mask R-CNN, tuned) | 0.67 | 0.64 | 0.69 / 0.55 |

Particle outlines are reliable. Satellite counts are right on average, but their bias depends on the powder (±15-20%
between datasets). **Trust before/after changes more than absolute % satellited.** Hand-count satellites on 2-3 of
your own images once to check. The models were trained on public data (Ni superalloy, Al, IN718, 17-4 SS, Ti
alloys), not yet on CAMAL images.

## Problems

| Problem | Fix |
|---|---|
| `run.cmd` not recognized | Type `.\run.cmd` and run it from `%USERPROFILE%\sem-powder` |
| Sizes in px instead of µm | Scale not found: add `--um-per-px 0.25` (your value) to the command |
| Red error text | Copy the last ~40 lines; full logs are in `runs\logs\` |
| Scripts blocked by policy | Use the setup line from step 1; it does not need script permissions |

## For developers

- Details, decisions and all results: [PROGRESS.md](PROGRESS.md).
- Retrain the models (NVIDIA GPU, ~75 min): `$env:SEM_STAGE = "qcmodels"`, then the setup line. Training data:
  AMPIS and SALAS hand labels, downloaded automatically.
- Publish a new model set: zip `qc_models.json`, `cp_particles`, `cp_particles.json`, `mrcnn_satellites.pt` from
  `models\`, attach it to a new GitHub release, update `MODELS_URL` and `MODELS_SHA256` in `semchar/release.py`.
- The earlier full size/shape workflow with your own labels: [docs/SIZE_SHAPE_WORKFLOW.md](docs/SIZE_SHAPE_WORKFLOW.md).
  Attempt 1 is archived in [attempt1/](attempt1/ATTEMPT1.md).
- Layout: `semchar/` Python package (`python -m semchar <stage>`), `hunt.ps1` setup, `run.cmd` launcher,
  `scripts/setup_and_run.ps1` environment. Not in git: `data/`, `runs/`, `labels/`, `models/`, `cache/`, `.venv/`.

## Credits

Training data: AMPIS (Cohn et al. 2021, MIT) and SALAS (Price et al. 2021, MIT). Test images: AA7075 + TiC, Mendeley
Data 42gzbgk5hb v3 (CC BY 4.0); Ti-6Al-4V / Cu, Strauch, Altmann, Tönjes, Zenodo 10.5281/zenodo.17037765
(CC BY-SA 4.0); DeCost & Holm renders, Mendeley Data 10.17632/tj4syyj9mr.1 (CC BY 4.0). Models: Cellpose-SAM
(Pachitariu, Rariden, Stringer; BSD-3), Mask R-CNN via torchvision (BSD-3), Segment Anything (Meta AI, Apache-2.0).

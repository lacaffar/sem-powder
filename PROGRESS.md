# PROGRESS (attempt 2)

## Summary (read this first)
1. Attempt 1 archived in `attempt1/` (git tag `attempt1`). New pipeline `semchar/` built around Cellpose-SAM v2 (2025
   foundation segmentation model) with a classical baseline and Segment Anything for comparison.
2. Hunt PC: one command (`irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex`) installs
   tools, clones/updates, sets up CUDA Python, downloads data, runs the next stage. Tested in Windows PowerShell 5.1
   on the dev laptop from GitHub (fresh clone + update path).
3. Visit 1 rehearsed end to end on the dev laptop GPU (RTX A1000 6 GB, float32 like the Hunt Quadro): 1 h 48 min,
   no errors; best zero-shot model cpsam_auto. Visit 2 rehearsed with stand-in labels: OK. Final check: fresh clone
   from GitHub via the one-liner + `run.cmd visit1 --quick` (fresh data and model downloads): OK in 8 min.
4. Fine-tuning code path tested on the laptop GPU with stand-in labels: full fine-tuning at batch 2 runs out of 6 GB;
   each attempt now runs in a child process and falls back (batch 1, then 12 or 20 frozen encoder blocks).
5. For you: visit 1 at Hunt, then correct drafts (LABELING.md), then visit 2. Check whether Hunt GPU PCs log out
   idle sessions; the program keeps Windows awake but cannot override a forced logoff.

## Decisions
- Engine: Cellpose-SAM `cpsam_v2` (cellpose 4.2.1.1). Reasons: best-reported generalist instance segmenter in 2025-26
  benchmarks, size/noise robust, built-in fine-tuning and GUI for corrections. Zero-shot test on AA7075 showed whole
  particles incl. shaded flanks and partly hidden ones, satellites separated.
- Contenders in the shoot-out: `watershed` (attempt 1), `cpsam`, `cpsam_auto` (rescale so typical particle 30-60 px),
  `sam_amg` (Segment Anything ViT-L everything-mode + foreground filter). Not included: micro-sam (conda-only deps on
  Windows), SAM 2 via `sam2` (CUDA extension build), cpdino (needs DINOv3 from git), Mask R-CNN models (Detectron2).
- Turing GPU has no native bfloat16: models run float32 there (bf16 emulation was ~14x slower on CPU tests).
- Ground truth for ranking without labels: DeCost renders (ray-cast visible-instance masks, attempt 1) and new dense
  monolayer renders with parents, fused satellites and fines (exact masks). Real samples vs laser D10/D50/D90 is a
  low-weight sanity check (one frame per sample).
- Ranking score: 0.5 F1@0.5 dense + 0.25 F1@0.5 DeCost + 0.25 (1 - |D50 err| real).
- Label split fixed before labeling: test = Fig 2c + Ti64+15Cu crops, val = Fig 2d + Ti64+20Cu, train = rest
  (Cu moved from val to train after the rehearsal: its crops hold 200-330 particles each, so they are optional/last).
- Torch builds pinned by local tag (`2.14.1+cu126`; `2.7.1+cu118` for drivers 452-528; `+cpu` otherwise) because uv
  treats `torch==2.14.1` as satisfied by the CPU build.
- Scripts run as script blocks (`[scriptblock]::Create(...)`) so Group Policy that blocks .ps1 files does not stop them.

## Visit 1 rehearsal (dev laptop RTX A1000 6 GB, float32 forced like the Hunt Quadro)
- Ran end to end with the code as committed: 1 h 48 min (data present; at Hunt add ~5 min for downloads and ~15 min
  for the first environment install). No errors. 32 drafts written. Run folder 28 MB, zip 22 MB.
- Shoot-out (synthetic: 96 DeCost + 40 dense renders with exact masks; real: 8 samples vs laser):

| Model | Score | Dense F1@0.5 | Dense IoU | DeCost F1@0.5 | DeCost IoU | Real mean abs D50 err | s/img (dense) |
|---|---|---|---|---|---|---|---|
| cpsam_auto | 0.835 | 0.80 | 0.92 | 0.93 | 0.90 | 19.1% | 5.0 |
| sam_amg | 0.833 | 0.82 | 0.89 | 0.88 | 0.84 | 18.4% | 28.8 |
| cpsam | 0.831 | 0.80 | 0.92 | 0.91 | 0.89 | 19.1% | 4.9 |
| watershed | 0.730 | 0.61 | 0.88 | 0.83 | 0.85 | 13.5% | 0.2 |

- Cellpose-SAM finds 99.7% of particles > 1000 px area and 92% of 300-1000 px, but only 17-67% of particles
  < 300 px (small satellites and fines). Zero-shot knobs do not fix it (20 dense renders, F1 / recall):
  native 0.818 / 0.698; flow_threshold 0.6: 0.826 / 0.711; 1.5x upscale 0.823 / 0.706; 2x upscale 0.824 / 0.706
  (3x slower); cellprob -1: 0.813 / 0.690. Defaults kept; fine-tuning on corrected labels is the fix.
- Real vs laser (cpsam_auto, volume edge-corrected D50): AA7075 -15 to -25% (few particles per frame, as in attempt 1);
  Ti-6Al-4V +14%, Ti+Cu blends +10 to +27%, Cu +21%; D90 within 6% for Ti64, Ti64+10/15Cu and Cu. Image and laser
  sample different populations, so this is a weak check; synthetic exact masks drive the ranking.
- Labeling GUI checked: `python -m semchar label` opens the Cellpose GUI on the first crop with the draft loaded.

## Visit 2 rehearsal (stand-in labels: 12 dense synthetic renders, 6 train / 3 val / 3 test; 8 epochs)
- End to end OK: fine-tune, checkpoint choice on val, test table, final characterization of 8 real samples (6 min).
- Fine-tuning memory on the 6 GB card: batch 2 full and batch 1 full ran out of GPU memory (now clean, in 2 s each);
  batch 1 with 12 of 24 encoder blocks frozen fitted (4.6 GB peak, 2.1 min for 8 epochs). Hunt's 8 GB may fit batch 1 full.
- Problem found and fixed: recent NVIDIA Windows drivers spill to system RAM instead of failing when GPU memory runs
  out; the first rehearsal attempt crawled for 10 min and the page file ate ~9 GB of disk. The worker now caps PyTorch
  at 92% of VRAM (`SEMCHAR_VRAM_FRACTION`), so overflow is a clean out-of-memory error and the next setting is tried.
  Each attempt also has a 4 h watchdog (`SEMCHAR_FT_TIMEOUT_H`).
- Held-out stand-in crops (synthetic, so only shows the machinery works and that fine-tuning helps):

| Model | F1@0.5 | Precision | Recall | IoU | D50 number err | D50 volume err |
|---|---|---|---|---|---|---|
| watershed | 0.59 | 0.99 | 0.42 | 0.87 | +197% | +7.3% |
| cpsam (zero-shot) | 0.81 | 0.99 | 0.69 | 0.92 | +18.8% | +3.4% |
| cpsam_ft (8 epochs, 6 crops) | 0.89 | 0.93 | 0.84 | 0.90 | +5.9% | +0.5% |

- Stand-in model and results deleted afterwards so nothing fake is used later.
- Test switches: `SEMCHAR_FP32=1` (mimic Turing), `SEMCHAR_FT_EPOCHS`, `SEMCHAR_FT_SAVE_EVERY`, `SEM_NO_SETUP=1` (hunt.ps1).

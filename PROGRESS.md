# PROGRESS (attempt 2)

## Summary (read this first)
1. Attempt 1 archived in `attempt1/` (git tag `attempt1`). New pipeline `semchar/` built around Cellpose-SAM v2 (2025
   foundation segmentation model) with a classical baseline and Segment Anything for comparison.
2. Hunt PC: one command (`irm https://raw.githubusercontent.com/lacaffar/sem-powder/main/hunt.ps1 | iex`) installs
   tools, clones/updates, sets up CUDA Python, downloads data, runs the next stage. Tested in Windows PowerShell 5.1
   on the dev laptop from GitHub (fresh clone + update path).
3. Visit 1 rehearsed end to end on the dev laptop GPU (RTX A1000 6 GB, float32 like the Hunt Quadro): see below.
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
- Label split fixed before labeling: test = Fig 2c + Ti64+15Cu crops, val = Fig 2d + Cu, train = rest.
- Torch builds pinned by local tag (`2.14.1+cu126`; `2.7.1+cu118` for drivers 452-528; `+cpu` otherwise) because uv
  treats `torch==2.14.1` as satisfied by the CPU build.
- Scripts run as script blocks (`[scriptblock]::Create(...)`) so Group Policy that blocks .ps1 files does not stop them.

## Visit 1 rehearsal (dev laptop GPU)
(filled in below when the rehearsal finishes)

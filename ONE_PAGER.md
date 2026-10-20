# SEM Powder Characterization Tool

**Image-only particle size and shape statistics for metal AM powder. Any alloy, CPU only, ~1 s per image.**

## Accuracy
**Synthetic SEM renders with exact 3D ground truth** (DeCost & Holm 2016; 8 PSDs; 383 images watershed, 127 Cellpose; ~400 visible particles per image):

| Metric (mean over 8 PSDs) | Watershed (default) | Cellpose 3 (zero-shot) |
|---|---|---|
| Number-weighted D50 error | +12% (range +7 to +19%) | +19% |
| Volume-weighted D50 error | +12% (range +10 to +14%) | 6% abs. (range -9 to +12%) |
| D90 error | +12% | +11% |
| Particles found (recall / precision) | 0.78 / 0.99 | 0.83 / 0.97 |
| Size of correctly found particles | within 2% | +10% |
| Ranks 8 PSDs by D90 / volume D50 (Spearman ρ) | 1.00 / 1.00 | 1.00 / 0.98 |
| CPU time per image | 0.3 s | 20 s |

D50 reads high because overlapping particles merge and particles under 5 px are not detected. The two groups of PSDs (D50 0.20 vs 0.24 to 0.26) separate with no overlap for every method.

**Real AA7075 powder vs laser diffraction** (Mendeley 42gzbgk5hb, one 300x Zeiss frame per powder, volume-weighted, edge-corrected):

| Powder | Particles | D50 image / laser (µm) | D50 error | D90 error |
|---|---|---|---|---|
| Base 7075 | 176 | 38.4 / 44.4 | -14% | -17% |
| 7075 + TiC (surface mod.) | 271 | 36.5 / 39.7 | -8% | -30% |
| 7075 + TiC (volume mod.) | 29 | 53.3 / 71.4 | -25% | -42% |

The image under-reads laser because one 381 x 286 µm frame holds few large particles, and those touching the edge are dropped. Laser diffraction also reads high on agglomerates (one sample has 6% volume at 1 to 3 mm). Error grows as particle count falls. Fix: several fields per powder, at least 300 particles.

**Scale bar reading:** 5 test data bars (Zeiss, FEI and JEOL styles; µm and mm labels): max error 0.25%. Zeiss and FEI TIFF metadata are read directly when present.

## What it does
`python analyze.py <folder>` on any folder of SEM images (TIF, PNG, JPG).
1. Scale: µm/px from SEM TIFF metadata or OCR of the scale bar, or a CSV or command-line value. If none is found, results are reported in px and flagged.
2. Crops the data bar, then segments particles (watershed or Cellpose). No alloy-specific settings.
3. Per particle: ECD, area, perimeter, circularity, aspect ratio, solidity.
4. Per image: count, D10/D50/D90 (number- and volume-weighted, plus volume edge-corrected), mean shape metrics, fraction of irregular particles (satellites or agglomerates).
5. Outputs: `results.csv`, per-particle CSVs, PSD plots, segmentation overlays, scale-bar debug images.

## Known limits
- **2D projection of 3D powder:** partly buried particles read small; overlapping particles merge.
- **Resolution:** particles under 5 px ECD are not detected. Small particles need higher magnification.
- **Field of view:** D90 needs many large particles. Few particles per frame gives a biased D90.
- **Agglomerates and satellite clusters** are measured as one particle (watershed). Cellpose splits more.
- **Background:** assumes particles brighter than the substrate (SE/BSE on carbon tape).
- **Scale-bar OCR:** validated on rendered bars only. Check the scale debug image or pass `--um-per-px`.
- **Not yet tested** on Ti-6Al-4V, 316L, or any CAMAL instrument.

## Ask
**20 to 30 SEM images of powder from any CAMAL run.** Any alloy, any magnification. No metadata needed; a visible scale bar or the original TIFF is enough. A laser-diffraction PSD for the same lot, if one exists, lets us report accuracy on CAMAL data.

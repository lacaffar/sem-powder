# Data mirror (fallback copies)

Small copies of the real-image test data, used only when the original repositories cannot be reached
(e.g. Zenodo rate-limiting a network). `semchar/data.py` always tries the original source first.
Files are unmodified. Licenses and attribution:

- `aa7075/`: "A novel approach for modifying aluminum powder EN AW-7075 with nanoparticles for additive
  manufacturing", Mendeley Data, doi:10.17632/42gzbgk5hb.3. License: CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Files: Fig 2a/2c/2d original SEM images, PSD.CSV, ReadMe_PSD.txt.
- `alloy2/`: A. Strauch, M. L. Altmann, A. Tönjes (Leibniz-IWT Bremen), "Experimental data on Ti-6Al-4V and Cu powder
  blends for additive manufacturing", Zenodo, doi:10.5281/zenodo.17037765. License: CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). Files: Particle_shape/*.tif and
  Particle_size_distribution/Table2_ParticleSizeDistribution.xlsx. Redistributed here under the same license.

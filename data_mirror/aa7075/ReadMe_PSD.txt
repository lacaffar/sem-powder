1. Dataset description
This dataset contains particle size distribution (PSD) results obtained from a laser diffraction analysis following the instrument’s standard operating procedure (SOP). Each row represents an individual measurement run or material sample, and each column provides its associated metadata, instrument settings, and calculated particle size metrics. The dataset is intended for reproducibility of particle characterization analyses and comparison of size distribution parameters such as d(0.1), d(0.5), and d(0.9)

2. Column definitions
Sample Name | Identifier of the analyzed material sample. 
Sample source type / bulk lot ref | Source and batch information for traceability.
SOP Name | Measurement procedure used for this analysis.
Operator name | Individual who conducted the measurement.
Measurement date and time / Analysis date and time | Time stamps for measurement and subsequent data evaluation.
Particle name | Designation of the particle material (e.g., silica, alumina, etc.).
Particle refractive index / absorption index | Optical constants used for particle-light interaction modeling.
Dispersant name / refractive index | Liquid or gas medium used for particle dispersion and its optical refractive index.
Accessory name / Analysis model | Measurement accessory (e.g., wet or dry cell) and mathematical model applied (Mie, Fraunhofer, etc.).
Start/Last result channel size | Size range limits (µm) defining the analyzed particle distribution interval.
Result emulation / Obscuration / Residual / Concentration | Instrument parameters for optical signal quality, fit residuals, and particle concentration.
Span | Indicator of width of the particle size distribution: (d(0.9) − d(0.1)) / d(0.5).
Result transform type | Mathematical representation of the PSD (volume, number, or surface-based).
D [4,3] | Volume-weighted mean particle diameter (µm).
D [3,2] | Surface-weighted mean particle size (µm).
Uniformity | Describes distribution homogeneity; lower values indicate narrower size range.
Specific surface area | Derived specific surface area based on particle size distribution (m²/g).
d(0.1), d(0.5), d(0.9) | Percentile diameters – 10%, 50% (median), and 90% of the cumulative particle size distribution.
Result Between User Sizes (Columns 1–99) | Full cumulative particle size distribution values across up to 99 defined size channels (µm). Each column corresponds to a distinct particle size class .
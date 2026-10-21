# Attempt 1 (archived)

First version of the SEM powder tool: watershed + zero-shot Cellpose 3 (cyto3) on CPU, scale-bar OCR, size/shape metrics, synthetic validation, AA7075 and Ti-6Al-4V checks, satellite detector. Results and decisions: `PROGRESS.md`. Summary for others: `ONE_PAGER.md`.

Superseded by the GPU foundation-model pipeline at the repository root. Kept unchanged for reference. Git tag `attempt1` marks the last commit where this was the root.

To run it, work inside this folder (its scripts expect `data/` and `.venv` next to them): copy or link the root `data/` folder to `attempt1/data/`, create a Python 3.13 venv, `python -m pip install --only-binary=:all: -r requirements.txt`, then `python analyze.py samples --out out/demo`.

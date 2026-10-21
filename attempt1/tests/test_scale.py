"""Check scale calibration on rendered data bars + the real AA7075 TIFFs."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scale import calibrate  # noqa: E402

DBG = ROOT / "out/scale_debug"


def main():
    print("== rendered data bars (OCR path, metadata off)")
    errs = []
    with open(ROOT / "data/scaletest/truth.csv") as fh:
        for row in csv.DictReader(fh):
            p = ROOT / "data/scaletest" / row["file"]
            r = calibrate(p, use_meta=False, debug_path=DBG / row["file"])
            t = float(row["um_per_px"])
            e = (r.um_per_px / t - 1) if r.um_per_px else float("nan")
            errs.append(abs(e))
            print(f"{row['file']:28s} true {t:.4f} got {r.um_per_px or float('nan'):.4f} "
                  f"err {e:+.2%} src={r.source} conf={r.confidence:.2f} label={r.label} "
                  f"bar={r.bar_px} truebar={row['bar_px']} crop={r.crop_box}")
    print("== real AA7075 TIFFs")
    for p in sorted((ROOT / "data/aa7075").glob("*.tif")):
        r = calibrate(p, debug_path=DBG / (p.stem + ".png"))
        print(f"{p.name:32s} {r.source:10s} {r.um_per_px} crop={r.crop_box} notes={'; '.join(r.notes)[:120]}")
    ok = all(e < 0.02 for e in errs)
    print("PASS" if ok else "FAIL", f"max OCR-path error {max(errs):.2%}")


if __name__ == "__main__":
    main()

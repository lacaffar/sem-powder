"""SEM powder characterizer.

  python -m semchar envcheck            GPU / library check
  python -m semchar data                download all datasets (idempotent)
  python -m semchar visit1 [--quick]    shoot-out + characterize real samples + labeling drafts
  python -m semchar drafts              (re)make labeling drafts for crops that have none yet
  python -m semchar label               open the Cellpose GUI on the next crop to correct
  python -m semchar status              labeling progress
  python -m semchar visit2              fine-tune on corrected crops, test, final results
  python -m semchar analyze <folder> [--model NAME] [--um-per-px X] [--out DIR]
  python -m semchar qcmodels [--quick]  train + benchmark particle and satellite models on public labeled data,
                                        pick the best, run QC on the real samples
  python -m semchar qc <folder>         sphericity + satellites per sample (subfolder = sample), with 95% CIs
  python -m semchar compare <before> <after>   is each QC metric different? (e.g. jet change)
  python -m semchar getmodels           download the trained QC models (GitHub Release) if missing
  python -m semchar auto                get QC models (download, else train); else visit2 if labels reviewed

Results go to runs/<stage>_<timestamp>/ (SUMMARY.md, index.html, CSVs, overlays) and a zip of it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def new_run(stage, resume=False):
    """New run directory, or (resume=True) the latest unfinished one from the last 3 days."""
    if resume:
        cands = sorted(d for d in RUNS.glob(f"{stage}_*") if d.is_dir() and not (d / "DONE").exists()
                       and time.time() - d.stat().st_mtime < 3 * 86400)
        if cands:
            d = cands[-1]
            log = Log(d / "log.txt")
            log(f"=== resuming unfinished run {d.name} ===")
            return d, log
    d = RUNS / f"{stage}_{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d, Log(d / "log.txt")


def keep_awake(on=True):
    """Ask Windows not to sleep while a long stage runs (does not override forced logoff policies)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def envcheck(log=print):
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    try:
        from importlib.metadata import version
        info["cellpose"] = version("cellpose")
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}",
                    torch_cuda=torch.version.cuda)
        from .models import native_bf16
        info["bf16_native"] = native_bf16()
        x = torch.randn(512, 512, device="cuda")
        info["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    try:
        smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
        out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = out.stdout.strip()
    except Exception:
        info["nvidia_smi"] = "not found"
    for k, v in info.items():
        log(f"  {k}: {v}")
    if not info["cuda_available"]:
        log("  !!! NO GPU IN USE: running on CPU (much slower; visit 1 switches to a reduced quick run)")
    return info


def finish(run_dir, log, title, sections):
    from .report import simple_page
    simple_page(run_dir / "index.html", title, sections)
    z = shutil.make_archive(str(run_dir), "zip", run_dir)
    (RUNS / "LATEST.txt").write_text(run_dir.name)
    log(f"results: {run_dir}")
    log(f"zip for copying back: {z}")


def latest(prefix):
    c = sorted(RUNS.glob(f"{prefix}_*/DONE"))
    return c[-1].parent if c else None


def visit1(quick=False):
    import pandas as pd
    from . import data as data_mod
    from .labeling import make_drafts, status
    from .models import ALL_ZERO_SHOT, make
    from .report import table_html
    from .shootout import run as shootout
    run_dir, log = new_run("visit1", resume=True)
    log("=== VISIT 1: setup check, model shoot-out, characterization, labeling drafts ===")
    env = envcheck(log)
    json.dump(env, open(run_dir / "env.json", "w"), indent=1)
    quick = quick or not env["cuda_available"]
    log("--- data")
    data_mod.main()
    models = ["watershed", "cpsam"] if quick else ALL_ZERO_SHOT
    log(f"--- shoot-out ({'QUICK' if quick else 'full'})")
    shoot, best = shootout(models, run_dir / "shootout", n_decost=2 if quick else 12, n_dense=6 if quick else 40,
                           budget_s=600 if quick else 2400, log=log)
    log("--- labeling drafts")
    draft_model = best if best.startswith("cpsam") else "cpsam"
    try:
        seg = make(draft_model)
        make_drafts(seg, log=log)
        seg.close()
    except Exception as e:
        log(f"  drafts FAILED: {e}\n{traceback.format_exc()}")
    st = status()
    best_dir = run_dir / "shootout" / "real" / best
    if best_dir.exists():
        shutil.copytree(best_dir, run_dir / "results_best_model", dirs_exist_ok=True)
    cols = [c for c in ["model", "score", "dense_F1@0.5", "dense_mean_iou_matched", "dense_D50_vol_err_mean",
                        "decost_F1@0.5", "decost_mean_iou_matched", "decost_D50_num_err_mean",
                        "real_D50_abs_err", "real_D90_abs_err", "dense_sec_per_image", "error"] if c in shoot]
    real_csv = best_dir / "samples.csv"
    real = pd.read_csv(real_csv) if real_csv.exists() else pd.DataFrame()
    rcols = [c for c in ["sample", "count", "D10_vol_ec", "D50_vol_ec", "D90_vol_ec", "ref_D10", "ref_D50", "ref_D90",
                         "err_D50_vol_ec", "err_D90_vol_ec", "circularity_median", "feret_ar_median",
                         "pct_spherical", "pct_with_satellite_like", "flags"] if c in real]
    md = [f"# Visit 1 results ({run_dir.name})", "", f"GPU: {env.get('gpu', 'NONE (CPU quick run)')}", "",
          f"Best zero-shot model: **{best}**", "", "## Shoot-out", shoot[cols].round(4).to_string(index=False), "",
          "## Real samples (best model) vs laser", real[rcols].round(3).to_string(index=False) if len(real) else "(none)", "",
          f"## Labeling: {int(st.reviewed.sum()) if len(st) else 0}/{len(st)} crops reviewed",
          "Next: correct the drafts (`run.cmd label`), then run visit 2 (`run.cmd`)."]
    (run_dir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    (run_dir / "DONE").write_text(time.strftime("%c"))
    finish(run_dir, log, "Visit 1 results",
           [("Environment", table_html(pd.DataFrame(list(env.items()), columns=["item", "value"]))),
            (f"Shoot-out (best: {best})", table_html(shoot[cols])),
            ("Real samples, best model, vs laser", table_html(real[rcols]) if len(real) else "none"),
            ("Per-sample reports", "<p>results_best_model/index.html and shootout/real/&lt;model&gt;/</p>"),
            ("Labeling", table_html(st) if len(st) else "no drafts")])
    log("=== VISIT 1 DONE ===")


def visit2():
    import pandas as pd
    from .characterize import gather, run as characterize
    from .finetune import test_table, train
    from .labeling import status
    from .models import make
    from .refs import DATA, REAL_SAMPLES, reference
    from .report import table_html
    run_dir, log = new_run("visit2")
    log("=== VISIT 2: fine-tune on corrected outlines, test, final characterization ===")
    env = envcheck(log)
    st = status()
    log(f"labels: {int(st.reviewed.sum())}/{len(st)} crops reviewed "
        + str(st[st.reviewed].groupby("split").size().to_dict()))
    sel = train(log=log)
    sel.to_csv(run_dir / "checkpoint_selection.csv", index=False)
    log("--- held-out test crops")
    test = test_table(["watershed", "cpsam", "cpsam_auto", "cpsam_ft"], log=log)
    test.to_csv(run_dir / "test_scores.csv", index=False)
    log("--- final characterization with the fine-tuned model")
    seg = make("cpsam_ft")
    refs = {k: reference(k) for k in REAL_SAMPLES if (DATA / REAL_SAMPLES[k][0]).exists()}
    real, _ = characterize([(k, DATA / REAL_SAMPLES[k][0]) for k in refs], seg, run_dir / "results_real",
                           refs=refs, log=log)
    camal = DATA / "camal"
    if camal.exists():
        characterize(gather(camal), seg, run_dir / "results_camal", log=log)
    seg.close()
    tcols = [c for c in ["model", "n_gt", "F1@0.5", "P@0.5", "R@0.5", "F1@0.75", "mean_iou_matched",
                         "D50_num_err_mean", "D50_vol_err_mean", "circularity_median_err_mean",
                         "sec_per_image"] if c in test]
    md = [f"# Visit 2 results ({run_dir.name})", "", "## Held-out test crops (user-corrected)",
          test[tcols].round(4).to_string(index=False) if len(test) else "(no reviewed test crops)", "",
          "## Real samples, fine-tuned model, vs laser",
          real[[c for c in ["sample", "count", "D50_vol_ec", "ref_D50", "err_D50_vol_ec", "err_D90_vol_ec",
                            "pct_spherical", "flags"] if c in real]].round(3).to_string(index=False)]
    (run_dir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    (run_dir / "DONE").write_text(time.strftime("%c"))
    finish(run_dir, log, "Visit 2 results",
           [("Held-out test crops", table_html(test[tcols]) if len(test) else "none"),
            ("Checkpoint selection (validation crops)", table_html(sel)),
            ("Real samples vs laser (fine-tuned)", table_html(real))])
    log("=== VISIT 2 DONE ===")


def analyze(inp, model=None, um_per_px=None, out=None):
    from .characterize import gather, run as characterize
    from .models import make
    if model is None:
        model = "cpsam_ft" if (ROOT / "models" / "cpsam_powder").exists() else None
    if model is None:
        v1 = latest("visit1")
        model = (v1 / "shootout" / "best_model.txt").read_text().strip() if v1 else "cpsam"
    run_dir, log = (Path(out), Log(Path(out) / "log.txt")) if out else new_run("analyze")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log(f"analyze {inp} with {model}")
    seg = make(model)
    samples, _ = characterize(gather(inp), seg, run_dir, um_per_px=um_per_px, log=log)
    seg.close()
    log(f"done: {run_dir / 'index.html'}")


def qcmodels(quick=False):
    import pandas as pd
    from . import data as data_mod
    from .qcmodels import PUBLISHED, run as train_bench, run_real_qc
    from .report import table_html
    run_dir, log = new_run("qcmodels", resume=False)
    log("=== QC MODELS: train + benchmark particle and satellite models, QC the real samples ===")
    env = envcheck(log)
    json.dump(env, open(run_dir / "env.json", "w"), indent=1)
    quick = quick or not env["cuda_available"]
    for f in (data_mod.get_aa7075, data_mod.get_alloy2):   # real samples only; public data fetched below
        try:
            f()
        except Exception as e:
            log(f"  data: {f.__name__} failed: {e}")
    bench, choice, sat_check = train_bench(run_dir, log=log, quick=quick)
    real, ex = (pd.DataFrame(), None)
    if len(choice) == 2:
        log("--- QC of the real samples")
        real, ex = run_real_qc(run_dir, log=log)
    bcols = [c for c in ["task", "model", "subset", "n_images", "n_gt", "P@0.5", "R@0.5", "F1@0.5", "count_ratio", "mean_iou",
                         "pixel_P", "pixel_R", "sec_per_image"] if c in bench]
    pub = pd.DataFrame([{"task": t, **v} for t, v in PUBLISHED.items()])
    md = [f"# QC models ({run_dir.name})", "", f"GPU: {env.get('gpu', 'NONE (CPU quick run)')}", "",
          f"Chosen: {json.dumps({k: v['name'] for k, v in choice.items()})}", "",
          "## Benchmark on published validation images", bench[bcols].round(3).to_string(index=False) if len(bench) else "(none)",
          "", "## Published", pub.to_string(index=False), "",
          "## % satellited particles vs hand labels", sat_check.round(3).to_string(index=False) if len(sat_check) else "(none)",
          "", "## QC of real samples", real.round(3).to_string(index=False) if len(real) else "(none)"]
    if ex is not None:
        md += ["", "## Example comparison: AA7075 base vs TiC surface-modified", ex.round(4).to_string(index=False)]
    (run_dir / "SUMMARY.md").write_text(chr(10).join(md), encoding="utf-8")
    (run_dir / "DONE").write_text(time.strftime("%c"))
    secs = [("Benchmark (published validation images)", table_html(bench[bcols]) if len(bench) else "none"),
            ("Published reference numbers", table_html(pub)),
            ("% satellited particles vs hand labels", table_html(sat_check) if len(sat_check) else "none"),
            ("QC of real samples", table_html(real) if len(real) else "none")]
    if ex is not None:
        secs.append(("Example comparison: AA7075 base vs TiC surface", table_html(ex)))
    finish(run_dir, log, "QC models", secs)
    log("=== QC MODELS DONE ===")


def qc_folder(inp, um_per_px=None, out=None):
    from .characterize import gather
    from .qc import run as qc_run
    from .report import simple_page, table_html
    run_dir, log = (Path(out), Log(Path(out) / "log.txt")) if out else new_run("qc")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log(f"QC of {inp}")
    samples, _, _ = qc_run(gather(inp), run_dir, um_per_px=um_per_px, log=log)
    log(chr(10) + samples.to_string(index=False))
    simple_page(Path(run_dir) / "index.html", f"Powder QC: {inp}", [("Samples (95% bootstrap CIs)", table_html(samples))])
    (RUNS / "LATEST.txt").write_text(Path(run_dir).name)
    return run_dir, samples


def compare_folders(a, b, um_per_px=None):
    import numpy as np
    from .characterize import gather
    from .qc import compare, run as qc_run
    from .report import simple_page, table_html
    run_dir, log = new_run("compare")
    log(f"compare A={a}  B={b}")
    res = {}
    from .qc import load_qc_models
    ps, ss, _ = load_qc_models()
    for tag, folder in (("A", a), ("B", b)):
        items = [(tag, p) for _, p in gather(folder)]   # each folder pooled as one sample
        samples, boots, points = qc_run(items, run_dir / tag, um_per_px=um_per_px, log=log, part_seg=ps, sat_seg=ss)
        res[tag] = (samples, boots[tag], points[tag])
    ps.close()
    ss.close()
    table = compare(res["A"][1], res["B"][1], res["A"][2], res["B"][2])
    table.to_csv(run_dir / "comparison.csv", index=False)
    log(chr(10) + table.round(4).to_string(index=False))
    simple_page(run_dir / "index.html", "Before / after comparison",
                [("Inputs", f"<p>A = {a}<br>B = {b}</p>"),
                 ("Difference B - A with 95% bootstrap CI", table_html(table)),
                 ("A", table_html(res["A"][0])), ("B", table_html(res["B"][0]))])
    (RUNS / "LATEST.txt").write_text(run_dir.name)
    _ = np
    return table


def show_status():
    from .labeling import status
    st = status()
    if not len(st):
        print("No labeling drafts yet (run visit 1).")
        return st
    print(st.groupby("split").reviewed.agg(["sum", "count"]).rename(columns={"sum": "reviewed", "count": "crops"}))
    return st


QC_USAGE = r"""
QC models are ready. In PowerShell, from %USERPROFILE%\sem-powder:
  .\run.cmd qc C:\path\to\images                    (one subfolder per sample)
  .\run.cmd compare C:\path\to\before C:\path\to\after
Results: runs\qc_<time>\ or runs\compare_<time>\ (index.html, CSVs, overlays)."""


def auto(quick=False):
    from .release import ensure_models
    if not ensure_models():
        print("Trained models could not be downloaded: training them here instead (~75 min on a GPU).")
        return qcmodels(quick)
    if latest("visit1") is None:
        print(QC_USAGE)
        return
    from .labeling import status
    st = status()
    tr = st[(st.split == "train") & st.reviewed] if len(st) else st
    if len(tr) >= 4:
        return visit2()
    print("Visit 1 is done. Next step is labeling: run `run.cmd label` (or python -m semchar label),")
    print("correct the draft outlines, save each with Ctrl+S, then run this again for visit 2.")
    show_status()


class _Tee:
    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def _session_log(stage):
    d = RUNS / "logs"
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{stage}.txt", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["envcheck", "data", "visit1", "drafts", "label", "status", "visit2", "analyze", "auto",
                                     "qcmodels", "qc", "compare", "getmodels"])
    ap.add_argument("input", nargs="?")
    ap.add_argument("input2", nargs="?")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--um-per-px", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.stage not in ("status", "label"):
        _session_log(a.stage)
        keep_awake(True)
    try:
        _dispatch(a)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


def _dispatch(a):
    if a.stage == "envcheck":
        envcheck()
    elif a.stage == "data":
        from . import data
        data.main()
    elif a.stage == "visit1":
        visit1(a.quick)
    elif a.stage == "drafts":
        from .labeling import make_drafts
        from .models import make
        v1 = latest("visit1")
        name = (v1 / "shootout" / "best_model.txt").read_text().strip() if v1 else "cpsam"
        seg = make(name if name.startswith("cpsam") else "cpsam")
        make_drafts(seg)
        seg.close()
    elif a.stage == "label":
        from .labeling import open_gui
        open_gui()
    elif a.stage == "status":
        show_status()
    elif a.stage == "visit2":
        visit2()
    elif a.stage == "analyze":
        if not a.input:
            sys.exit("usage: python -m semchar analyze <folder>")
        analyze(a.input, a.model, a.um_per_px, a.out)
    elif a.stage == "auto":
        auto(a.quick)
    elif a.stage == "qcmodels":
        qcmodels(a.quick)
    elif a.stage == "getmodels":
        from .release import ensure_models
        if not ensure_models():
            sys.exit(1)
        print(QC_USAGE)
    elif a.stage == "qc":
        if not a.input:
            sys.exit("usage: python -m semchar qc <folder>")
        qc_folder(a.input, a.um_per_px, a.out)
    elif a.stage == "compare":
        if not (a.input and a.input2):
            sys.exit("usage: python -m semchar compare <before_folder> <after_folder>")
        compare_folders(a.input, a.input2, a.um_per_px)


if __name__ == "__main__":
    main()

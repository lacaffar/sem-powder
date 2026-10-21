"""One fine-tuning attempt in its own process (python -m semchar._ft_worker cfg.json).

Running out of GPU memory during training can crash the CUDA runtime outright
(segfault on Windows), which no Python handler can catch. Running each attempt in
a child process lets the parent fall back to a lighter setting instead of dying.
Writes cfg['result'] JSON on success.
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    import semchar.models  # noqa: F401  (sets CELLPOSE_LOCAL_MODELS_PATH before cellpose is imported)
    import torch
    from cellpose import models, train as cptrain
    from semchar.finetune import _freeze
    from semchar.labeling import load_split
    import semchar.labeling as L
    if cfg.get("labels_dir"):
        from pathlib import Path
        L.LABELS = Path(cfg["labels_dir"])
    tr_i, tr_m, _ = load_split("train")
    va_i, va_m, _ = load_split("val")
    if torch.cuda.is_available():
        # Recent NVIDIA Windows drivers spill to system RAM instead of failing when GPU memory runs out
        # (sysmem fallback), making training crawl and the page file balloon. Cap PyTorch at the real VRAM
        # so overflow becomes a clean OutOfMemoryError and the parent falls back to a lighter setting.
        import os
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("SEMCHAR_VRAM_FRACTION", 0.92)))
    m = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam_v2", use_bfloat16=False)
    _freeze(m.net, cfg["frozen"])
    n_train = sum(p.numel() for p in m.net.parameters() if p.requires_grad)
    print(f"[worker] batch {cfg['batch_size']}, frozen blocks {cfg['frozen']}, trainable params {n_train / 1e6:.0f}M",
          flush=True)
    t0 = time.time()
    cptrain.train_seg(m.net, train_data=tr_i, train_labels=tr_m,
                      test_data=va_i or None, test_labels=va_m or None,
                      batch_size=cfg["batch_size"], learning_rate=cfg["lr"], weight_decay=0.1,
                      n_epochs=cfg["n_epochs"], save_path=cfg["run_dir"], save_every=cfg["save_every"],
                      save_each=True, nimg_per_epoch=max(16, len(tr_i)), model_name="cpsam_powder",
                      min_train_masks=1)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    json.dump({"ok": True, "minutes": (time.time() - t0) / 60, "peak_vram_gb": peak,
               "trainable_params_m": n_train / 1e6}, open(cfg["result"], "w"))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            print("[worker] out of GPU memory", flush=True)
            sys.exit(3)
        raise

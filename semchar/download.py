"""Downloads via curl.exe (ships with Windows 10+). Python's own HTTP clients get
Cloudflare 403s from Mendeley Data, curl does not (attempt1/PROGRESS.md)."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


def curl():
    c = shutil.which("curl") or shutil.which("curl.exe")
    if not c:
        raise RuntimeError("curl not found (Windows 10+ includes C:\\Windows\\System32\\curl.exe)")
    return c


def fetch(url, dst, sha256=None, timeout=3600):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    subprocess.run([curl(), "-sSL", "--retry", "3", "-m", str(timeout), "-A", "Mozilla/5.0",
                    "-o", str(tmp), url], check=True)
    if sha256 and sha256_of(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}")
    tmp.replace(dst)
    return dst


def get_json(url):
    import json
    out = subprocess.run([curl(), "-sSL", "--retry", "3", "-m", "120", "-A", "Mozilla/5.0", url],
                         check=True, capture_output=True)
    return json.loads(out.stdout.decode("utf-8"))


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

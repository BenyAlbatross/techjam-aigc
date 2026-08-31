"""Per-run logging. One directory per run, everything reproducible from it.

    runs/<utc>__<name>/
        config.json          resolved config + git SHA + dirty flag + seeds
        env.json             package versions, CUDA/driver, GPU, host
        manifest.json        dataset build hash, split hashes, per-cell row counts
        train_log.jsonl      per-step loss/lr/grad-norm, per-epoch val metrics
        predictions.parquet  per-image id/group/label/source/transform/severity/score/split
        metrics.json         point metrics, CIs, breakdowns
        report.md            generated

`predictions.parquet` is the source of truth: every table in every report is recomputed
from it, so no number can drift away from the predictions that produced it, and a new
breakdown never requires re-running the model.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(os.environ.get("ACAI_RUNS", "runs"))


def _git() -> dict:
    def sh(*a):
        try:
            return subprocess.check_output(a, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    return {"sha": sh("git", "rev-parse", "HEAD"),
            "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the SHA does not describe the code that ran. Recorded
            # rather than blocked, but recorded loudly.
            "dirty": bool(sh("git", "status", "--porcelain"))}


def _env() -> dict:
    e = {"python": sys.version.split()[0], "platform": platform.platform(),
         "machine": platform.machine(), "host": platform.node()}
    try:
        import torch
        e["torch"] = torch.__version__
        e["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            e["gpu"] = torch.cuda.get_device_name(0)
            e["capability"] = list(torch.cuda.get_device_capability(0))
            e["cuda"] = torch.version.cuda
    except Exception:
        pass
    for mod in ("timm", "numpy", "sklearn", "scipy", "pywt", "PIL"):
        try:
            e[mod] = __import__(mod).__version__
        except Exception:
            pass
    return e


def _jsonable(o):
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return o


class Run:
    """A single experiment run. Use as a context manager so failures are still recorded.

    A crashed run that leaves no trace is the one you rerun by accident; `status` is
    written either way.
    """

    def __init__(self, name: str, config: dict, root: Path | str = RUNS):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(root) / f"{stamp}__{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.config, self.t0 = name, config, time.time()
        self._log = open(self.dir / "train_log.jsonl", "a")
        self.write("config.json", {**_jsonable(config), "git": _git(), "name": name})
        self.write("env.json", _env())

    # ---------------------------------------------------------------- primitives
    def write(self, fname: str, obj) -> Path:
        p = self.dir / fname
        p.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True))
        return p

    def log(self, **kw) -> None:
        """Append one JSONL record (a step, or an epoch's val metrics)."""
        self._log.write(json.dumps(_jsonable({"t": round(time.time() - self.t0, 3), **kw})) + "\n")
        self._log.flush()

    def predictions(self, df) -> Path:
        """Persist per-image predictions. Required columns are checked, not assumed.

        Missing `group` would silently turn every grouped bootstrap into a row-level one,
        understating every CI in the report -- so it is a hard requirement, not a default.
        """
        need = {"id", "group", "label", "score", "split", "source", "transform", "severity"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"predictions missing required columns: {sorted(missing)}")
        p = self.dir / "predictions.parquet"
        df.to_parquet(p, index=False)
        return p

    def metrics(self, obj) -> Path:
        return self.write("metrics.json", obj)

    # ------------------------------------------------------------------- lifecycle
    def finish(self, status: str = "ok", **extra) -> None:
        self.write("status.json", {"status": status, "seconds": round(time.time() - self.t0, 1),
                                   **_jsonable(extra)})
        self._log.close()
        self._append_index(status)

    def _append_index(self, status: str) -> None:
        """One greppable line per run in runs/index.csv."""
        import csv
        idx = Path(self.dir).parent / "index.csv"
        m = {}
        mp = self.dir / "metrics.json"
        if mp.exists():
            try:
                d = json.loads(mp.read_text())
                m = d.get("overall", d) if isinstance(d, dict) else {}
            except Exception:
                pass
        row = {"run": self.dir.name, "name": self.name, "status": status,
               "seconds": round(time.time() - self.t0, 1),
               "fusion": self.config.get("fusion", ""),
               "variant": self.config.get("variant", ""),
               "img_size": self.config.get("img_size", ""),
               "auroc": m.get("auroc", ""), "auprc": m.get("auprc", ""),
               "tpr_at_fpr01": m.get("tpr_at_fpr01", ""),
               "git": _git()["sha"][:8]}
        new = not idx.exists()
        with open(idx, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish("ok" if exc_type is None else f"failed:{exc_type.__name__}")
        return False

"""Experiment tracking.

Every benchmark run writes a self-contained folder:

    outputs/runs/<YYYYmmdd-HHMMSS>_<target>_<run_name>/
        config.yaml          exact config used
        meta.json            git commit, data hash, package versions, timings
        cv_folds.csv         every model x every fold
        cv_summary.csv       mean/std per model  (this is the benchmark table)
        coverage_<model>.csv threshold -> coverage/accuracy table
        holdout_*.csv/json   final model on the untouched future data
        figures/*.png
        model.joblib         final model + threshold + classes (for predict.py)

and appends one row per model to outputs/leaderboard.csv, so all runs ever made
can be compared in one table. If MLflow is enabled it gets the same info
(browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

_PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "mlflow"]


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True, timeout=5).stdout.strip()
    except Exception:
        return None  # not a git repo / git not installed: that's fine


def file_hash(path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


class Run:
    def __init__(self, cfg: dict, target: str, run_name: str):
        self.cfg, self.target, self.run_name = cfg, target, run_name
        self.started = datetime.now()
        self.id = f"{self.started:%Y%m%d-%H%M%S}_{target}_{run_name}"
        self.dir = Path(cfg["tracking"]["runs_dir"]) / self.id
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.meta = {
            "run_id": self.id,
            "target": target,
            "run_name": run_name,
            "started": self.started.isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")) if _git("rev-parse", "HEAD") else None,
            "data_path": cfg["data"]["path"],
            "data_sha256": file_hash(cfg["data"]["path"]),
            "versions": _versions(),
        }
        self._mlflow = None
        if cfg["tracking"].get("mlflow"):
            self._start_mlflow()

    # -- files ---------------------------------------------------------------
    def save_df(self, df: pd.DataFrame, name: str, index: bool = False):
        df.to_csv(self.dir / name, index=index)

    def save_json(self, obj: dict, name: str):
        with open(self.dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    def figure_path(self, name: str) -> Path:
        return self.dir / "figures" / name

    # -- leaderboard ---------------------------------------------------------
    def append_leaderboard(self, summary: pd.DataFrame, extra: dict):
        """One row per model: run info + CV means/stds + whatever is in extra."""
        path = Path(self.cfg["tracking"]["leaderboard"])
        rows = summary.copy()
        rows.insert(0, "run_id", self.id)
        rows.insert(1, "target", self.target)
        rows.insert(2, "run_name", self.run_name)
        rows.insert(3, "git_commit", self.meta["git_commit"])
        rows.insert(4, "data_sha256", self.meta["data_sha256"])
        for k, v in extra.items():
            rows[k] = rows["model"].map(v) if isinstance(v, dict) else v
        if path.exists():
            rows = pd.concat([pd.read_csv(path), rows], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False)

    # -- mlflow (optional, never allowed to crash a run) ---------------------
    def _start_mlflow(self):
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow
            mlflow.set_tracking_uri(self.cfg["tracking"]["mlflow_tracking_uri"])
            mlflow.set_experiment(f"{self.cfg['project']}-{self.target}")
            mlflow.start_run(run_name=self.id)
            mlflow.set_tags({"git_commit": self.meta["git_commit"],
                             "data_sha256": self.meta["data_sha256"],
                             "run_name": self.run_name})
            self._mlflow = mlflow
        except Exception as e:
            warnings.warn(f"MLflow disabled for this run: {e}")

    def log_model_result(self, model: str, params: dict, metrics: dict):
        if not self._mlflow:
            return
        try:
            with self._mlflow.start_run(run_name=model, nested=True):
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                          if pd.notna(v)})
        except Exception as e:
            warnings.warn(f"MLflow logging failed for {model}: {e}")

    def finish(self, status: str = "FINISHED"):
        self.meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self.meta["duration_s"] = round((datetime.now() - self.started).total_seconds(), 1)
        self.meta["status"] = status
        self.save_json(self.meta, "meta.json")
        if self._mlflow:
            try:
                self._mlflow.log_artifacts(str(self.dir))
                self._mlflow.end_run(status=status)
            except Exception as e:
                warnings.warn(f"MLflow finish failed: {e}")

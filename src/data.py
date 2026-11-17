"""Loading, cleaning, label handling and the time-based split.

Rule of thumb for what goes where:
  * Here (data.py / features.py): STATELESS row-by-row work, e.g. fixing
    "xyz.ABC" -> "xyz.abc" or deriving month from a date. Safe before splitting.
  * Inside the sklearn pipeline (models.py): anything LEARNED from the data,
    e.g. imputation values, scaling, category encodings. Learned per CV fold so
    nothing leaks from validation/future rows into training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

OTHER = "__other__"


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
def load_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    df[d["date_column"]] = pd.to_datetime(df[d["date_column"]], format=d.get("date_format"))
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stateless text cleanup on every string column (features and targets)."""
    c = cfg.get("cleaning", {})
    strip, lower = c.get("strip_whitespace", True), c.get("lowercase", True)

    def fix(v):
        if pd.isna(v):
            return np.nan
        v = str(v)
        if strip:
            v = " ".join(v.split())
        if lower:
            v = v.lower()
        return v or np.nan

    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(fix).astype(object)
    for col, mapping in (c.get("value_maps") or {}).items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def load_clean(cfg: dict) -> pd.DataFrame:
    df = clean(load_raw(cfg), cfg)
    return df.sort_values(cfg["data"]["date_column"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
def parse_labels(series: pd.Series, sep: str) -> pd.Series:
    """'a; b' -> ['a', 'b']. Empty/missing -> []."""
    def _split(v):
        if pd.isna(v):
            return []
        return [p.strip() for p in str(v).split(sep) if p.strip()]
    return series.map(_split)


@dataclass
class TargetData:
    y: np.ndarray                 # encoded primary label, shape (n,)
    label_sets: list[set[int]]    # encoded set of ALL labels per row
    classes: list[str]            # index -> class name


def encode_target(label_lists: pd.Series, train_mask: np.ndarray, min_count: int) -> TargetData:
    """Encode labels. Class vocabulary + rare-class merging use TRAIN rows only."""
    primary = label_lists.map(lambda ls: ls[0])
    counts = primary[train_mask].value_counts()
    keep = sorted(counts[counts >= min_count].index)
    classes = keep + ([OTHER] if len(keep) < len(set(primary)) else [])
    index = {c: i for i, c in enumerate(classes)}
    other_idx = index.get(OTHER)

    def enc(label):
        return index.get(label, other_idx)

    y = primary.map(enc).to_numpy()
    sets = [{enc(l) for l in ls} for ls in label_lists]
    return TargetData(y=y.astype(int), label_sets=sets, classes=classes)


# ----------------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------------
def time_holdout_mask(n: int, test_size: float) -> np.ndarray:
    """Data is already sorted by date: last `test_size` fraction is the holdout."""
    n_test = int(round(n * test_size))
    mask = np.ones(n, dtype=bool)
    mask[n - n_test:] = False
    return mask  # True = train/CV, False = holdout


def cv_splits(n_train: int, n_splits: int):
    """Expanding window: train on the past, validate on the next chunk."""
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_train)))

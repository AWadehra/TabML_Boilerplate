"""Quick data/target health check. Run this before benchmarking.

    python -m src.check_data

Prints (and saves to outputs/data_check/):
  * column types, cardinality, missing %
  * per target: class balance, multi-label share, rare classes
  * per target: class share by quarter (spots drift, and classes that only
    exist late in time, which the model can't learn from the past)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import load_clean, parse_labels, time_holdout_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    out = Path("outputs/data_check")
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160, "display.max_rows", 200)

    df = load_clean(cfg)
    date = df[d["date_column"]]
    print(f"{len(df)} rows x {df.shape[1]} cols | {date.min():%Y-%m-%d} -> {date.max():%Y-%m-%d}\n")

    cols = pd.DataFrame({"dtype": df.dtypes.astype(str), "n_unique": df.nunique(),
                         "pct_missing": (100 * df.isna().mean()).round(2)})
    print(cols, "\n")
    cols.to_csv(out / "columns.csv")

    train_mask = time_holdout_mask(len(df), cfg["split"]["test_size"])
    for target in d["targets"]:
        labels = parse_labels(df[target], d["label_separator"])
        n_labels = labels.map(len)
        exploded = labels.explode().dropna()
        primary = labels.map(lambda ls: ls[0] if ls else None)

        bal = pd.DataFrame({
            "n_primary": primary.value_counts(),
            "n_any": exploded.value_counts(),
            "n_primary_train": primary[train_mask].value_counts(),
            "n_primary_holdout": primary[~train_mask].value_counts(),
        }).fillna(0).astype(int).sort_values("n_primary", ascending=False)
        bal["pct_primary"] = (100 * bal["n_primary"] / bal["n_primary"].sum()).round(2)
        bal.to_csv(out / f"{target}_class_balance.csv")

        rare = bal[bal["n_primary_train"] < d["min_class_count"]]
        print(f"=== {target} ===")
        print(f"labels per row: {n_labels.value_counts().sort_index().to_dict()} "
              f"({(n_labels > 1).mean():.1%} multi-label)")
        print(f"{len(bal)} classes; largest {bal['pct_primary'].iloc[0]:.1f}%, "
              f"imbalance ratio (largest/smallest) {bal['n_primary'].max() / max(bal['n_primary'].min(), 1):.0f}x")
        print(f"{len(rare)} classes have < {d['min_class_count']} training rows "
              f"-> merged into '__other__'")
        print(bal.head(30), "\n")

        quarterly = pd.crosstab(date.dt.to_period("Q"), primary, normalize="index").round(3)
        quarterly.to_csv(out / f"{target}_share_by_quarter.csv")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()

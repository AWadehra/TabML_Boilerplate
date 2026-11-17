"""Score new rows with a saved model and flag which ones need an analyst.

    python -m src.predict --run outputs/runs/<run_id> --input new.csv --output preds.csv

Output columns: pred_1..3 / proba_1..3 (top-3 suggestions), confidence,
needs_review (True = low confidence, send to the analyst).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import clean
from src.features import build_features


def predict(bundle: dict, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    cfg = bundle["config"]
    date_col = cfg["data"]["date_column"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format=cfg["data"].get("date_format"))
    df = build_features(clean(df, cfg), cfg)
    proba = bundle["model"].predict_proba(df[bundle["feature_columns"]])
    classes = np.array(bundle["classes"])
    order = np.argsort(-proba, axis=1)[:, :top_k]

    out = pd.DataFrame(index=df.index)
    for k in range(order.shape[1]):
        out[f"pred_{k + 1}"] = classes[order[:, k]]
        out[f"proba_{k + 1}"] = np.take_along_axis(proba, order[:, [k]], axis=1).ravel().round(4)
    out["confidence"] = proba.max(axis=1).round(4)
    thr = bundle["threshold"]
    out["needs_review"] = True if np.isnan(thr) else out["confidence"] < thr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder containing model.joblib")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()

    bundle = joblib.load(Path(args.run) / "model.joblib")
    df = pd.read_csv(args.input)
    preds = predict(bundle, df)
    pd.concat([df, preds], axis=1).to_csv(args.output, index=False)
    print(f"{len(preds)} rows scored with {bundle['model_name']} (threshold "
          f"{bundle['threshold']:.2f}); {preds['needs_review'].mean():.1%} need review "
          f"-> {args.output}")


if __name__ == "__main__":
    main()

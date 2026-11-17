"""Feature engineering (stateless) and the per-fold preprocessors (learned).

Put your own engineered features in `build_features`. Keep them row-by-row
(no statistics over the whole dataset) so they can't leak future information.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    date_col = cfg["data"]["date_column"]
    if cfg["features"].get("add_date_parts"):
        d = df[date_col]
        df["date_month"] = d.dt.month
        df["date_dayofweek"] = d.dt.dayofweek
        df["date_quarter"] = d.dt.quarter

    # --- Add your own features here, e.g. ---------------------------------
    # df["domain"] = df["some_code"].str.split(".").str[0]   # "xyz.abc" -> "xyz"
    # df["amount_per_item"] = df["amount"] / df["items"].clip(lower=1)
    # ----------------------------------------------------------------------
    return df


def feature_columns(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) feature columns."""
    d, f = cfg["data"], cfg["features"]
    excluded = set(d["targets"]) | set(d.get("drop_columns") or []) | {d["date_column"]}
    candidates = [c for c in df.columns if c not in excluded]

    cat = f.get("categorical_columns")
    num = f.get("numeric_columns")
    if cat is None:
        cat = [c for c in candidates
               if (num is None or c not in num)
               and not pd.api.types.is_numeric_dtype(df[c])]
    if num is None:
        num = [c for c in candidates if c not in cat and pd.api.types.is_numeric_dtype(df[c])]
    return list(cat), list(num)


def make_preprocessor(kind: str, cat: list[str], num: list[str]) -> ColumnTransformer:
    cat_impute = SimpleImputer(strategy="constant", fill_value="__missing__")
    if kind == "linear":
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        cat_pipe = Pipeline([("impute", cat_impute),
                             ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                      min_frequency=5, sparse_output=False))])
    elif kind == "tree":
        num_pipe = SimpleImputer(strategy="median")
        cat_pipe = Pipeline([("impute", cat_impute),
                             ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                       unknown_value=-1))])
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind}")
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)],
                             remainder="drop", verbose_feature_names_out=False)

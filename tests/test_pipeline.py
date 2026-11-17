"""End-to-end smoke test on synthetic data:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import clean, encode_target, parse_labels


def make_fake_data(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat1 = rng.choice(["xyz.abc", "xyz.ABC ", "foo.bar", "baz.qux"], n)
    cat2 = rng.choice(["red", "green", "blue", None], n, p=[.4, .3, .29, .01])
    num1 = rng.normal(size=n)
    num2 = rng.uniform(0, 100, n)
    base = np.where(num1 > 0.5, "class_a", np.where(np.char.lower(np.char.strip(cat1.astype(str))) == "xyz.abc", "class_b", "class_c"))
    base = np.where(rng.random(n) < 0.03, "rare_class", base)
    target_a = [f"{b};class_c" if rng.random() < 0.05 else b for b in base]
    target_b = np.where(num2 > 50, "high", "low")
    return pd.DataFrame({
        "report_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "cat1": cat1, "cat2": cat2, "num1": num1, "num2": num2,
        "target_a": target_a, "target_b": target_b,
    })


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_fake_data().to_csv(tmp_path / "data.csv", index=False)
    c = load_config(__file__.replace("tests/test_pipeline.py", "config.yaml"))
    c["data"]["path"] = str(tmp_path / "data.csv")
    c["split"]["n_splits"] = 3
    c["tracking"]["mlflow"] = False
    c["importance"]["n_repeats"] = 2
    for name in ("random_forest", "xgboost", "lightgbm"):  # keep the test fast
        c["models"][name]["enabled"] = False
    return c


def test_cleaning_merges_case_variants(cfg):
    df = clean(pd.DataFrame({"c": ["xyz.ABC ", "xyz.abc", None]}), cfg)
    assert df["c"].tolist()[:2] == ["xyz.abc", "xyz.abc"]
    assert pd.isna(df["c"].iloc[2])


def test_rare_and_unseen_classes_go_to_other():
    labels = pd.Series([["a"], ["a"], ["b"], ["c"]])
    td = encode_target(labels, np.array([True, True, True, False]), min_count=2)
    assert td.classes == ["a", "__other__"]
    assert td.y.tolist() == [0, 0, 1, 1]


def test_parse_labels():
    assert parse_labels(pd.Series(["a; b", None]), ";").tolist() == [["a", "b"], []]


def test_benchmark_and_predict(cfg):
    from src.benchmark import run_target
    from src.predict import predict
    import joblib
    from pathlib import Path

    run_id = run_target(cfg, "target_a", "smoke")
    run_dir = Path(cfg["tracking"]["runs_dir"]) / run_id
    for f in ["config.yaml", "meta.json", "cv_folds.csv", "cv_summary.csv",
              "holdout_metrics.json", "model.joblib", "figures/coverage_vs_accuracy.png"]:
        assert (run_dir / f).exists(), f

    summary = pd.read_csv(run_dir / "cv_summary.csv")
    assert set(summary["model"]) == {"dummy", "logreg", "hist_gb"}
    best = summary.set_index("model")["f1_macro_mean"]
    assert best.max() > best["dummy"]  # real models beat the baseline

    lb = pd.read_csv(cfg["tracking"]["leaderboard"])
    assert len(lb) == 3 and lb["is_final"].sum() == 1
    assert json.load(open(run_dir / "meta.json"))["status"] == "FINISHED"

    preds = predict(joblib.load(run_dir / "model.joblib"), make_fake_data(20, seed=1))
    assert {"pred_1", "proba_1", "confidence", "needs_review"} <= set(preds.columns)

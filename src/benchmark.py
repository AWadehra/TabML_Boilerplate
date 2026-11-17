"""Benchmark every enabled model on the same time-based CV folds.

    python -m src.benchmark                          # all targets in config
    python -m src.benchmark --target target_a --name added_domain_feature

Steps per target:
  1. load + clean + build features, sort by date
  2. last `test_size` of time = holdout (not touched until step 5)
  3. expanding-window CV on the rest, every model on identical folds
  4. pick a confidence threshold per model from out-of-fold predictions
  5. refit the chosen model on all non-holdout data, score it on the holdout
  6. save everything in a run folder + append to the leaderboard
"""
from __future__ import annotations

import argparse
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src import evaluate as ev
from src.config import load_config
from src.data import (cv_splits, encode_target, load_clean, parse_labels,
                      time_holdout_mask)
from src.features import build_features, feature_columns
from src.models import build_models


def run_target(cfg: dict, target: str, run_name: str) -> str:
    d, conf_cfg = cfg["data"], cfg["confidence"]

    # 1. data -------------------------------------------------------------------
    df = build_features(load_clean(cfg), cfg)
    labels = parse_labels(df[target], d["label_separator"])
    keep = (labels.map(len) > 0).to_numpy()
    if (~keep).any():
        print(f"[{target}] dropping {(~keep).sum()} rows with no label")
    df, labels = df[keep].reset_index(drop=True), labels[keep].reset_index(drop=True)

    # 2. split + target encoding (vocabulary from train only) ----------------
    train_mask = time_holdout_mask(len(df), cfg["split"]["test_size"])
    td = encode_target(labels, train_mask, d["min_class_count"])
    cat, num = feature_columns(df, cfg)
    X = df[cat + num]
    X_tr, X_te = X[train_mask].reset_index(drop=True), X[~train_mask].reset_index(drop=True)
    y_tr, y_te = td.y[train_mask], td.y[~train_mask]
    sets = np.array(td.label_sets, dtype=object)
    sets_tr, sets_te = list(sets[train_mask]), list(sets[~train_mask])
    K = len(td.classes)
    date = df[d["date_column"]]

    from src.tracking import Run  # imported late so `--help` stays fast
    run = Run(cfg, target, run_name)
    run.meta.update({
        "n_rows": len(df), "n_train": int(train_mask.sum()), "n_test": int((~train_mask).sum()),
        "train_period": [str(date[train_mask].min()), str(date[train_mask].max())],
        "test_period": [str(date[~train_mask].min()), str(date[~train_mask].max())],
        "n_classes": K, "categorical": cat, "numeric": num,
        "multi_label_rows_pct": round(100 * float((labels.map(len) > 1).mean()), 2),
    })
    class_counts = pd.DataFrame({
        "class": td.classes,
        "n_train": np.bincount(y_tr, minlength=K),
        "n_test": np.bincount(y_te, minlength=K),
    }).sort_values("n_train", ascending=False)
    run.save_df(class_counts, "class_counts.csv")
    print(f"[{target}] {len(df)} rows | train {len(y_tr)} / holdout {len(y_te)} | "
          f"{K} classes | {len(cat)} categorical + {len(num)} numeric features")

    # 3. cross-validation ----------------------------------------------------
    folds = cv_splits(len(X_tr), cfg["split"]["n_splits"])
    oof_idx = np.concatenate([va for _, va in folds])
    models = build_models(cfg, cat, num, K)
    fold_rows, summary_rows, curves, thresholds = [], [], {}, {}

    for name, model in models.items():
        oof = np.zeros((len(X_tr), K))
        for k, (tr, va) in enumerate(folds):
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = clone(model).fit(X_tr.iloc[tr], y_tr[tr])
            fit_s = time.perf_counter() - t0
            p = m.predict_proba(X_tr.iloc[va])
            oof[va] = p
            fold_rows.append({"model": name, "fold": k, "n_train": len(tr), "n_val": len(va),
                              "fit_s": round(fit_s, 2),
                              **ev.compute_metrics(y_tr[va], p, [sets_tr[i] for i in va])})

        # 4. confidence threshold from pooled out-of-fold predictions -------
        curve = ev.coverage_curve(oof[oof_idx], [sets_tr[i] for i in oof_idx])
        thr, cov = ev.pick_threshold(curve, conf_cfg["target_accuracy"], conf_cfg["min_rows"])
        curves[name], thresholds[name] = curve, thr
        run.save_df(curve, f"coverage_{name}.csv")

        f = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        row = {"model": name}
        for metric in [c for c in f.columns if c not in ("model", "fold", "n_train", "n_val")]:
            row[f"{metric}_mean"] = f[metric].mean()
            row[f"{metric}_std"] = f[metric].std()
        row["coverage_at_target"] = cov
        row["confidence_threshold"] = thr
        summary_rows.append(row)
        run.log_model_result(name, {"preprocessor": cfg["models"][name]["preprocessor"],
                                    **(cfg["models"][name].get("params") or {})},
                             {k: v for k, v in row.items() if k != "model"})
        print(f"  {name:<15} f1_macro {row['f1_macro_mean']:.3f} ± {row['f1_macro_std']:.3f} | "
              f"any-label acc {row['any_label_accuracy_mean']:.3f} | "
              f"auto-handle {cov:5.1%} @ {conf_cfg['target_accuracy']:.0%} acc")

    folds_df = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    means = [c for c in summary.columns if c.endswith("_mean") and c != "fit_s_mean"]
    stds = [c for c in summary.columns if c.endswith("_std") and c != "fit_s_std"]
    summary = summary[["model", "coverage_at_target", "confidence_threshold", *means, *stds,
                       "fit_s_mean", "fit_s_std"]].round(4)
    primary = cfg["selection"]["primary_metric"]
    sort_col = primary if primary in summary.columns else f"{primary}_mean"
    summary = summary.sort_values(sort_col, ascending=False).reset_index(drop=True)
    run.save_df(folds_df, "cv_folds.csv")
    run.save_df(summary, "cv_summary.csv")
    ev.plot_coverage_curves(curves, conf_cfg["target_accuracy"],
                            run.figure_path("coverage_vs_accuracy.png"), conf_cfg["min_rows"])

    # 5. final model on the holdout ------------------------------------------
    final = cfg["selection"]["final_model"]
    final = summary.loc[0, "model"] if final == "auto" else final
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = clone(models[final]).fit(X_tr, y_tr)
    p_te = final_model.predict_proba(X_te)
    holdout = ev.compute_metrics(y_te, p_te, sets_te)
    holdout_conf = ev.apply_threshold(p_te, sets_te, thresholds[final])
    run.save_json({"model": final, "metrics": holdout, "confidence": holdout_conf},
                  "holdout_metrics.json")
    run.save_df(ev.per_class_report(y_te, p_te, td.classes), "holdout_per_class.csv", index=True)
    ev.plot_confusion(y_te, p_te, td.classes, run.figure_path("holdout_confusion.png"))

    if cfg.get("importance", {}).get("enabled", True):
        r = permutation_importance(final_model, X_te, y_te, scoring="f1_macro",
                                   n_repeats=cfg["importance"].get("n_repeats", 5),
                                   random_state=cfg.get("seed", 42), n_jobs=-1)
        imp = pd.DataFrame({"feature": X_te.columns, "importance_mean": r.importances_mean,
                            "importance_std": r.importances_std}
                           ).sort_values("importance_mean", ascending=False)
        run.save_df(imp, "holdout_importance.csv")
        ev.plot_importance(imp, run.figure_path("holdout_importance.png"))

    joblib.dump({"model": final_model, "model_name": final, "classes": td.classes,
                 "threshold": thresholds[final], "feature_columns": cat + num,
                 "target": target, "config": cfg, "run_id": run.id},
                run.dir / "model.joblib")

    # 6. leaderboard -----------------------------------------------------------
    run.append_leaderboard(summary, {
        "n_classes": K, "n_train": len(y_tr), "n_test": len(y_te),
        "is_final": {final: True},
        **{f"holdout_{k}": {final: v} for k, v in {**holdout, **holdout_conf}.items()},
    })
    run.meta["final_model"] = final
    run.finish()

    t = holdout_conf
    print(f"\n[{target}] final = {final} | holdout f1_macro {holdout['f1_macro']:.3f}, "
          f"any-label acc {holdout['any_label_accuracy']:.3f}")
    if np.isnan(t["threshold"]):
        print(f"  no threshold reached {conf_cfg['target_accuracy']:.0%} accuracy on CV, "
              "so everything goes to the analyst")
    else:
        print(f"  confidence >= {t['threshold']:.2f}: {t['high_conf_share']:.1%} of holdout cases "
              f"auto-handled at {t['high_conf_accuracy']:.1%} accuracy; "
              f"{t['low_conf_share']:.1%} go to the analyst")
    print(f"  results: {run.dir}\n")
    return run.id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--target", help="one target column (default: all in config)")
    ap.add_argument("--name", default="baseline", help="short label for this run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for target in [args.target] if args.target else cfg["data"]["targets"]:
        run_target(cfg, target, args.name)


if __name__ == "__main__":
    main()

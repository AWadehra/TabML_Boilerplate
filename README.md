# opl-ml

A benchmark for classifying the target columns in `training_dataset.csv`. It
also gives each prediction a **confidence level**, so high-confidence cases can
skip manual review and the low-confidence ones go to an analyst.

## Layout

```
opl-ml/
├── config.yaml            # all settings: columns, split, models, thresholds
├── requirements.txt
├── data/processed/training_dataset.csv
├── notebooks/             # EDA (import from src/, don't copy code)
├── src/
│   ├── config.py          # loads config.yaml
│   ├── data.py            # load, clean (xyz.ABC -> xyz.abc), labels, time split
│   ├── features.py        # build_features() + encoders used inside each fold
│   ├── models.py          # model zoo
│   ├── evaluate.py        # metrics, confidence/coverage analysis, plots
│   ├── tracking.py        # run folders, leaderboard.csv, MLflow
│   ├── check_data.py      # CLI: class balance, multi-label share, drift
│   ├── benchmark.py       # CLI: run the benchmark
│   └── predict.py         # CLI: score new rows, flag ones needing review
├── outputs/
│   ├── leaderboard.csv    # one row per model per run, across ALL runs
│   ├── runs/<run_id>/     # everything about a single run
│   └── data_check/
└── tests/test_pipeline.py # end-to-end test on synthetic data
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                    # check the setup works (synthetic data)
# edit the TODOs in config.yaml (date column, targets, label separator)
python -m src.check_data               # class balance, multi-label rows, drift
python -m src.benchmark                # all targets, all models
python -m src.benchmark --target target_a --name domain_feature   # one experiment
mlflow ui --backend-store-uri sqlite:///mlflow.db                 # optional UI
```

Run all commands from the project root.

## How the evaluation works

- **Time-based split.** The data is sorted by date. The last 20% is a holdout
  that is only used once, to score the final model. The other 80% gets
  expanding-window CV: each fold trains on the past and validates on the next
  period. This mirrors how the model will be used, which is predicting the future.
- **Multiple labels per row.** The model trains on the first label. A
  prediction counts as correct (`any_label_accuracy`) if it matches any of the
  row's labels.
- **Rare and new classes.** A class with fewer than `min_class_count` rows in
  the training data becomes `__other__`.
- **Confidence.** For each model, `confidence_threshold` is the lowest
  predicted probability at which accuracy on out-of-fold predictions stays at
  or above `confidence.target_accuracy`. `coverage_at_target` is the share of
  cases above that threshold, i.e. the share an analyst would not need to
  review. The final model's threshold is then checked on the holdout
  (`holdout_high_conf_accuracy`) to confirm it still holds on future data.
  The **accuracy vs coverage** plot shows this trade-off for every model.

## Tracking results

Each `python -m src.benchmark` call creates `outputs/runs/<time>_<target>_<name>/`
with the config used, the git commit (if the project is a git repo), a hash of
the data file, per-fold scores, a summary table, the holdout results, figures
and the saved model. It also appends one row per model to
`outputs/leaderboard.csv`. Open that file in Excel or pandas to compare runs.
Give each experiment a meaningful `--name`.

## What's next

1. Put the real column names in `config.yaml` and run `check_data`.
2. Run a baseline benchmark. Then add features in `features.build_features`
   (and fixes in `cleaning.value_maps`), and rerun with a new `--name`.
3. Pick the metric that matters to the business (`selection.primary_metric`)
   and the accuracy the analysts need (`confidence.target_accuracy`).
4. Only then tune the best 1–2 models. Default settings are usually close
   enough while the features are still changing.

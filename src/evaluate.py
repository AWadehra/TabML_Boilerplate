"""Metrics, confidence analysis and figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss)

# Higher is better for every metric reported here (log loss is negated).
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
           "any_label_accuracy", "top3_accuracy", "neg_log_loss",
           "coverage_at_target", "confidence_threshold"]


def any_label_hits(proba: np.ndarray, label_sets: list[set[int]]) -> np.ndarray:
    """1 if the top prediction is ANY of the row's true labels."""
    pred = proba.argmax(axis=1)
    return np.array([p in s for p, s in zip(pred, label_sets)])


def compute_metrics(y, proba, label_sets) -> dict:
    n_classes = proba.shape[1]
    pred = proba.argmax(axis=1)
    top3 = np.argsort(-proba, axis=1)[:, :3]
    p = np.clip(proba, 1e-15, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "any_label_accuracy": any_label_hits(proba, label_sets).mean(),
        "top3_accuracy": np.mean([t in row for t, row in zip(y, top3)]),
        "neg_log_loss": -log_loss(y, p, labels=np.arange(n_classes)),
    }


# ----------------------------------------------------------------------------
# Confidence: how much work can be taken off the analyst?
# ----------------------------------------------------------------------------
def coverage_curve(proba, label_sets) -> pd.DataFrame:
    """For each threshold t: share of rows with max-prob >= t (coverage) and
    the accuracy on those rows. Correct = top prediction in the row's labels."""
    conf = proba.max(axis=1)
    hit = any_label_hits(proba, label_sets)
    rows = []
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        m = conf >= t
        rows.append({"threshold": t, "coverage": m.mean(), "n_rows": int(m.sum()),
                     "accuracy": hit[m].mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, target_acc: float, min_rows: int) -> tuple[float, float]:
    """Lowest threshold whose accuracy >= target (=> max coverage).
    Returns (threshold, coverage); (nan, 0) if the target is never reached."""
    ok = curve[(curve["accuracy"] >= target_acc) & (curve["n_rows"] >= min_rows)]
    if ok.empty:
        return float("nan"), 0.0
    best = ok.iloc[0]
    return float(best["threshold"]), float(best["coverage"])


def apply_threshold(proba, label_sets, threshold: float) -> dict:
    """How a fixed threshold performs on new data (e.g. the holdout)."""
    conf = proba.max(axis=1)
    hit = any_label_hits(proba, label_sets)
    high = conf >= threshold if not np.isnan(threshold) else np.zeros(len(conf), bool)
    return {
        "threshold": threshold,
        "high_conf_share": high.mean(),
        "high_conf_accuracy": hit[high].mean() if high.any() else np.nan,
        "low_conf_share": (~high).mean(),
        "low_conf_accuracy": hit[~high].mean() if (~high).any() else np.nan,
    }


def per_class_report(y, proba, classes) -> pd.DataFrame:
    pred = proba.argmax(axis=1)
    rep = classification_report(y, pred, labels=np.arange(len(classes)),
                                target_names=classes, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


# ----------------------------------------------------------------------------
# Figures (matplotlib, saved as PNG)
# ----------------------------------------------------------------------------
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_coverage_curves(curves: dict[str, pd.DataFrame], target_acc: float, path,
                         min_rows: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for i, (name, c) in enumerate(curves.items()):
        c = c[c["n_rows"] >= min_rows]  # tiny groups are just noise
        ax.plot(c["coverage"], c["accuracy"], lw=2, color=_SERIES[i % len(_SERIES)], label=name)
    ax.axhline(target_acc, color=_MUTED, lw=1, ls="--")
    ax.text(0.01, target_acc, f" target {target_acc:.0%}", va="bottom", fontsize=8, color=_MUTED)
    ax.set_xlabel("Share of cases auto-handled (coverage)", color=_INK, fontsize=9)
    ax.set_ylabel("Accuracy on those cases", color=_INK, fontsize=9)
    ax.set_xlim(0, 1)
    _style(ax, "Accuracy vs share auto-handled (out-of-fold CV)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y, proba, classes, path, max_classes: int = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = proba.argmax(axis=1)
    labels = np.arange(len(classes))
    if len(classes) > max_classes:  # keep the most frequent classes readable
        labels = pd.Series(y).value_counts().index[:max_classes].sort_values().to_numpy()
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    names = [classes[i] for i in labels]
    size = max(5, 0.35 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size), dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True (row-normalised)", fontsize=9)
    ax.set_title("Holdout confusion matrix", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path, top: int = 25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(imp) + 1.5), dpi=150)
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"],
            color=_SERIES[0], height=0.6, error_kw={"ecolor": _MUTED, "lw": 1})
    ax.set_xlabel("Drop in macro F1 when the column is shuffled", fontsize=9)
    _style(ax, "Permutation importance (holdout)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

"""Model zoo. Add a model by adding an entry to `_FACTORIES` and to config.yaml."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline

from src.features import make_preprocessor


def _dummy(p):
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(**p)


def _logreg(p):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**p)


def _rf(p):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**p)


def _hgb(p):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**p)


def _xgb(p):
    from xgboost import XGBClassifier
    return XGBClassifier(**p)


def _lgbm(p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**p)


_FACTORIES = {
    "dummy": _dummy,
    "logreg": _logreg,
    "random_forest": _rf,
    "hist_gb": _hgb,
    "xgboost": _xgb,
    "lightgbm": _lgbm,
}

# Models that accept a random_state; seeded from config for reproducibility.
_SEEDABLE = {"logreg", "random_forest", "hist_gb", "xgboost", "lightgbm"}


class LabelSafeClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a classifier so it always outputs probabilities for ALL classes.

    With a time-based split, early folds may not contain every class. Some
    libraries (XGBoost) refuse non-contiguous labels, and every library returns
    a narrower probability matrix. This wrapper re-indexes labels for fitting
    and pads missing classes with probability 0.
    """

    def __init__(self, estimator, n_classes: int):
        self.estimator = estimator
        self.n_classes = n_classes

    def fit(self, X, y):
        y = np.asarray(y)
        self.present_ = np.unique(y)
        self.estimator_ = clone(self.estimator).fit(X, np.searchsorted(self.present_, y))
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, X):
        p = self.estimator_.predict_proba(X)
        out = np.zeros((p.shape[0], self.n_classes))
        out[:, self.present_] = p
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_models(cfg: dict, cat: list[str], num: list[str], n_classes: int) -> dict:
    """Return {name: unfitted LabelSafeClassifier(pipeline)} for enabled models."""
    models = {}
    for name, spec in cfg["models"].items():
        if not spec.get("enabled", True):
            continue
        if name not in _FACTORIES:
            warnings.warn(f"No factory for model '{name}', skipping.")
            continue
        params = dict(spec.get("params") or {})
        if name in _SEEDABLE:
            params.setdefault("random_state", cfg.get("seed", 42))
        try:
            est = _FACTORIES[name](params)
        except ImportError as e:
            warnings.warn(f"Skipping {name}: {e}")
            continue
        pipe = Pipeline([("prep", make_preprocessor(spec["preprocessor"], cat, num)),
                         ("model", est)])
        models[name] = LabelSafeClassifier(pipe, n_classes)
    return models

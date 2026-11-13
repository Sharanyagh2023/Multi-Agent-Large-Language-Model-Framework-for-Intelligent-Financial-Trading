"""Walk-forward validation and calibrated training.

Walk-forward (expanding window, purged):
    dates:  |------------- train -------------|gap| test 1 |
            |------------------ train ------------------|gap| test 2 |  ...
* Test blocks are always AFTER their training data (no shuffling, no look-ahead).
* A gap of `horizon` trading days is removed between train and test, because the
  last training labels look `horizon` days ahead and would otherwise overlap the
  test period (this is called purging).

The same procedure scores three baselines so the report can show whether
LightGBM adds anything:
    class_prior         always predicts the historical class mix (= chance)
    logistic_regression linear model on the same features
    random_forest       the model the previous RIO version used
"""
import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import CLASS_NAMES, FEATURE_COLUMNS
from ml.model import DirectionModel
from ml.settings import TechnicalModelSettings

logger = logging.getLogger(__name__)
LABELS = list(range(len(CLASS_NAMES)))
DOWN, FLAT, UP = 0, 1, 2


# ----- splitting -------------------------------------------------------------
def split_by_date(dataset: pd.DataFrame, holdout_share: float, gap_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier rows (minus a purge gap) and the most recent `holdout_share` of dates."""
    dates = np.sort(dataset["date"].unique())
    holdout_start = int(len(dates) * (1 - holdout_share))
    train_dates = dates[: max(holdout_start - gap_days, 0)]
    holdout_dates = dates[holdout_start:]
    return dataset[dataset["date"].isin(train_dates)], dataset[dataset["date"].isin(holdout_dates)]


def walk_forward_folds(dataset: pd.DataFrame, settings: TechnicalModelSettings) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(train_dates, test_dates) pairs, expanding window, purged by `horizon_days`."""
    dates = np.sort(dataset["date"].unique())
    first_test = int(len(dates) * (1 - settings.validation_share))
    blocks = np.array_split(np.arange(first_test, len(dates)), settings.validation_folds)
    folds = []
    for block in blocks:
        if len(block) == 0:
            continue
        train_end = block[0] - settings.horizon_days
        if train_end <= 0:
            continue
        folds.append((dates[:train_end], dates[block]))
    return folds


# ----- training with calibration ------------------------------------------------
def fit_calibrated_model(train: pd.DataFrame, settings: TechnicalModelSettings, refit_on_all: bool) -> DirectionModel:
    """Fits LightGBM on older rows, calibrates its temperature on the most recent rows.

    With `refit_on_all`, the trees are then re-fitted on ALL rows (so the newest data
    is used for prediction) and the fitted temperature is kept.
    """
    fit_part, calibration_part = split_by_date(train, settings.calibration_share, settings.horizon_days)
    model = DirectionModel().fit(fit_part[FEATURE_COLUMNS], fit_part["label"])
    if len(calibration_part) > 0:
        model.calibrate(calibration_part[FEATURE_COLUMNS], calibration_part["label"])
    if refit_on_all:
        temperature = model.temperature
        model.fit(train[FEATURE_COLUMNS], train["label"])
        model.temperature = temperature
    return model


# ----- candidate models ---------------------------------------------------------
class _ClassPrior:
    """Baseline: probabilities = class frequencies in the training data."""

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "_ClassPrior":
        counts = np.bincount(labels.astype(int), minlength=len(LABELS)).astype(float)
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile(self.prior, (len(features), 1))


def _sklearn_model(estimator) -> Callable[[pd.DataFrame, TechnicalModelSettings], Any]:
    def fit(train: pd.DataFrame, settings: TechnicalModelSettings):
        return estimator().fit(train[FEATURE_COLUMNS], train["label"].astype(int))
    return fit


CANDIDATES: Dict[str, Callable[[pd.DataFrame, TechnicalModelSettings], Any]] = {
    "lightgbm": lambda train, settings: fit_calibrated_model(train, settings, refit_on_all=False),
    "class_prior": lambda train, settings: _ClassPrior().fit(train, train["label"]),
    "logistic_regression": _sklearn_model(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    ),
    "random_forest": _sklearn_model(
        lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=100, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )
    ),
}


# ----- metrics ------------------------------------------------------------------
def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, forward_returns: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(int)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    directional = predictions != FLAT
    if directional.any():
        called_up = predictions[directional] == UP
        went_up = forward_returns[directional] > 0
        directional_accuracy = float((called_up == went_up).mean())
    else:
        directional_accuracy = float("nan")

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro", labels=LABELS, zero_division=0)), 4),
        "log_loss": round(float(log_loss(labels, probabilities, labels=LABELS)), 4),
        "directional_accuracy": round(directional_accuracy, 4),
        "directional_call_share": round(float(directional.mean()), 4),
    }


# ----- walk-forward evaluation --------------------------------------------------
def walk_forward_evaluate(
    dataset: pd.DataFrame, settings: TechnicalModelSettings, include_random_forest: bool = False
) -> Dict[str, Any]:
    """Scores every candidate on the same purged walk-forward folds (pooled out-of-sample predictions)."""
    names = [n for n in CANDIDATES if include_random_forest or n != "random_forest"]
    folds = walk_forward_folds(dataset, settings)
    if not folds:
        raise ValueError("Not enough dates for walk-forward validation.")

    probabilities: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    fold_scores: Dict[str, List[float]] = {name: [] for name in names}
    test_frames = []

    for fold_number, (train_dates, test_dates) in enumerate(folds, start=1):
        train = dataset[dataset["date"].isin(train_dates)]
        test = dataset[dataset["date"].isin(test_dates)]
        test_frames.append(test)
        logger.info("Fold %d: train %d rows (to %s), test %d rows (%s .. %s)", fold_number, len(train),
                    pd.Timestamp(train_dates[-1]).date(), len(test),
                    pd.Timestamp(test_dates[0]).date(), pd.Timestamp(test_dates[-1]).date())
        for name in names:
            model = CANDIDATES[name](train, settings)
            fold_probabilities = model.predict_proba(test[FEATURE_COLUMNS])
            probabilities[name].append(fold_probabilities)
            fold_scores[name].append(
                float(balanced_accuracy_score(test["label"].astype(int), fold_probabilities.argmax(axis=1)))
            )

    test_all = pd.concat(test_frames)
    labels, forward_returns = test_all["label"].to_numpy(), test_all["forward_return"].to_numpy()
    results = {
        name: {
            **classification_metrics(labels, np.vstack(probabilities[name]), forward_returns),
            "fold_balanced_accuracy": [round(score, 4) for score in fold_scores[name]],
        }
        for name in names
    }

    class_share = np.bincount(labels.astype(int), minlength=len(LABELS)) / len(labels)
    return {
        "method": f"purged walk-forward, {len(folds)} expanding folds, {settings.horizon_days}-day gap",
        "test_period": f"{pd.Timestamp(test_all['date'].min()).date()} to {pd.Timestamp(test_all['date'].max()).date()}",
        "test_rows": int(len(test_all)),
        "test_class_share": {CLASS_NAMES[i]: round(float(class_share[i]), 4) for i in LABELS},
        "models": results,
    }


def reliability_of(validation: Dict[str, Any], settings: TechnicalModelSettings) -> Tuple[str, float]:
    """'useful' / 'weak' / 'no_edge' from the LightGBM walk-forward scores."""
    lightgbm_scores = validation["models"]["lightgbm"]
    prior_scores = validation["models"]["class_prior"]
    edge = lightgbm_scores["balanced_accuracy"] - 1 / 3
    beats_prior_log_loss = lightgbm_scores["log_loss"] < prior_scores["log_loss"]
    if edge >= 2 * settings.min_edge and beats_prior_log_loss:
        return "useful", edge
    if edge >= settings.min_edge:
        return "weak", edge
    return "no_edge", edge

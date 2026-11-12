"""The direction model: LightGBM classifier + temperature calibration + SHAP explanations.

Why LightGBM: on tabular indicator features, gradient-boosted trees are the strongest
standard choice, train in seconds, handle features on different scales, and give
exact per-prediction explanations (TreeSHAP) for free via `pred_contrib=True`.

Why temperature scaling: boosted trees tend to be over-confident. One extra
parameter T (fitted on the most recent held-out data) softens the probabilities
so that "60% up" really means roughly 60%.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from ml.features import CLASS_NAMES, FEATURE_COLUMNS, FEATURE_DESCRIPTIONS

N_CLASSES = len(CLASS_NAMES)

# Conservative settings: shallow trees, large leaves and row/column sampling,
# because daily stock returns are mostly noise and deep trees memorise it.
DEFAULT_PARAMS: Dict[str, Any] = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    # No class re-weighting: in walk-forward tests "balanced" weights gave the same balanced
    # accuracy but biased probabilities (log loss worse than the class-prior baseline).
    class_weight=None,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class DirectionModel:
    """3-class (down / flat / up) LightGBM model with calibrated probabilities and SHAP reasons."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.estimator: Optional[LGBMClassifier] = None
        self.temperature: float = 1.0
        self.card: Dict[str, Any] = {}  # metadata: training period, validation scores ...

    # ----- training ---------------------------------------------------------
    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "DirectionModel":
        labels = labels.astype(int)
        missing = set(range(N_CLASSES)) - set(labels.unique())
        if missing:
            raise ValueError(f"Training data has no rows for classes {[CLASS_NAMES[i] for i in missing]}.")
        self.estimator = LGBMClassifier(**self.params).fit(features[FEATURE_COLUMNS], labels)
        return self

    def calibrate(self, features: pd.DataFrame, labels: pd.Series) -> float:
        """Fits the temperature T on held-out data by minimising log loss. Returns T."""
        raw_scores = self._raw_scores(features)
        labels = labels.astype(int).to_numpy()

        def loss(temperature: float) -> float:
            return log_loss(labels, softmax(raw_scores / temperature), labels=list(range(N_CLASSES)))

        result = minimize_scalar(loss, bounds=(0.25, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self.temperature

    # ----- prediction -------------------------------------------------------
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities, columns in CLASS_NAMES order (down, flat, up)."""
        return softmax(self._raw_scores(features) / self.temperature)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    def explain(self, features_row: pd.DataFrame, class_index: int, top_n: int) -> List[Dict[str, Any]]:
        """Top SHAP contributions toward `class_index` for a single row, largest absolute effect first."""
        self._check_fitted()
        row = features_row[FEATURE_COLUMNS].iloc[[0]]
        contributions = np.asarray(self.estimator.predict(row, pred_contrib=True))
        # Multiclass layout: [class0 features..., class0 bias, class1 features..., class1 bias, ...]
        per_class = contributions.reshape(N_CLASSES, len(FEATURE_COLUMNS) + 1)[class_index, :-1]

        order = np.argsort(-np.abs(per_class))[:top_n]
        return [
            {
                "feature": FEATURE_COLUMNS[i],
                "description": FEATURE_DESCRIPTIONS[FEATURE_COLUMNS[i]],
                "value": round(float(row.iloc[0, i]), 4),
                "contribution": round(float(per_class[i]), 4),
                "effect": "supports" if per_class[i] >= 0 else "opposes",
            }
            for i in order
        ]

    def feature_importance(self) -> Dict[str, int]:
        """Split-count importance, highest first (for the training report)."""
        self._check_fitted()
        pairs = sorted(zip(FEATURE_COLUMNS, self.estimator.feature_importances_), key=lambda p: -p[1])
        return {name: int(value) for name, value in pairs}

    # ----- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.card.setdefault("saved_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.card.setdefault("lightgbm_version", lightgbm.__version__)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "DirectionModel":
        model = joblib.load(path)
        if not isinstance(model, DirectionModel):
            raise TypeError(f"{path} does not contain a DirectionModel.")
        return model

    # ----- helpers ----------------------------------------------------------
    def _raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.estimator.predict(features[FEATURE_COLUMNS], raw_score=True))

    def _check_fitted(self) -> None:
        if self.estimator is None:
            raise RuntimeError("DirectionModel is not trained yet.")

"""Turns a model prediction into the structured `MLDirectionEvidence` the technical analyst reads."""
from typing import Optional

import numpy as np
import pandas as pd

from ml.features import CLASS_NAMES, build_feature_frame, flat_band
from ml.registry import ModelRegistry, ModelUnavailable
from ml.schemas import ClassProbabilities, FeatureDriver, MLDirectionEvidence, ModelQuality
from ml.settings import TechnicalModelSettings


class MLDirectionService:
    def __init__(self, registry: ModelRegistry, settings: TechnicalModelSettings) -> None:
        self.registry = registry
        self.settings = settings

    def evidence(self, prices: pd.DataFrame, benchmark_close: Optional[pd.Series], as_of_date: str) -> MLDirectionEvidence:
        """Prediction for the last row of `prices`. Raises ModelUnavailable (with a reason) when it cannot be made."""
        features = build_feature_frame(prices, benchmark_close)
        latest = features.iloc[[-1]]
        missing = latest.columns[latest.isna().iloc[0]].tolist()
        if missing:
            raise ModelUnavailable(
                f"not enough history to compute {', '.join(missing[:4])} (needs about 200 trading days "
                "of the stock and the benchmark)."
            )

        model = self.registry.get_model(as_of_date)
        probabilities = model.predict_proba(latest)[0]
        predicted = int(np.argmax(probabilities))
        drivers = model.explain(latest, predicted, self.settings.top_drivers)
        band = flat_band(prices["Close"], self.settings.horizon_days, self.settings.flat_threshold_vol_multiple).iloc[-1]

        card = model.card
        return MLDirectionEvidence(
            model=card.get("model", "LightGBM"),
            horizon_days=card.get("horizon_days", self.settings.horizon_days),
            predicted_class=CLASS_NAMES[predicted],
            probabilities=ClassProbabilities(**{name: round(float(p), 3) for name, p in zip(CLASS_NAMES, probabilities)}),
            flat_band_pct=round(float(band), 4),
            top_drivers=[FeatureDriver(**driver) for driver in drivers],
            quality=self._quality(card),
            trained_until=card["trained_until"],
            training_rows=card["training_rows"],
            training_stocks=card["training_stocks"],
        )

    @staticmethod
    def _quality(card: dict) -> ModelQuality:
        validation = card["validation"]
        scores = validation["models"]["lightgbm"]
        logistic = validation["models"].get("logistic_regression", {})
        directional = scores.get("directional_accuracy")
        return ModelQuality(
            validation_method=validation["method"],
            test_period=validation["test_period"],
            test_rows=validation["test_rows"],
            balanced_accuracy=scores["balanced_accuracy"],
            accuracy=scores["accuracy"],
            macro_f1=scores["macro_f1"],
            log_loss=scores["log_loss"],
            class_prior_log_loss=validation["models"]["class_prior"]["log_loss"],
            directional_accuracy=None if directional is None or np.isnan(directional) else directional,
            logistic_regression_balanced_accuracy=logistic.get("balanced_accuracy"),
            edge_over_chance=card["edge_over_chance"],
            reliability=card["reliability"],
        )

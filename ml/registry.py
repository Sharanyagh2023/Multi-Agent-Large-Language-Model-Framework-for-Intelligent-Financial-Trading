"""Finds the right saved model for a date, or trains one.

A model trained with cutoff C has only seen prices up to C. For an analysis on
date D it is therefore allowed only if C <= D (no future knowledge), and it is
re-used while D - C <= max_model_age_days. Otherwise a new model is trained with
cutoff D and saved. This makes live analysis and walk-forward backtests share the
same point-in-time-safe code path.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.market_data import MarketDataService
from ml.dataset import TrainingDataBuilder
from ml.model import DirectionModel
from ml.settings import TechnicalModelSettings
from ml.validation import fit_calibrated_model, reliability_of, walk_forward_evaluate

logger = logging.getLogger(__name__)
MODEL_PREFIX = "direction_lgbm"


class ModelUnavailable(RuntimeError):
    """No usable model / prediction (not enough data, training failed ...). The message says why."""


class ModelRegistry:
    def __init__(self, market_data: MarketDataService, settings: TechnicalModelSettings,
                 data_builder: Optional[TrainingDataBuilder] = None) -> None:
        self.market_data = market_data
        self.settings = settings
        self.data_builder = data_builder or TrainingDataBuilder(market_data, settings)
        self._loaded: Dict[Path, DirectionModel] = {}

    # ----- lookup -----------------------------------------------------------
    def get_model(self, as_of_date: str) -> DirectionModel:
        usable = [(cutoff, path) for cutoff, path in self.saved_models() if cutoff <= as_of_date]
        newest = max(usable) if usable else None

        if newest is not None and self._age_days(newest[0], as_of_date) <= self.settings.max_model_age_days:
            return self._load(newest[1])

        logger.info("No fresh model for %s; training one (first run downloads the universe, ~1 min).", as_of_date)
        try:
            return self.train(as_of_date)
        except Exception as error:
            if newest is not None:
                logger.warning("Retraining failed (%s); using the older model trained until %s.", error, newest[0])
                return self._load(newest[1])
            raise ModelUnavailable(f"model could not be trained: {error}") from error

    def saved_models(self) -> List[Tuple[str, Path]]:
        """(cutoff, path) of saved models whose settings match the current ones."""
        found = []
        for card_path in sorted(self.settings.models_dir.glob(f"{MODEL_PREFIX}_*.json")):
            model_path = card_path.with_suffix(".joblib")
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if model_path.exists() and self._matches_settings(card):
                found.append((card["trained_until"], model_path))
        return found

    # ----- training ---------------------------------------------------------
    def train(self, cutoff: str, include_random_forest: bool = False) -> DirectionModel:
        """Builds the dataset up to `cutoff`, validates walk-forward, fits the final model and saves it."""
        dataset = self.data_builder.build(cutoff)
        if len(dataset) < self.settings.min_training_rows:
            raise ModelUnavailable(
                f"only {len(dataset)} training rows up to {cutoff} (need {self.settings.min_training_rows})."
            )

        validation = walk_forward_evaluate(dataset, self.settings, include_random_forest)
        reliability, edge = reliability_of(validation, self.settings)
        model = fit_calibrated_model(dataset, self.settings, refit_on_all=True)
        model.card = {
            "model": "LightGBM gradient-boosted trees, 3 classes (down/flat/up), temperature-calibrated",
            "trained_until": cutoff,
            "training_start": str(dataset["date"].min().date()),
            "last_labelled_date": str(dataset["date"].max().date()),
            "training_rows": int(len(dataset)),
            "training_stocks": int(dataset["ticker"].nunique()),
            "benchmark": self.market_data.benchmark_index,
            "horizon_days": self.settings.horizon_days,
            "flat_threshold_vol_multiple": self.settings.flat_threshold_vol_multiple,
            "temperature": round(model.temperature, 4),
            "validation": validation,
            "reliability": reliability,
            "edge_over_chance": round(edge, 4),
            "feature_importance": model.feature_importance(),
        }

        model_path = self._model_path(cutoff)
        model.save(model_path)
        model_path.with_suffix(".json").write_text(json.dumps(model.card, indent=2, default=str), encoding="utf-8")
        self._loaded[model_path] = model
        logger.info("Saved %s (reliability: %s, balanced accuracy %.3f)", model_path.name, reliability,
                    validation["models"]["lightgbm"]["balanced_accuracy"])
        return model

    # ----- helpers ----------------------------------------------------------
    def _model_path(self, cutoff: str) -> Path:
        return self.settings.models_dir / f"{MODEL_PREFIX}_h{self.settings.horizon_days}_{cutoff}.joblib"

    def _load(self, path: Path) -> DirectionModel:
        if path not in self._loaded:
            self._loaded[path] = DirectionModel.load(path)
        return self._loaded[path]

    def _matches_settings(self, card: Dict[str, Any]) -> bool:
        return (
            card.get("horizon_days") == self.settings.horizon_days
            and card.get("flat_threshold_vol_multiple") == self.settings.flat_threshold_vol_multiple
            and card.get("benchmark") == self.market_data.benchmark_index
        )

    @staticmethod
    def _age_days(cutoff: str, as_of_date: str) -> int:
        return (datetime.strptime(as_of_date, "%Y-%m-%d") - datetime.strptime(cutoff, "%Y-%m-%d")).days

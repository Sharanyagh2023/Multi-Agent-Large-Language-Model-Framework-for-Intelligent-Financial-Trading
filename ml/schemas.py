"""Structured output of the ML direction model, as handed to the technical analyst."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

DirectionClass = Literal["down", "flat", "up"]
Reliability = Literal["useful", "weak", "no_edge"]


class ClassProbabilities(BaseModel):
    down: float
    flat: float
    up: float


class FeatureDriver(BaseModel):
    """One SHAP reason: how much a feature pushed today's prediction toward the predicted class."""

    feature: str
    description: str
    value: float = Field(description="Today's value of the feature.")
    contribution: float = Field(description="SHAP value in log-odds; positive supports the predicted class.")
    effect: Literal["supports", "opposes"]


class ModelQuality(BaseModel):
    """How well the model did on data it had never seen (walk-forward validation)."""

    validation_method: str
    test_period: str
    test_rows: int
    balanced_accuracy: float = Field(description="Mean recall over down/flat/up. Chance = 0.333.")
    chance_balanced_accuracy: float = 1 / 3
    accuracy: float
    macro_f1: float
    log_loss: float
    class_prior_log_loss: float = Field(description="Log loss of always predicting the historical class mix.")
    directional_accuracy: Optional[float] = Field(
        None, description="When the model called up or down, how often the stock really moved that way."
    )
    logistic_regression_balanced_accuracy: Optional[float] = None
    edge_over_chance: float
    reliability: Reliability


class MLDirectionEvidence(BaseModel):
    """The model's view for one stock on one date, with its quality and reasons."""

    model: str
    horizon_days: int
    predicted_class: DirectionClass
    probabilities: ClassProbabilities
    flat_band_pct: float = Field(description="Moves smaller than +-this (as a fraction) count as 'flat' for this stock now.")
    top_drivers: List[FeatureDriver]
    quality: ModelQuality
    trained_until: str
    training_rows: int
    training_stocks: int

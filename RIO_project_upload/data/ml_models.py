"""Machine-learning helpers built on scikit-learn and VADER.

* DirectionPredictor : random forest that estimates the chance a stock is higher in 5 trading days.
* NewsSentimentScorer: VADER, a ready-made sentiment model tuned for short social/news text.

Their outputs are *evidence* handed to the LLM agents, not decisions. The
validation accuracy is reported alongside every prediction so that the agent
can judge how much to trust it (it is often close to a coin flip, and that is
useful information).
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.indicators import TechnicalIndicatorCalculator

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_20d", "rsi_14", "macd_histogram",
    "bollinger_pct_b", "atr_pct", "volatility_20d", "volume_ratio",
]


class DirectionPredictor:
    """Trains on the stock's own past and predicts whether it rises over the next `horizon_days`."""

    MIN_TRAINING_ROWS = 150

    def __init__(self, horizon_days: int = 5, validation_share: float = 0.2) -> None:
        self.horizon_days = horizon_days
        self.validation_share = validation_share
        self.indicator_calculator = TechnicalIndicatorCalculator()

    def predict(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Returns probability_up plus honest quality numbers, or {} if there is too little data."""
        features = self.indicator_calculator.build_indicator_frame(prices)[FEATURE_COLUMNS]
        future_return = prices["Close"].shift(-self.horizon_days) / prices["Close"] - 1.0

        latest_features = features.iloc[[-1]]
        if latest_features.isna().any(axis=None):
            return {}

        # Only rows whose outcome is already known can be used for training.
        labelled_rows = features.dropna().join(future_return.rename("future_return")).dropna()
        if len(labelled_rows) < self.MIN_TRAINING_ROWS:
            return {}

        x_all = labelled_rows[FEATURE_COLUMNS]
        y_all = (labelled_rows["future_return"] > 0).astype(int)

        # Time-ordered split: validate on the most recent rows, never on shuffled data.
        split_index = int(len(labelled_rows) * (1 - self.validation_share))
        validation_model = self._new_model().fit(x_all.iloc[:split_index], y_all.iloc[:split_index])
        validation_accuracy = float(validation_model.score(x_all.iloc[split_index:], y_all.iloc[split_index:]))
        always_up_accuracy = float(y_all.iloc[split_index:].mean())

        final_model = self._new_model().fit(x_all, y_all)
        probability_up = float(final_model.predict_proba(latest_features)[0][1])

        return {
            "horizon_days": self.horizon_days,
            "probability_up": round(probability_up, 3),
            "validation_accuracy": round(validation_accuracy, 3),
            "always_predict_up_accuracy": round(always_up_accuracy, 3),
            "training_rows": len(labelled_rows),
        }

    @staticmethod
    def _new_model() -> RandomForestClassifier:
        # Shallow trees and large leaves keep the model from memorising noise.
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=42)


class NewsSentimentScorer:
    """Scores headlines from -1 (very negative) to +1 (very positive) with VADER."""

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()

    def score_articles(self, articles: List[Dict[str, str]]) -> Dict[str, object]:
        if not articles:
            return {}

        scored_articles = []
        for article in articles:
            text = f"{article['title']}. {article['summary']}".strip()
            compound_score = self.analyzer.polarity_scores(text)["compound"]
            scored_articles.append({"title": article["title"], "date": article["date"], "score": compound_score})

        scores = np.array([item["score"] for item in scored_articles])
        return {
            "average_score": round(float(scores.mean()), 3),
            "positive_articles": int((scores > 0.05).sum()),
            "negative_articles": int((scores < -0.05).sum()),
            "neutral_articles": int(((scores >= -0.05) & (scores <= 0.05)).sum()),
            "articles": scored_articles,
        }


"""
Financial sentiment analysis for RIO.

Features:
1. FinBERT financial sentiment classification.
2. VADER fallback when FinBERT is unavailable.
3. Normalized sentiment scores between -1 and +1.
4. Exponential recency-weighted sentiment aggregation.
5. Article and source counting.
6. Evidence-based confidence calculation.
7. Future-dated article filtering for point-in-time safety.
8. Inference time measurement.

This module produces sentiment evidence. It does not make trading decisions.
"""

from datetime import date, datetime
from math import exp
from time import perf_counter
from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class FinancialSentimentModel:
    """
    Financial sentiment classifier using FinBERT with VADER fallback.

    FinBERT is preferred because it is trained on financial language.
    VADER provides a lightweight fallback when FinBERT cannot load.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, prefer_finbert: bool = True) -> None:
        self.backend = "vader"
        self.pipeline = None
        self.vader = SentimentIntensityAnalyzer()
        self.load_error: Optional[str] = None

        if prefer_finbert:
            self._load_finbert()

    def _load_finbert(self) -> None:
        """Load FinBERT lazily to avoid requiring it for VADER usage."""

        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=-1,
            )

            self.backend = "finbert"

        except Exception as exc:
            self.load_error = str(exc)
            self.pipeline = None
            self.backend = "vader"

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize sentiment labels into three standard categories."""

        label = label.lower()

        if "positive" in label:
            return "positive"

        if "negative" in label:
            return "negative"

        return "neutral"

    def _score_vader(self, text: str):
        """Calculate sentiment using VADER."""

        compound = float(
            self.vader.polarity_scores(text)["compound"]
        )

        if compound >= 0.05:
            label = "positive"

        elif compound <= -0.05:
            label = "negative"

        else:
            label = "neutral"

        return label, compound

    def score(self, text: str) -> Dict[str, Any]:
        """
        Classify text and return normalized sentiment evidence.

        Score range:
            -1.0 = Very negative
             0.0 = Neutral
            +1.0 = Very positive
        """

        text = (text or "").strip()

        if not text:
            return {
                "label": "neutral",
                "score": 0.0,
                "model_confidence": None,
                "backend": self.backend,
                "inference_ms": 0.0,
            }

        start = perf_counter()

        if self.backend == "finbert" and self.pipeline is not None:

            try:
                result = self.pipeline(
                    text,
                    truncation=True,
                    max_length=512,
                )[0]

                label = self._normalize_label(result["label"])
                probability = float(result["score"])

                if label == "positive":
                    sentiment_score = probability

                elif label == "negative":
                    sentiment_score = -probability

                else:
                    sentiment_score = 0.0

                backend = "finbert"
                model_confidence = probability

            except Exception:
                # Keep the analyst operational if FinBERT inference fails.
                label, sentiment_score = self._score_vader(text)
                backend = "vader"
                model_confidence = None

        else:
            label, sentiment_score = self._score_vader(text)
            backend = "vader"
            model_confidence = None

        inference_ms = (perf_counter() - start) * 1000

        return {
            "label": label,
            "score": round(sentiment_score, 4),
            "model_confidence": (
                round(model_confidence, 4)
                if model_confidence is not None
                else None
            ),
            "backend": backend,
            "inference_ms": round(inference_ms, 2),
        }


def aggregate_sentiment(
    articles: List[Dict[str, Any]],
    model: FinancialSentimentModel,
    as_of_date: str,
    half_life_days: float = 5.0,
) -> Dict[str, Any]:
    """
    Aggregate financial article sentiment using recency weighting.

    Newer articles receive greater weight than older articles.

    Weight formula:
        weight = exp(-0.693147 * article_age / half_life_days)

    Confidence:
        Evidence coverage based on the number of valid articles.
        This is not model accuracy or a calibrated probability.

    Future-dated articles are excluded to prevent look-ahead bias.
    """

    if half_life_days <= 0:
        raise ValueError("half_life_days must be greater than zero")

    if not articles:
        return {}

    reference_date = (
        as_of_date
        if isinstance(as_of_date, date)
        else date.fromisoformat(str(as_of_date)[:10])
    )

    weighted_total = 0.0
    total_weight = 0.0
    total_inference_ms = 0.0

    scored_articles = []
    sources = set()

    counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
    }

    backend_counts = {
        "finbert": 0,
        "vader": 0,
    }

    for article in articles:

        title = article.get("title") or ""
        summary = article.get("summary") or ""

        text = f"{title}. {summary}".strip()

        if not text or text == ".":
            continue

        published = article.get("date")

        if not published:
            continue

        try:
            if isinstance(published, datetime):
                published_date = published.date()

            elif isinstance(published, date):
                published_date = published

            else:
                published_date = date.fromisoformat(
                    str(published)[:10]
                )

        except (ValueError, TypeError):
            continue

        age_days = (reference_date - published_date).days

        # Exclude future articles.
        if age_days < 0:
            continue

        # Run sentiment classification.
        result = model.score(text)

        label = result["label"]
        score = result["score"]

        # Exponential recency weighting.
        weight = exp(
            -0.693147 * age_days / half_life_days
        )

        weighted_total += score * weight
        total_weight += weight

        total_inference_ms += result["inference_ms"]

        counts[label] += 1
        backend_counts[result["backend"]] += 1

        source = article.get("source")

        if source:
            sources.add(str(source).strip().lower())

        scored_articles.append({
            "title": title,
            "date": published_date.isoformat(),
            "source": source,
            "label": label,
            "score": round(score, 4),
            "weight": round(weight, 4),
            "backend": result["backend"],
            "model_confidence": result["model_confidence"],
        })

    # Return no view if no valid articles remain.
    if total_weight == 0:
        return {}

    weighted_score = weighted_total / total_weight

    article_count = len(scored_articles)
    source_count = len(sources)

    # Evidence coverage, not model accuracy.
    article_confidence = min(
        1.0,
        article_count / 10.0,
    )

    return {
        "weighted_score": round(weighted_score, 3),

        # Retained as a convenient compatibility alias.
        "average_score": round(weighted_score, 3),

        "article_count": article_count,
        "source_count": source_count,

        "confidence": round(article_confidence, 3),
        "confidence_basis": (
            "Article coverage: min(valid article count / 10, 1.0). "
            "Not a calibrated probability."
        ),

        "positive_articles": counts["positive"],
        "negative_articles": counts["negative"],
        "neutral_articles": counts["neutral"],

        "model_usage": backend_counts,

        "average_inference_ms": round(
            total_inference_ms / article_count, 2
        ),

        "half_life_days": half_life_days,

        "articles": scored_articles,
    }

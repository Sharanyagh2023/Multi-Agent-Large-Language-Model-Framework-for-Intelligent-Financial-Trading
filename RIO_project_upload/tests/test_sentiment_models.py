import pytest

from data.sentiment_models import (
    FinancialSentimentModel,
    aggregate_sentiment,
)


class FakeSentimentModel:
    """Predictable model for testing aggregation without FinBERT."""

    def __init__(self):
        self.calls = []

    def score(self, text):
        self.calls.append(text)

        text_lower = text.lower()

        if "growth" in text_lower:
            label, score = "positive", 0.8
        elif "loss" in text_lower:
            label, score = "negative", -0.8
        else:
            label, score = "neutral", 0.0

        return {
            "label": label,
            "score": score,
            "model_confidence": 0.9,
            "backend": "vader",
            "inference_ms": 2.0,
        }


def test_vader_positive_sentiment():
    model = FinancialSentimentModel(prefer_finbert=False)
    result = model.score("The company reported excellent growth and profits.")

    assert result["label"] == "positive"
    assert result["score"] > 0


def test_vader_negative_sentiment():
    model = FinancialSentimentModel(prefer_finbert=False)
    result = model.score("The company suffered massive losses and poor performance.")

    assert result["label"] == "negative"
    assert result["score"] < 0


def test_empty_text_returns_neutral():
    model = FinancialSentimentModel(prefer_finbert=False)
    result = model.score("")

    assert result["label"] == "neutral"
    assert result["score"] == 0.0


def test_empty_articles_return_empty_result():
    model = FakeSentimentModel()

    result = aggregate_sentiment(
        articles=[],
        model=model,
        as_of_date="2026-11-12",
    )

    assert result == {}


def test_future_articles_are_excluded():
    model = FakeSentimentModel()

    articles = [
        {
            "title": "Company reports strong growth",
            "date": "2026-11-11",
            "source": "Source A",
        },
        {
            "title": "Company reports heavy loss",
            "date": "2026-11-13",
            "source": "Source B",
        },
    ]

    result = aggregate_sentiment(
        articles=articles,
        model=model,
        as_of_date="2026-11-12",
    )

    assert result["article_count"] == 1
    assert result["positive_articles"] == 1
    assert result["negative_articles"] == 0
    assert len(model.calls) == 1


def test_recency_weighting():
    model = FakeSentimentModel()

    articles = [
        {
            "title": "Company reports strong growth",
            "date": "2026-11-12",
            "source": "Source A",
        },
        {
            "title": "Company reports heavy loss",
            "date": "2026-11-07",
            "source": "Source B",
        },
    ]

    result = aggregate_sentiment(
        articles=articles,
        model=model,
        as_of_date="2026-11-12",
        half_life_days=5,
    )

    # Recent positive news should receive more weight
    # than the five-day-old negative news.
    assert result["weighted_score"] > 0


def test_article_count_and_confidence():
    model = FakeSentimentModel()

    articles = [
        {
            "title": f"Company reports growth number {i}",
            "date": "2026-11-12",
            "source": f"Source {i}",
        }
        for i in range(5)
    ]

    result = aggregate_sentiment(
        articles=articles,
        model=model,
        as_of_date="2026-11-12",
    )

    assert result["article_count"] == 5
    assert result["confidence"] == 0.5
    assert result["source_count"] == 5


def test_invalid_dates_are_skipped():
    model = FakeSentimentModel()

    articles = [
        {
            "title": "Company reports growth",
            "date": "invalid-date",
            "source": "Source A",
        },
        {
            "title": "Company reports growth",
            "date": "2026-11-12",
            "source": "Source B",
        },
    ]

    result = aggregate_sentiment(
        articles=articles,
        model=model,
        as_of_date="2026-11-12",
    )

    assert result["article_count"] == 1


def test_invalid_half_life_raises_error():
    model = FakeSentimentModel()

    with pytest.raises(ValueError):
        aggregate_sentiment(
            articles=[
                {
                    "title": "Company reports growth",
                    "date": "2026-11-12",
                }
            ],
            model=model,
            as_of_date="2026-11-12",
            half_life_days=0,
        )


def test_no_valid_articles_returns_empty_result():
    model = FakeSentimentModel()

    articles = [
        {
            "title": "Company reports growth",
            "date": "2026-11-13",
            "source": "Future Source",
        },
        {
            "title": "",
            "date": "2026-11-12",
            "source": "Empty Source",
        },
    ]

    result = aggregate_sentiment(
        articles=articles,
        model=model,
        as_of_date="2026-11-12",
    )

    assert result == {}

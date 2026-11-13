
from unittest.mock import Mock

from agents.sentiment_analyst import SentimentAnalyst


class FakeNewsService:
    def __init__(self, articles=None, should_fail=False):
        self.articles = articles or []
        self.should_fail = should_fail
        self.calls = []

    def get_articles(self, ticker, as_of, limit):
        self.calls.append({
            "ticker": ticker,
            "as_of": as_of,
            "limit": limit,
        })

        if self.should_fail:
            raise RuntimeError("Simulated news service failure")

        return self.articles


class FakeSentimentModel:
    def score(self, text):
        return {
            "label": "positive",
            "score": 0.8,
            "model_confidence": 0.9,
            "backend": "vader",
            "inference_ms": 1.0,
        }


class FakeMarketData:
    market = {
        "name": "Indian Market",
        "currency": "INR",
    }


def sample_articles():
    return [
        {
            "title": "Company reports strong growth",
            "summary": "Revenue and profits increased.",
            "date": "2026-11-12",
            "source": "Test Financial News",
            "url": "https://example.com/news",
        }
    ]


def test_collect_data_uses_news_service(monkeypatch):
    news = FakeNewsService(articles=sample_articles())

    agent = SentimentAnalyst(
        llm=None,
        market_data=FakeMarketData(),
        news_limit=5,
        news_service=news,
        sentiment_model=FakeSentimentModel(),
    )

    monkeypatch.setattr(
        "agents.sentiment_analyst.collect_market_mood",
        lambda **kwargs: {},
    )

    data = agent.collect_data("HDFCBANK", "2026-11-12")

    assert news.calls == [{
        "ticker": "HDFCBANK",
        "as_of": "2026-11-12",
        "limit": 5,
    }]

    assert "news_sentiment" in data
    assert data["news_sentiment"]["article_count"] == 1


def test_collect_data_handles_news_service_failure(monkeypatch):
    news = FakeNewsService(should_fail=True)

    agent = SentimentAnalyst(
        llm=None,
        market_data=FakeMarketData(),
        news_limit=5,
        news_service=news,
        sentiment_model=FakeSentimentModel(),
    )

    monkeypatch.setattr(
        "agents.sentiment_analyst.collect_market_mood",
        lambda **kwargs: {},
    )

    data = agent.collect_data("HDFCBANK", "2026-11-12")

    assert data == {}


def test_run_skips_llm_when_no_data(monkeypatch):
    news = FakeNewsService()

    agent = SentimentAnalyst(
        llm=None,
        market_data=FakeMarketData(),
        news_limit=5,
        news_service=news,
        sentiment_model=FakeSentimentModel(),
    )

    monkeypatch.setattr(
        "agents.sentiment_analyst.collect_market_mood",
        lambda **kwargs: {},
    )

    state = {
        "ticker": "HDFCBANK",
        "as_of_date": "2026-11-12",
    }

    result = agent.run(state)

    report = result["analyst_reports"][0]

    assert report.analyst == "sentiment_analyst"
    assert report.ticker == "HDFCBANK"
    assert report.signal == "neutral"
    assert report.confidence == 0.0
    assert "No data was available" in report.summary


def test_run_calls_llm_when_evidence_exists(monkeypatch):
    news = FakeNewsService(articles=sample_articles())

    fake_llm = Mock()

    agent = SentimentAnalyst(
        llm=fake_llm,
        market_data=FakeMarketData(),
        news_limit=5,
        news_service=news,
        sentiment_model=FakeSentimentModel(),
    )

    monkeypatch.setattr(
        "agents.sentiment_analyst.collect_market_mood",
        lambda **kwargs: {},
    )

    fake_response = Mock()
    fake_response.model_dump.return_value = {
        "signal": "bullish",
        "confidence": 0.8,
        "summary": "Positive company news suggests an optimistic outlook.",
        "key_points": ["Strong revenue growth"],
    }

    structured_llm = Mock()
    structured_llm.invoke.return_value = fake_response

    fake_llm.with_structured_output.return_value = structured_llm

    state = {
        "ticker": "HDFCBANK",
        "as_of_date": "2026-11-12",
    }

    result = agent.run(state)

    fake_llm.with_structured_output.assert_called_once()
    structured_llm.invoke.assert_called_once()

    report = result["analyst_reports"][0]

    assert report.analyst == "sentiment_analyst"
    assert report.ticker == "HDFCBANK"
    assert report.signal == "bullish"
    assert report.confidence == 0.8

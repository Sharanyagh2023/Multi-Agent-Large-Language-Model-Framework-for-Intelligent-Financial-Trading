"""Sentiment analyst: how do people and the market *feel* about the stock?

Evidence: VADER sentiment scores of recent articles, and the India VIX "fear index"
as a market-wide mood gauge.
"""
from typing import Any, Dict

from agents.base_agent import AnalystAgent
from data.ml_models import NewsSentimentScorer


class SentimentAnalyst(AnalystAgent):
    name = "sentiment_analyst"
    role_description = (
        "You measure the mood around the stock: tone of recent coverage and the market-wide fear level."
    )

    def __init__(self, llm, market_data, news_limit: int) -> None:
        super().__init__(llm, market_data)
        self.news_limit = news_limit
        self.sentiment_scorer = NewsSentimentScorer()

    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        articles = self.market_data.get_news(ticker, as_of_date, self.news_limit)
        news_sentiment = self.sentiment_scorer.score_articles(articles)
        if news_sentiment:
            data["news_sentiment"] = news_sentiment

        vix = self.market_data.get_index_series(self.market_data.fear_index, as_of_date, lookback_days=30)
        if len(vix) >= 6:
            data["india_vix"] = {
                "latest": round(float(vix.iloc[-1]), 2),
                "change_over_5_days": round(float(vix.iloc[-1] - vix.iloc[-6]), 2),
            }
        return data

    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        return (
            f"Stock: {ticker}\n"
            f"Date: {as_of_date}\n\n"
            f"Sentiment evidence:\n{self.to_json(data)}\n\n"
            "Guidance: article scores range from -1 (very negative) to +1 (very positive). "
            f"An India VIX above {self.market_data.market['fear_index_high']} signals fear; "
            f"below {self.market_data.market['fear_index_low']} signals complacency. "
            "Few articles means low confidence."
        )

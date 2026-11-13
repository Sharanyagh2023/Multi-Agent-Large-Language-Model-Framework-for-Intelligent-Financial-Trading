
"""
Sentiment analyst for Indian equities.

Evidence:
- Company news from WP4 NewsService
- Financial sentiment from FinBERT with VADER fallback
- Recency-weighted article sentiment
- Article/source coverage and confidence
- India VIX and available market-wide mood indicators

All evidence is collected before the LLM is asked to interpret it.
"""

import logging
from typing import Any, Dict, Optional

from agents.base_agent import AnalystAgent
from data.news import NewsService
from data.sentiment_models import (
    FinancialSentimentModel,
    aggregate_sentiment,
)
from data.market_mood import collect_market_mood

logger = logging.getLogger(__name__)


class SentimentAnalyst(AnalystAgent):
    name = "sentiment_analyst"

    role_description = (
        "You analyse company news sentiment and broader Indian market mood "
        "using financial sentiment models and market indicators."
    )

    def __init__(
        self,
        llm,
        market_data,
        news_limit: int,
        news_service: Optional[NewsService] = None,
        sentiment_model: Optional[FinancialSentimentModel] = None,
    ) -> None:
        super().__init__(llm, market_data)

        self.news_limit = news_limit

        # Reuse an injected service/model when supplied.
        self.news_service = news_service or NewsService()

        # FinBERT is preferred; FinancialSentimentModel handles
        # fallback to VADER if FinBERT is unavailable.
        self.sentiment_model = (
            sentiment_model
            if sentiment_model is not None
            else FinancialSentimentModel(prefer_finbert=True)
        )

    def collect_data(
        self,
        ticker: str,
        as_of_date: str,
    ) -> Dict[str, Any]:

        data: Dict[str, Any] = {}

        # -------------------------------------------------
        # 1. Collect company news from WP4
        # -------------------------------------------------
        articles = []

        try:
            articles = self.news_service.get_articles(
                ticker=ticker,
                as_of=as_of_date,
                limit=self.news_limit,
            )

        except Exception as error:
            logger.warning(
                "Could not retrieve news for %s: %s",
                ticker,
                error,
            )

        # -------------------------------------------------
        # 2. Score and aggregate company news
        # -------------------------------------------------
        if articles:
            try:
                news_sentiment = aggregate_sentiment(
                    articles=articles,
                    model=self.sentiment_model,
                    as_of_date=as_of_date,
                )

                if news_sentiment:
                    data["news_sentiment"] = news_sentiment

            except Exception as error:
                logger.warning(
                    "Could not calculate sentiment for %s: %s",
                    ticker,
                    error,
                )

        # -------------------------------------------------
        # 3. Collect market-wide mood
        # -------------------------------------------------
        try:
            market_mood = collect_market_mood(
                market_data=self.market_data,
                as_of=as_of_date,
                lookback_days=30,
            )

            if market_mood:
                data["market_mood"] = market_mood

        except Exception as error:
            logger.warning(
                "Could not collect market mood: %s",
                error,
            )

        # -------------------------------------------------
        # 4. No evidence means no analyst report
        # -------------------------------------------------
        if not data:
            logger.info(
                "No sentiment evidence available for %s on %s",
                ticker,
                as_of_date,
            )
            return {}

        return data

    def build_prompt(
        self,
        ticker: str,
        as_of_date: str,
        data: Dict[str, Any],
    ) -> str:

        if not data:
            return ""

        return (
            f"Stock: {ticker}\n"
            f"Date: {as_of_date}\n\n"

            "SENTIMENT AND MARKET MOOD EVIDENCE\n"
            "----------------------------------\n"
            f"{self.to_json(data)}\n\n"

            "ANALYSIS GUIDELINES\n"
            "1. Interpret article sentiment scores between -1 and +1.\n"
            "2. Consider recency-weighted sentiment rather than "
            "treating every article equally.\n"
            "3. Consider article count and source count when assessing "
            "the strength of the evidence.\n"
            "4. Evidence confidence represents article coverage, "
            "not a calibrated probability of correctness.\n"
            "5. Distinguish company-specific financial sentiment from "
            "general news such as bank holidays or market-wide events.\n"
            "6. Interpret India VIX as a market-wide volatility/fear "
            "indicator, not as direct evidence of company performance.\n"
            "7. Do not invent missing PCR, advance/decline or FII/DII data.\n"
            "8. Clearly distinguish facts from interpretation.\n"
            "9. If evidence is insufficient, state that confidence is low.\n"
        )

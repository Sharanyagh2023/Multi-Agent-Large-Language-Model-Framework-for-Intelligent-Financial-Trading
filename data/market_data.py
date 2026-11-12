"""Market data access (prices, fundamentals, news) using the yfinance library, for the
market selected in config.yaml (phase 1: Indian equities on NSE/BSE).

Every method takes an `as_of` date and never returns anything newer than it, so
the same code serves live analysis and historical backtests.

Backtest caveat: yfinance only provides *today's* fundamentals and the *latest*
news. In `point_in_time=True` mode those two sources are switched off (empty
results) rather than leaking future information into a historical decision.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


METADATA_FIELDS = [
    "longName",
    "sector",
    "industry",
]

COMMON_FUNDAMENTAL_FIELDS = [
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "dividendYield",
    "beta",
]

NON_BANK_FUNDAMENTAL_FIELDS = [
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "currentRatio",
    "freeCashflow",
]

BANK_FUNDAMENTAL_FIELDS = [
    "returnOnAssets",
]

FUNDAMENTAL_FIELDS = (
    METADATA_FIELDS
    + COMMON_FUNDAMENTAL_FIELDS
    + NON_BANK_FUNDAMENTAL_FIELDS
    + BANK_FUNDAMENTAL_FIELDS
)


class MarketDataService:
    """Single place where the project talks to the outside data provider."""

    def __init__(self, market: Dict[str, Any], point_in_time: bool = False) -> None:
        self.market = market  # the active entry of config.yaml -> markets
        self.point_in_time = point_in_time

    # ----- market settings --------------------------------------------------
    @property
    def benchmark_index(self) -> str:
        return self.market["benchmark_index"]

    @property
    def fear_index(self) -> str:
        return self.market["fear_index"]

    def normalize_ticker(self, symbol: str) -> str:
        """'reliance' -> 'RELIANCE.NS'. Rejects symbols from exchanges the market does not cover."""
        symbol = symbol.strip().upper()
        suffixes = self.market["exchange_suffixes"]
        if "." not in symbol:
            return symbol + self.market["default_suffix"]
        if not symbol.endswith(tuple(suffixes)):
            raise ValueError(f"'{symbol}' is not listed on {self.market['name']}. Allowed suffixes: {suffixes}")
        return symbol

    # ----- prices -----------------------------------------------------------
    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily OHLCV between two dates (both inclusive), with a plain date index."""
        end_exclusive = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        prices = yf.Ticker(ticker).history(start=start, end=end_exclusive, auto_adjust=True)
        if prices.empty:
            return prices
        prices.index = prices.index.tz_localize(None).normalize()
        return prices[["Open", "High", "Low", "Close", "Volume"]]

    def get_recent_prices(self, ticker: str, as_of: str, lookback_days: int) -> pd.DataFrame:
        start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return self.get_price_history(ticker, start, as_of)

    # ----- fundamentals -----------------------------------------------------
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Collect fundamental metrics according to the company's sector."""

        if self.point_in_time:
            return {}

        try:
            info = yf.Ticker(ticker).info
        except Exception as error:
            logger.warning(
                "Could not load fundamentals for %s: %s",
                ticker,
                error
            )
            return {}

        sector = info.get("sector")
        industry = info.get("industry")

        # Select metrics according to the company's sector.
        if sector == "Financial Services" and industry and "bank" in industry.lower():
            selected_fields = (
                METADATA_FIELDS
                + COMMON_FUNDAMENTAL_FIELDS
                + BANK_FUNDAMENTAL_FIELDS
            )
        else:
            selected_fields = (
                METADATA_FIELDS
                + COMMON_FUNDAMENTAL_FIELDS
                + NON_BANK_FUNDAMENTAL_FIELDS
            )

        return {
            field: info[field]
            for field in selected_fields
            if info.get(field) is not None
        }

    # ----- news -------------------------------------------------------------
    def get_news(self, ticker: str, as_of: str, limit: int) -> List[Dict[str, str]]:
        """Recent headlines as [{'date', 'title', 'summary'}], newest first (empty in point-in-time mode)."""
        if self.point_in_time:
            return []
        try:
            raw_items = yf.Ticker(ticker).news or []
        except Exception as error:
            logger.warning("Could not load news for %s: %s", ticker, error)
            return []

        as_of_date = datetime.strptime(as_of, "%Y-%m-%d").date()
        articles = [self._parse_news_item(item) for item in raw_items]
        articles = [a for a in articles if a is not None and a["date"] <= as_of_date.isoformat()]
        articles.sort(key=lambda article: article["date"], reverse=True)
        return articles[:limit]

    @staticmethod
    def _parse_news_item(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """yfinance has used two news formats; this reads both."""
        content = item.get("content", item)
        title = content.get("title")
        if not title:
            return None

        published = content.get("pubDate")
        if published:
            date_text = published[:10]
        elif item.get("providerPublishTime"):
            date_text = datetime.fromtimestamp(item["providerPublishTime"]).date().isoformat()
        else:
            return None
        return {"date": date_text, "title": title, "summary": content.get("summary") or ""}

    # ----- market context ---------------------------------------------------
    def get_index_series(self, symbol: str, as_of: str, lookback_days: int = 60) -> pd.Series:
        """Closing prices for a market index, e.g. the benchmark (^NSEI) or the fear index (^INDIAVIX)."""
        prices = self.get_recent_prices(symbol, as_of, lookback_days)
        return prices["Close"] if not prices.empty else pd.Series(dtype=float)

    def debug_fundamental_metadata(self, ticker: str):
        info = yf.Ticker(ticker).info

        print(f"\n--- {ticker} ---")

        fields = [
            "longName",
            "currency",
            "financialCurrency",
            "dividendYield",
            "marketCap",
            "freeCashflow",
            "sector",
            "industry",
        ]

        for field in fields:
            print(f"{field}: {info.get(field)}")
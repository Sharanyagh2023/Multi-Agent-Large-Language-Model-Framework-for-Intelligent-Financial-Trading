"""Builds the pooled training table (many stocks, many days) up to a cutoff date.

Point-in-time rule: prices are cut at the cutoff BEFORE features and labels are
computed, so a row's label can only use prices that existed on the cutoff date.
The last `horizon` days of every stock therefore have no label and are dropped.
"""
import logging
import pickle
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from data.market_data import MarketDataService
from ml.features import FEATURE_COLUMNS, build_feature_frame, build_labels
from ml.settings import TechnicalModelSettings

logger = logging.getLogger(__name__)

WARMUP_CALENDAR_DAYS = 450   # enough for 200-day averages and 52-week ranges to exist on the first training day
MIN_ROWS_PER_STOCK = 300


class PriceCache:
    """Disk cache for daily prices, so retraining (e.g. during a backtest) does not re-download ~50 stocks."""

    def __init__(self, market_data: MarketDataService, cache_dir: Path) -> None:
        self.market_data = market_data
        self.cache_dir = cache_dir

    def get(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily OHLCV for start..end (inclusive), served from disk when the cached range covers it."""
        path = self.cache_dir / f"{re.sub(r'[^A-Za-z0-9.]', '_', ticker)}.pkl"
        effective_end = min(end, date.today().isoformat())
        entry = self._read(path)

        if entry is None or entry["start"] > start or entry["covered_end"] < effective_end:
            download_start = min(start, entry["start"]) if entry else start
            download_end = max(effective_end, entry["covered_end"]) if entry else effective_end
            prices = self.market_data.get_price_history(ticker, download_start, download_end)
            if prices.empty:
                return prices
            entry = {"start": download_start, "covered_end": download_end, "prices": prices}
            self._write(path, entry)

        prices = entry["prices"]
        return prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]

    @staticmethod
    def _read(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as file:
                return pickle.load(file)
        except Exception as error:  # a corrupt cache file is simply rebuilt
            logger.warning("Ignoring unreadable price cache %s: %s", path, error)
            return None

    @staticmethod
    def _write(path: Path, entry: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(entry, file)


class TrainingDataBuilder:
    """Turns the universe's price histories into one feature/label table."""

    def __init__(self, market_data: MarketDataService, settings: TechnicalModelSettings,
                 cache: Optional[PriceCache] = None) -> None:
        self.market_data = market_data
        self.settings = settings
        self.cache = cache or PriceCache(market_data, settings.price_cache_dir)

    def build(self, cutoff: str) -> pd.DataFrame:
        """Columns: date, ticker, FEATURE_COLUMNS..., label, forward_return, flat_band. Sorted by date."""
        cutoff_ts = pd.Timestamp(cutoff)
        train_start = cutoff_ts - pd.DateOffset(years=self.settings.training_years)
        download_start = (train_start - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

        benchmark_prices = self.cache.get(self.market_data.benchmark_index, download_start, cutoff)
        if benchmark_prices.empty:
            raise RuntimeError(f"No benchmark ({self.market_data.benchmark_index}) prices up to {cutoff}.")
        benchmark_close = benchmark_prices["Close"]

        frames = []
        for ticker in self.settings.universe:
            try:
                prices = self.cache.get(ticker, download_start, cutoff)
            except Exception as error:  # one bad ticker should not stop training
                logger.warning("Skipping %s: %s", ticker, error)
                continue
            if len(prices) < MIN_ROWS_PER_STOCK:
                logger.warning("Skipping %s: only %d price rows", ticker, len(prices))
                continue

            frame = build_feature_frame(prices, benchmark_close).join(
                build_labels(prices["Close"], self.settings.horizon_days, self.settings.flat_threshold_vol_multiple)
            )
            frame = frame[frame.index >= train_start].dropna(subset=FEATURE_COLUMNS + ["label"])
            frame.insert(0, "ticker", ticker)
            frame.insert(0, "date", frame.index)
            frames.append(frame.reset_index(drop=True))

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", *FEATURE_COLUMNS, "label", "forward_return", "flat_band"])
        dataset = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        logger.info("Training table: %d rows, %d stocks, %s .. %s", len(dataset), dataset["ticker"].nunique(),
                    dataset["date"].min().date(), dataset["date"].max().date())
        return dataset

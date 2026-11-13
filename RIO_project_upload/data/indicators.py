"""Technical indicators computed with pandas (no extra TA library needed)."""
from typing import Dict

import numpy as np
import pandas as pd


class TechnicalIndicatorCalculator:
    """Turns a price table (Open/High/Low/Close/Volume) into indicator columns."""

    def build_indicator_frame(self, prices: pd.DataFrame) -> pd.DataFrame:
        """One row per day with every indicator. Early rows contain NaN until enough history exists."""
        close = prices["Close"]
        daily_returns = close.pct_change()

        indicators = pd.DataFrame(index=prices.index)
        indicators["close"] = close
        indicators["return_1d"] = daily_returns
        indicators["return_5d"] = close.pct_change(5)
        indicators["return_20d"] = close.pct_change(20)
        indicators["sma_20"] = close.rolling(20).mean()
        indicators["sma_50"] = close.rolling(50).mean()
        indicators["sma_200"] = close.rolling(200).mean()
        indicators["rsi_14"] = self._rsi(close, period=14)
        indicators["macd_histogram"] = self._macd_histogram(close)
        indicators["bollinger_pct_b"] = self._bollinger_percent_b(close, window=20)
        indicators["atr_pct"] = self._average_true_range(prices, period=14) / close
        indicators["volatility_20d"] = daily_returns.rolling(20).std() * np.sqrt(252)
        indicators["volume_ratio"] = prices["Volume"] / prices["Volume"].rolling(20).mean()
        return indicators

    def latest_summary(self, prices: pd.DataFrame) -> Dict[str, float]:
        """The most recent indicator values, rounded for readable prompts."""
        indicators = self.build_indicator_frame(prices)
        latest = indicators.iloc[-1]
        return {name: round(float(value), 4) for name, value in latest.items() if not np.isnan(value)}

    # ----- individual indicators -------------------------------------------
    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        """Relative Strength Index (Wilder smoothing). Above 70 = overbought, below 30 = oversold."""
        change = close.diff()
        average_gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        average_loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        relative_strength = average_gain / average_loss.replace(0, np.nan)
        return 100 - 100 / (1 + relative_strength)

    @staticmethod
    def _macd_histogram(close: pd.Series) -> pd.Series:
        """MACD line minus its signal line, divided by price so it is comparable across stocks."""
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return (macd_line - signal_line) / close

    @staticmethod
    def _bollinger_percent_b(close: pd.Series, window: int) -> pd.Series:
        """Where the price sits inside its Bollinger bands: 0 = lower band, 1 = upper band."""
        middle = close.rolling(window).mean()
        spread = 2 * close.rolling(window).std()
        return (close - (middle - spread)) / (2 * spread)

    @staticmethod
    def _average_true_range(prices: pd.DataFrame, period: int) -> pd.Series:
        previous_close = prices["Close"].shift(1)
        true_range = pd.concat(
            [
                prices["High"] - prices["Low"],
                (prices["High"] - previous_close).abs(),
                (prices["Low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(period).mean()

"""Technical indicators computed with pandas (no extra TA library needed).

Most indicators are scale-free (ratios, percentages, 0-100 oscillators), so a value
means the same thing for a Rs 100 stock and a Rs 10,000 stock. This matters twice:
the LLM can apply fixed rules of thumb, and the ML model can learn from many stocks
at once.

Every value on day t uses only prices up to and including day t (no look-ahead).
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


class TechnicalIndicatorCalculator:
    """Turns a price table (Open/High/Low/Close/Volume) into indicator columns."""

    def build_indicator_frame(self, prices: pd.DataFrame) -> pd.DataFrame:
        """One row per day with every indicator. Early rows contain NaN until enough history exists."""
        close, high, low, volume = prices["Close"], prices["High"], prices["Low"], prices["Volume"]
        daily_returns = close.pct_change(fill_method=None)

        indicators = pd.DataFrame(index=prices.index)
        indicators["close"] = close

        # ----- returns ------------------------------------------------------
        indicators["return_1d"] = daily_returns
        for days in (5, 10, 20, 60):
            indicators[f"return_{days}d"] = close.pct_change(days, fill_method=None)

        # ----- trend --------------------------------------------------------
        for window in (20, 50, 200):
            indicators[f"sma_{window}"] = close.rolling(window).mean()
            indicators[f"price_vs_sma_{window}"] = close / indicators[f"sma_{window}"] - 1.0
        indicators["sma_50_vs_sma_200"] = indicators["sma_50"] / indicators["sma_200"] - 1.0
        indicators["adx_14"], indicators["plus_di_14"], indicators["minus_di_14"] = self._adx(prices, period=14)

        # ----- momentum -----------------------------------------------------
        indicators["rsi_14"] = self._rsi(close, period=14)
        indicators["macd_line"], indicators["macd_histogram"] = self._macd(close)
        indicators["stochastic_k_14"] = self._stochastic_k(prices, period=14)

        # ----- volatility ---------------------------------------------------
        indicators["bollinger_pct_b"], indicators["bollinger_bandwidth"] = self._bollinger(close, window=20)
        indicators["atr_pct"] = self._average_true_range(prices, period=14) / close
        indicators["volatility_20d"] = daily_returns.rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        indicators["volatility_60d"] = daily_returns.rolling(60).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # ----- volume -------------------------------------------------------
        indicators["volume_ratio"] = volume / volume.rolling(20).mean()
        indicators["obv_trend_20"] = self._obv_trend(close, volume, window=20)

        # ----- price levels -------------------------------------------------
        indicators["support_20d"] = low.rolling(20).min()
        indicators["resistance_20d"] = high.rolling(20).max()
        indicators["high_52w"] = high.rolling(TRADING_DAYS_PER_YEAR, min_periods=126).max()
        indicators["low_52w"] = low.rolling(TRADING_DAYS_PER_YEAR, min_periods=126).min()
        indicators["pct_from_52w_high"] = close / indicators["high_52w"] - 1.0
        indicators["pct_from_52w_low"] = close / indicators["low_52w"] - 1.0

        return indicators.replace([np.inf, -np.inf], np.nan)

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
    def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """MACD line and histogram (line minus signal), both divided by price so they compare across stocks."""
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line / close, (macd_line - signal_line) / close

    @staticmethod
    def _macd_histogram(close: pd.Series) -> pd.Series:
        """Kept for backward compatibility with older callers."""
        return TechnicalIndicatorCalculator._macd(close)[1]

    @staticmethod
    def _stochastic_k(prices: pd.DataFrame, period: int) -> pd.Series:
        """Where the close sits in the last `period` days' range: 0 = at the low, 100 = at the high."""
        lowest = prices["Low"].rolling(period).min()
        highest = prices["High"].rolling(period).max()
        return 100 * (prices["Close"] - lowest) / (highest - lowest).replace(0, np.nan)

    @staticmethod
    def _bollinger(close: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """%B (0 = lower band, 1 = upper band) and bandwidth (band width / middle band)."""
        middle = close.rolling(window).mean()
        spread = 2 * close.rolling(window).std()
        percent_b = (close - (middle - spread)) / (2 * spread)
        bandwidth = (2 * spread) / middle
        return percent_b, bandwidth

    @staticmethod
    def _bollinger_percent_b(close: pd.Series, window: int) -> pd.Series:
        """Kept for backward compatibility with older callers."""
        return TechnicalIndicatorCalculator._bollinger(close, window)[0]

    @staticmethod
    def _true_range(prices: pd.DataFrame) -> pd.Series:
        previous_close = prices["Close"].shift(1)
        return pd.concat(
            [
                prices["High"] - prices["Low"],
                (prices["High"] - previous_close).abs(),
                (prices["Low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

    @classmethod
    def _average_true_range(cls, prices: pd.DataFrame, period: int) -> pd.Series:
        return cls._true_range(prices).rolling(period).mean()

    @classmethod
    def _adx(cls, prices: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index (trend strength, 0-100) with the +DI / -DI direction lines.

        ADX above 25 = strong trend, below 20 = weak / no trend. +DI above -DI = buyers in control.
        """
        up_move = prices["High"].diff()
        down_move = -prices["Low"].diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=prices.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=prices.index)

        smoothed_true_range = cls._true_range(prices).ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / smoothed_true_range
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / smoothed_true_range
        directional_index = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = directional_index.ewm(alpha=1 / period, adjust=False).mean()

        # The first `period` rows are not meaningful yet (smoothing has no history).
        warmup = prices.index[: period * 2]
        for series in (adx, plus_di, minus_di):
            series.loc[warmup] = np.nan
        return adx, plus_di, minus_di

    @staticmethod
    def _obv_trend(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
        """Change in On-Balance Volume over `window` days, divided by the volume traded in that time.

        Ranges from -1 (all volume on down days) to +1 (all volume on up days).
        """
        direction = np.sign(close.diff()).fillna(0.0)
        on_balance_volume = (direction * volume).cumsum()
        return on_balance_volume.diff(window) / volume.rolling(window).sum().replace(0, np.nan)

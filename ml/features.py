"""Model features and labels.

Features: scale-free technical indicators of the stock plus market context from
the benchmark index (Nifty 50). Every feature on day t uses only data up to day t.

Labels (3 classes) look `horizon` days ahead:
    up   : forward return >  +band
    down : forward return <  -band
    flat : otherwise
with band = flat_multiple * (20-day std of daily returns) * sqrt(horizon). The band
scales with each stock's own volatility, so a 1% move is "flat" for a volatile
stock but "up" for a quiet one.
"""
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from data.indicators import TechnicalIndicatorCalculator

CLASS_NAMES: Tuple[str, str, str] = ("down", "flat", "up")  # label 0, 1, 2

# Feature name -> plain-English description (used in SHAP explanations for the LLM).
FEATURE_DESCRIPTIONS = OrderedDict(
    [
        ("return_1d", "1-day return"),
        ("return_5d", "5-day return"),
        ("return_10d", "10-day return"),
        ("return_20d", "20-day return"),
        ("return_60d", "60-day return"),
        ("price_vs_sma_20", "price vs 20-day average"),
        ("price_vs_sma_50", "price vs 50-day average"),
        ("price_vs_sma_200", "price vs 200-day average"),
        ("sma_50_vs_sma_200", "50-day vs 200-day average (golden/death cross)"),
        ("adx_14", "ADX trend strength"),
        ("di_spread", "+DI minus -DI (buyer vs seller pressure)"),
        ("rsi_14", "RSI(14)"),
        ("macd_line", "MACD line (scaled by price)"),
        ("macd_histogram", "MACD histogram (scaled by price)"),
        ("stochastic_k_14", "stochastic %K(14)"),
        ("bollinger_pct_b", "Bollinger %B"),
        ("bollinger_bandwidth", "Bollinger bandwidth"),
        ("atr_pct", "ATR as % of price"),
        ("volatility_20d", "20-day volatility"),
        ("volatility_ratio_20_60", "20-day vs 60-day volatility"),
        ("log_volume_ratio", "volume vs 20-day average (log)"),
        ("obv_trend_20", "on-balance-volume trend"),
        ("pct_from_52w_high", "distance from 52-week high"),
        ("pct_from_52w_low", "distance from 52-week low"),
        ("benchmark_return_5d", "benchmark 5-day return"),
        ("benchmark_return_20d", "benchmark 20-day return"),
        ("benchmark_vs_sma_50", "benchmark vs its 50-day average"),
        ("excess_return_20d", "20-day return minus benchmark"),
        ("excess_return_60d", "60-day return minus benchmark"),
    ]
)
FEATURE_COLUMNS = list(FEATURE_DESCRIPTIONS)

_COPIED_INDICATORS = [
    "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
    "price_vs_sma_20", "price_vs_sma_50", "price_vs_sma_200", "sma_50_vs_sma_200", "adx_14",
    "rsi_14", "macd_line", "macd_histogram", "stochastic_k_14", "bollinger_pct_b", "bollinger_bandwidth",
    "atr_pct", "volatility_20d", "obv_trend_20", "pct_from_52w_high", "pct_from_52w_low",
]


def build_feature_frame(
    prices: pd.DataFrame,
    benchmark_close: Optional[pd.Series],
    calculator: Optional[TechnicalIndicatorCalculator] = None,
) -> pd.DataFrame:
    """One row per trading day with every column of FEATURE_COLUMNS (NaN until enough history)."""
    calculator = calculator or TechnicalIndicatorCalculator()
    indicators = calculator.build_indicator_frame(prices)

    features = indicators[_COPIED_INDICATORS].copy()
    features["di_spread"] = (indicators["plus_di_14"] - indicators["minus_di_14"]) / 100.0
    features["volatility_ratio_20_60"] = indicators["volatility_20d"] / indicators["volatility_60d"]
    features["log_volume_ratio"] = np.log(indicators["volume_ratio"].where(indicators["volume_ratio"] > 0))

    if benchmark_close is not None and not benchmark_close.empty:
        benchmark = benchmark_close.reindex(prices.index).ffill(limit=5)
        features["benchmark_return_5d"] = benchmark.pct_change(5, fill_method=None)
        features["benchmark_return_20d"] = benchmark.pct_change(20, fill_method=None)
        features["benchmark_vs_sma_50"] = benchmark / benchmark.rolling(50).mean() - 1.0
        features["excess_return_20d"] = features["return_20d"] - features["benchmark_return_20d"]
        features["excess_return_60d"] = features["return_60d"] - benchmark.pct_change(60, fill_method=None)
    else:
        for column in ("benchmark_return_5d", "benchmark_return_20d", "benchmark_vs_sma_50",
                       "excess_return_20d", "excess_return_60d"):
            features[column] = np.nan

    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def flat_band(close: pd.Series, horizon: int, flat_multiple: float) -> pd.Series:
    """Size of a 'normal' move over `horizon` days times `flat_multiple`, known on each day."""
    daily_volatility = close.pct_change(fill_method=None).rolling(20).std()
    return flat_multiple * daily_volatility * np.sqrt(horizon)


def build_labels(close: pd.Series, horizon: int, flat_multiple: float) -> pd.DataFrame:
    """Columns: label (0 down / 1 flat / 2 up, NaN when the future is unknown), forward_return, flat_band."""
    band = flat_band(close, horizon, flat_multiple)
    forward_return = close.shift(-horizon) / close - 1.0

    labels = pd.Series(1.0, index=close.index)
    labels[forward_return > band] = 2.0
    labels[forward_return < -band] = 0.0
    labels[forward_return.isna() | band.isna()] = np.nan

    return pd.DataFrame({"label": labels, "forward_return": forward_return, "flat_band": band})

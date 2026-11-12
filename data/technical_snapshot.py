"""Structured technical picture of one stock on one day.

The technical analyst does not hand the LLM a flat list of numbers. It hands it a
`TechnicalSnapshot`: indicators grouped by theme (trend, momentum, volatility,
volume, levels, relative strength), each with a label computed by fixed rules in
Python. The LLM then interprets labels that are already correct, instead of having
to apply thresholds itself (which small local models often get wrong).
"""
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from data.indicators import TRADING_DAYS_PER_YEAR, TechnicalIndicatorCalculator

# Thresholds used for the labels. They are standard textbook values.
ADX_STRONG, ADX_WEAK = 25.0, 20.0
RSI_OVERBOUGHT, RSI_OVERSOLD = 70.0, 30.0
VOLATILITY_HIGH_RATIO, VOLATILITY_LOW_RATIO = 1.3, 0.8   # 20-day volatility vs its 1-year median
VOLUME_HIGH, VOLUME_LOW = 1.5, 0.7                       # today's volume vs 20-day average
RELATIVE_STRENGTH_BAND = 0.02                            # +-2% excess return over 20 days counts as "in line"


class PriceLevels(BaseModel):
    close: float
    support_20d: Optional[float] = Field(None, description="Lowest low of the last 20 trading days.")
    resistance_20d: Optional[float] = Field(None, description="Highest high of the last 20 trading days.")
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    pct_from_52w_high: Optional[float] = Field(None, description="-0.10 = 10% below the 52-week high.")
    pct_from_52w_low: Optional[float] = Field(None, description="0.25 = 25% above the 52-week low.")
    position_in_52w_range: Optional[float] = Field(
        None, description="0 = at the 52-week low, 0.5 = middle of the range, 1 = at the 52-week high."
    )
    range_label: Literal[
        "near 52-week low", "lower part of 52-week range", "middle of 52-week range",
        "upper part of 52-week range", "near 52-week high", "unknown",
    ] = "unknown"


class TrendView(BaseModel):
    label: Literal["uptrend", "downtrend", "sideways"]
    strength: Literal["strong", "moderate", "weak", "unknown"]
    price_vs_sma_20: Optional[float] = Field(None, description="0.03 = price 3% above its 20-day average.")
    price_vs_sma_50: Optional[float] = None
    price_vs_sma_200: Optional[float] = None
    sma_50_above_sma_200: Optional[bool] = Field(None, description="True = 'golden cross' state, False = 'death cross'.")
    adx_14: Optional[float] = Field(None, description="Trend strength 0-100: >25 strong, <20 weak.")
    plus_di_above_minus_di: Optional[bool] = Field(None, description="True = buyers in control.")


class MomentumView(BaseModel):
    label: Literal["bullish", "bearish", "neutral"]
    rsi_14: Optional[float] = None
    rsi_zone: Literal["overbought", "oversold", "normal", "unknown"]
    macd_histogram: Optional[float] = Field(None, description="Positive = momentum rising (scaled by price).")
    stochastic_k_14: Optional[float] = Field(None, description="0 = at 14-day low, 100 = at 14-day high.")
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None


class VolatilityView(BaseModel):
    regime: Literal["high", "normal", "low", "unknown"]
    volatility_20d: Optional[float] = Field(None, description="Annualised standard deviation of daily returns.")
    volatility_vs_1y_median: Optional[float] = Field(None, description="1.5 = 50% more volatile than usual.")
    atr_pct: Optional[float] = Field(None, description="Average daily range as a share of price.")
    bollinger_pct_b: Optional[float] = Field(None, description="0 = lower band, 1 = upper band.")
    bollinger_bandwidth: Optional[float] = Field(None, description="Low values = squeeze (often precedes a big move).")


class VolumeView(BaseModel):
    label: Literal["high", "normal", "low", "unknown"]
    volume_ratio: Optional[float] = Field(None, description="Today's volume / 20-day average volume.")
    obv_trend_20: Optional[float] = Field(None, description="-1..1: share of 20-day volume on up days minus down days.")
    confirms_price_trend: Optional[bool] = Field(
        None, description="True when volume flow (OBV) points the same way as the 20-day price move."
    )


class RelativeStrengthView(BaseModel):
    label: Literal["outperforming", "underperforming", "in line"]
    benchmark: str
    stock_return_20d: Optional[float] = None
    benchmark_return_20d: Optional[float] = None
    excess_return_20d: Optional[float] = None
    excess_return_60d: Optional[float] = None


class TechnicalSnapshot(BaseModel):
    """Everything the technical analyst knows about the stock's price action on the as-of date."""

    last_price_date: str
    trading_days_used: int
    price_levels: PriceLevels
    trend: TrendView
    momentum: MomentumView
    volatility: VolatilityView
    volume: VolumeView
    relative_strength: Optional[RelativeStrengthView] = None


def _value(row: pd.Series, column: str, digits: int = 4) -> Optional[float]:
    """A rounded float, or None when the indicator is not available yet."""
    value = row.get(column, np.nan)
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


class TechnicalSnapshotBuilder:
    """Builds a `TechnicalSnapshot` from a price table (and optionally the benchmark index)."""

    def __init__(self, calculator: Optional[TechnicalIndicatorCalculator] = None) -> None:
        self.calculator = calculator or TechnicalIndicatorCalculator()

    def build(
        self,
        prices: pd.DataFrame,
        benchmark_close: Optional[pd.Series] = None,
        benchmark_name: str = "benchmark",
    ) -> TechnicalSnapshot:
        indicators = self.calculator.build_indicator_frame(prices)
        latest = indicators.iloc[-1]

        return TechnicalSnapshot(
            last_price_date=indicators.index[-1].strftime("%Y-%m-%d"),
            trading_days_used=len(indicators),
            price_levels=self._price_levels(latest),
            trend=self._trend(latest),
            momentum=self._momentum(latest),
            volatility=self._volatility(indicators, latest),
            volume=self._volume(latest),
            relative_strength=self._relative_strength(prices, benchmark_close, benchmark_name),
        )

    # ----- one method per theme --------------------------------------------
    @staticmethod
    def _price_levels(latest: pd.Series) -> PriceLevels:
        high, low = latest.get("high_52w", np.nan), latest.get("low_52w", np.nan)
        position, range_label = None, "unknown"
        if pd.notna(high) and pd.notna(low) and high > low:
            position = round(float((latest["close"] - low) / (high - low)), 3)
            if position < 0.1:
                range_label = "near 52-week low"
            elif position < 0.4:
                range_label = "lower part of 52-week range"
            elif position <= 0.6:
                range_label = "middle of 52-week range"
            elif position <= 0.9:
                range_label = "upper part of 52-week range"
            else:
                range_label = "near 52-week high"
        return PriceLevels(
            position_in_52w_range=position,
            range_label=range_label,
            close=round(float(latest["close"]), 2),
            support_20d=_value(latest, "support_20d", 2),
            resistance_20d=_value(latest, "resistance_20d", 2),
            high_52w=_value(latest, "high_52w", 2),
            low_52w=_value(latest, "low_52w", 2),
            pct_from_52w_high=_value(latest, "pct_from_52w_high"),
            pct_from_52w_low=_value(latest, "pct_from_52w_low"),
        )

    @staticmethod
    def _trend(latest: pd.Series) -> TrendView:
        close, sma_50, sma_200 = latest["close"], latest["sma_50"], latest["sma_200"]
        if pd.notna(sma_200):
            if close > sma_50 > sma_200:
                label = "uptrend"
            elif close < sma_50 < sma_200:
                label = "downtrend"
            else:
                label = "sideways"
        elif pd.notna(sma_50):  # less than 200 days of history: judge on the 50-day average only
            label = "uptrend" if close > sma_50 * 1.02 else "downtrend" if close < sma_50 * 0.98 else "sideways"
        else:
            label = "sideways"

        adx = latest.get("adx_14", np.nan)
        if pd.isna(adx):
            strength = "unknown"
        else:
            strength = "strong" if adx > ADX_STRONG else "weak" if adx < ADX_WEAK else "moderate"

        plus_di, minus_di = latest.get("plus_di_14", np.nan), latest.get("minus_di_14", np.nan)
        return TrendView(
            label=label,
            strength=strength,
            price_vs_sma_20=_value(latest, "price_vs_sma_20"),
            price_vs_sma_50=_value(latest, "price_vs_sma_50"),
            price_vs_sma_200=_value(latest, "price_vs_sma_200"),
            sma_50_above_sma_200=bool(sma_50 > sma_200) if pd.notna(sma_200) and pd.notna(sma_50) else None,
            adx_14=_value(latest, "adx_14", 1),
            plus_di_above_minus_di=bool(plus_di > minus_di) if pd.notna(plus_di) and pd.notna(minus_di) else None,
        )

    @staticmethod
    def _momentum(latest: pd.Series) -> MomentumView:
        rsi, macd = latest.get("rsi_14", np.nan), latest.get("macd_histogram", np.nan)
        if pd.isna(rsi):
            rsi_zone = "unknown"
        else:
            rsi_zone = "overbought" if rsi > RSI_OVERBOUGHT else "oversold" if rsi < RSI_OVERSOLD else "normal"

        # Momentum is bullish when RSI is above its midline AND MACD is rising (and the reverse).
        if pd.notna(rsi) and pd.notna(macd) and rsi > 50 and macd > 0:
            label = "bullish"
        elif pd.notna(rsi) and pd.notna(macd) and rsi < 50 and macd < 0:
            label = "bearish"
        else:
            label = "neutral"

        return MomentumView(
            label=label,
            rsi_14=_value(latest, "rsi_14", 1),
            rsi_zone=rsi_zone,
            macd_histogram=_value(latest, "macd_histogram", 5),
            stochastic_k_14=_value(latest, "stochastic_k_14", 1),
            return_5d=_value(latest, "return_5d"),
            return_20d=_value(latest, "return_20d"),
            return_60d=_value(latest, "return_60d"),
        )

    @staticmethod
    def _volatility(indicators: pd.DataFrame, latest: pd.Series) -> VolatilityView:
        history = indicators["volatility_20d"].tail(TRADING_DAYS_PER_YEAR).dropna()
        current = latest.get("volatility_20d", np.nan)
        ratio = None
        regime = "unknown"
        if pd.notna(current) and len(history) >= 60 and history.median() > 0:
            ratio = float(current / history.median())
            regime = "high" if ratio > VOLATILITY_HIGH_RATIO else "low" if ratio < VOLATILITY_LOW_RATIO else "normal"

        return VolatilityView(
            regime=regime,
            volatility_20d=_value(latest, "volatility_20d"),
            volatility_vs_1y_median=round(ratio, 3) if ratio is not None else None,
            atr_pct=_value(latest, "atr_pct"),
            bollinger_pct_b=_value(latest, "bollinger_pct_b", 3),
            bollinger_bandwidth=_value(latest, "bollinger_bandwidth"),
        )

    @staticmethod
    def _volume(latest: pd.Series) -> VolumeView:
        ratio = latest.get("volume_ratio", np.nan)
        if pd.isna(ratio):
            label = "unknown"
        else:
            label = "high" if ratio > VOLUME_HIGH else "low" if ratio < VOLUME_LOW else "normal"

        obv, return_20d = latest.get("obv_trend_20", np.nan), latest.get("return_20d", np.nan)
        confirms = None
        if pd.notna(obv) and pd.notna(return_20d) and return_20d != 0:
            confirms = bool(np.sign(obv) == np.sign(return_20d))

        return VolumeView(
            label=label,
            volume_ratio=_value(latest, "volume_ratio", 3),
            obv_trend_20=_value(latest, "obv_trend_20", 3),
            confirms_price_trend=confirms,
        )

    @staticmethod
    def _relative_strength(
        prices: pd.DataFrame, benchmark_close: Optional[pd.Series], benchmark_name: str
    ) -> Optional[RelativeStrengthView]:
        if benchmark_close is None or benchmark_close.empty:
            return None
        benchmark = benchmark_close.reindex(prices.index).ffill(limit=5)
        stock_close = prices["Close"]

        def period_return(series: pd.Series, days: int) -> Optional[float]:
            if len(series.dropna()) <= days:
                return None
            value = series.iloc[-1] / series.iloc[-1 - days] - 1.0
            return None if pd.isna(value) else float(value)

        stock_20, bench_20 = period_return(stock_close, 20), period_return(benchmark, 20)
        stock_60, bench_60 = period_return(stock_close, 60), period_return(benchmark, 60)
        if stock_20 is None or bench_20 is None:
            return None

        excess_20 = stock_20 - bench_20
        excess_60 = stock_60 - bench_60 if stock_60 is not None and bench_60 is not None else None
        if excess_20 > RELATIVE_STRENGTH_BAND:
            label = "outperforming"
        elif excess_20 < -RELATIVE_STRENGTH_BAND:
            label = "underperforming"
        else:
            label = "in line"

        return RelativeStrengthView(
            label=label,
            benchmark=benchmark_name,
            stock_return_20d=round(stock_20, 4),
            benchmark_return_20d=round(bench_20, 4),
            excess_return_20d=round(excess_20, 4),
            excess_return_60d=round(excess_60, 4) if excess_60 is not None else None,
        )

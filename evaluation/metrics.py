"""Financial performance and risk metrics.

All functions take a pandas Series of *periodic simple returns* (e.g. daily
returns of 0.01 = +1%). They are used twice: by the risk manager (to describe
how risky a stock has been) and by the backtester (to score the strategy).
"""
from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cumulative_return(returns: pd.Series) -> float:
    """Total growth over the whole period, e.g. 0.12 = +12%."""
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized return above the risk-free rate, per unit of volatility."""
    if len(returns) < 2:
        return 0.0
    daily_excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    daily_std = daily_excess_returns.std()
    if daily_std == 0 or np.isnan(daily_std):
        return 0.0
    return float(daily_excess_returns.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, returned as a negative number (e.g. -0.25)."""
    if returns.empty:
        return 0.0
    equity_curve = (1.0 + returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdowns = equity_curve / running_peak - 1.0
    return float(drawdowns.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical 1-day VaR as a positive loss: 'on 95% of days we lose less than this'."""
    if returns.empty:
        return 0.0
    return float(-np.percentile(returns, (1.0 - confidence) * 100.0))


def summarize_performance(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """The three headline metrics from the project brief, plus volatility."""
    return {
        "cumulative_return": cumulative_return(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "annualized_volatility": annualized_volatility(returns),
    }

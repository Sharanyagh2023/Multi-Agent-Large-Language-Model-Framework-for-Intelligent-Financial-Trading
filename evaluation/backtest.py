"""Walk-forward backtest of the trading workflow on one stock.

How it works
  * Every `rebalance_days` trading days the team makes a decision using only
    data up to that day (no look-ahead for prices).
  * BUY / SELL set the stock weight (long-only: SELL means weight 0); HOLD keeps
    the previous weight. The rest of the portfolio sits in cash earning 0.
  * The new weight applies from the NEXT trading day.
  * Results are compared with simply buying and holding the stock.

Limits to keep in mind: the LLM was trained on historical text, so it may "remember"
how the period ended; and fundamentals/news are switched off (see MarketDataService).
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from data.market_data import MarketDataService
from evaluation.metrics import summarize_performance
from workflow.trading_graph import TradingWorkflow

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    ticker: str
    decisions: pd.DataFrame          # one row per rebalance date
    strategy_metrics: Dict[str, float]
    buy_and_hold_metrics: Dict[str, float]
    strategy_returns: pd.Series      # daily returns, used for equity-curve charts
    buy_and_hold_returns: pd.Series


# Called after every rebalance decision: (row, index, total). Lets a UI show progress.
DecisionCallback = Callable[[Dict[str, Any], int, int], None]


class Backtester:
    def __init__(self, workflow: TradingWorkflow, market_data: MarketDataService, risk_free_rate: float) -> None:
        self.workflow = workflow
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        rebalance_days: int,
        on_decision: Optional[DecisionCallback] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """`run_config` is passed to every agent run (the web UI uses it to cancel a backtest)."""
        prices = self.market_data.get_price_history(ticker, start_date, end_date)
        if len(prices) < rebalance_days * 2:
            raise ValueError(f"Not enough price history for {ticker} between {start_date} and {end_date}.")

        daily_returns = prices["Close"].pct_change().fillna(0.0)
        decision_dates = list(prices.index[::rebalance_days])

        weight_by_date, decision_rows = self._collect_decisions(ticker, decision_dates, on_decision, run_config)

        # A decision made on day t is traded at the close of t and earns returns from t+1.
        weights = pd.Series(weight_by_date).reindex(prices.index).ffill().fillna(0.0).shift(1).fillna(0.0)
        strategy_returns = weights * daily_returns

        return BacktestResult(
            ticker=ticker,
            decisions=pd.DataFrame(decision_rows),
            strategy_metrics=summarize_performance(strategy_returns, self.risk_free_rate),
            buy_and_hold_metrics=summarize_performance(daily_returns, self.risk_free_rate),
            strategy_returns=strategy_returns,
            buy_and_hold_returns=daily_returns,
        )

    def _collect_decisions(
        self,
        ticker: str,
        decision_dates: List[pd.Timestamp],
        on_decision: Optional[DecisionCallback] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        current_weight = 0.0
        weight_by_date: Dict[pd.Timestamp, float] = {}
        decision_rows = []

        for index, decision_date in enumerate(decision_dates):
            date_text = decision_date.strftime("%Y-%m-%d")
            logger.info("Backtest decision for %s on %s", ticker, date_text)
            decision = self.workflow.decide(ticker, date_text, run_config)

            if decision.action == "BUY":
                current_weight = decision.position_size_pct
            elif decision.action == "SELL":
                current_weight = 0.0
            # HOLD keeps current_weight unchanged

            weight_by_date[decision_date] = current_weight
            row = {
                "date": date_text,
                "action": decision.action,
                "weight": current_weight,
                "risk_level": decision.risk_level,
            }
            decision_rows.append(row)
            if on_decision is not None:
                on_decision(row, index, len(decision_dates))
        return weight_by_date, decision_rows


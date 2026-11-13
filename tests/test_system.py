
"""System-level tests for RIO: input validation, schemas, risk guardrails, finalizer,
metrics, the LangGraph workflow order and the backtester. Offline (fake LLM and data).

Run from the project folder:  python -m pytest tests -q
"""
import sys
from pathlib import Path
from unittest import result

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import technical_analyst  # noqa: E402
from agents.risk_manager import RiskManager  # noqa: E402
from core.schemas import AnalystOpinion, FinalDecision, RiskAssessment, TradeProposal  # noqa: E402
from data.market_data import MarketDataService  # noqa: E402
from evaluation import metrics  # noqa: E402
from evaluation.backtest import Backtester  # noqa: E402
from ml.registry import ModelUnavailable  # noqa: E402
from workflow.trading_graph import TradingWorkflow  # noqa: E402

INDIA = {
    "name": "Indian equities (NSE/BSE)", "exchange_suffixes": [".NS", ".BO"], "default_suffix": ".NS",
    "benchmark_index": "^NSEI", "benchmark_name": "Nifty 50", "fear_index": "^INDIAVIX",
    "fear_index_high": 20, "fear_index_low": 13, "currency": "INR", "risk_free_rate": 0.07,
}


def make_prices(days: int = 600, end: str = "2025-06-30", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=days)
    close = 100 * np.exp(np.cumsum(0.0005 + 0.012 * rng.standard_normal(days)))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": np.full(days, 1e6)}, index=index)


class FakeMarketData:
    def __init__(self, news=None, fundamentals=None) -> None:
        self.market = INDIA
        self.benchmark_index, self.fear_index = INDIA["benchmark_index"], INDIA["fear_index"]
        self.news, self.fundamentals = news or [], fundamentals or {}

    def get_price_history(self, ticker, start, end):
        return make_prices().loc[start:end]

    def get_recent_prices(self, ticker, as_of, lookback_days):
        start = (pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return self.get_price_history(ticker, start, as_of)

    def get_index_series(self, symbol, as_of, lookback_days=60):
        prices = self.get_recent_prices(symbol, as_of, lookback_days)["Close"]
        return prices / prices.iloc[0] * 15 if symbol == "^INDIAVIX" else prices

    def get_fundamentals(self, ticker, as_of=None):
        return self.fundamentals

    def get_news(self, ticker, as_of, limit):
        return self.news[:limit]


class RecordingLLM:
    """Answers every structured request with a fixed object and records who asked, in order."""

    def __init__(self, trade=None, risk=None) -> None:
        self.calls = []
        self.answers = {
            "AnalystOpinion": AnalystOpinion(signal="bullish", confidence=0.6, summary="s", key_points=["k"]),
            "TradeProposal": trade or TradeProposal(action="BUY", position_size_pct=0.5, rationale="r"),
            "RiskAssessment": risk or RiskAssessment(decision="APPROVE", risk_level="medium",
                                                     adjusted_position_size_pct=0.5, comments="c"),
        }

    def with_structured_output(self, schema,**kwargs):
        llm = self

        class Runnable:
            def invoke(self, messages):
                llm.calls.append(schema.__name__)
                return llm.answers[schema.__name__]
        return Runnable()


class FakeConfig:
    trading = {"price_lookback_days": 730, "news_limit": 10, "max_position_pct": 0.20}
    market = INDIA

    def __init__(self, llm) -> None:
        self.llm = llm

    def get_llm_for_agent(self, name):
        return self.llm


@pytest.fixture
def no_ml_model(monkeypatch):
    """Keeps the workflow tests fast and offline: the ML model reports itself unavailable."""
    def unavailable(self, prices, benchmark_close, as_of_date):
        raise ModelUnavailable("disabled in tests")
    monkeypatch.setattr(technical_analyst.MLDirectionService, "evidence", unavailable)


# ----- 1. input validation ---------------------------------------------------
@pytest.mark.parametrize("raw, expected", [("reliance", "RELIANCE.NS"), (" tcs.bo ", "TCS.BO"), ("INFY.NS", "INFY.NS")])
def test_ticker_normalisation(raw, expected):
    assert MarketDataService(INDIA).normalize_ticker(raw) == expected


def test_non_indian_ticker_is_rejected():
    with pytest.raises(ValueError, match="not listed"):
        MarketDataService(INDIA).normalize_ticker("AAPL.US")


# ----- 2. structured messages ---------------------------------------------------
def test_schemas_reject_invalid_values():
    with pytest.raises(ValidationError):
        AnalystOpinion(signal="bullish", confidence=1.5, summary="s", key_points=[])
    with pytest.raises(ValidationError):
        AnalystOpinion(signal="very bullish", confidence=0.5, summary="s", key_points=[])
    with pytest.raises(ValidationError):
        TradeProposal(action="SHORT", position_size_pct=0.1, rationale="r")
    with pytest.raises(ValidationError):
        RiskAssessment(decision="APPROVE", risk_level="low", adjusted_position_size_pct=-0.1, comments="c")


# ----- 3. risk guardrails and finalizer -----------------------------------------
@pytest.mark.parametrize("decision, llm_size, expected", [
    ("APPROVE", 0.50, 0.20),   # above the cap -> capped
    ("APPROVE", 0.20, 0.20),   # exactly at the cap (boundary) -> unchanged
    ("ADJUST", 0.05, 0.05),    # below the cap -> unchanged
    ("REJECT", 0.15, 0.00),    # reject -> always zero
])
def test_risk_manager_guardrails(decision, llm_size, expected):
    manager = RiskManager(RecordingLLM(), FakeMarketData(), 365, max_position_pct=0.20)
    assessment = RiskAssessment(decision=decision, risk_level="high", adjusted_position_size_pct=llm_size, comments="c")
    result = manager.apply_guardrails(assessment, llm_size)
    assert result.adjusted_position_size_pct == pytest.approx(expected)


@pytest.mark.parametrize("decision, proposed, approved, action, size", [
    ("APPROVE", 0.15, 0.10, "BUY", 0.10),
    ("APPROVE", 0.05, 0.10, "BUY", 0.05),
    ("ADJUST", 0.15, 0.08, "BUY", 0.08),
    ("REJECT", 0.15, 0.00, "HOLD", 0.00),
])
def test_finalizer_rules(decision, proposed, approved, action, size):
    state = {
        "ticker": "TCS.NS", "as_of_date": "2025-06-30",
        "trade_proposal": TradeProposal(action="BUY", position_size_pct=proposed, rationale="r"),
        "risk_assessment": RiskAssessment(decision=decision, risk_level="medium",
                                          adjusted_position_size_pct=approved, comments="c"),
    }
    final = TradingWorkflow._finalize_decision(state)["final_decision"]
    assert (final.action, final.position_size_pct) == (action, pytest.approx(size))


# ----- 4. metrics -----------------------------------------------------------------
def test_metrics_on_known_returns():
    returns = pd.Series([0.10, -0.10, 0.05])
    assert metrics.cumulative_return(returns) == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert metrics.max_drawdown(returns) == pytest.approx(-0.10)
    assert metrics.value_at_risk(pd.Series(np.linspace(-0.05, 0.05, 101)), 0.95) == pytest.approx(0.045)


def test_metrics_edge_cases_do_not_crash():
    empty, flat = pd.Series(dtype=float), pd.Series([0.01, 0.01, 0.01])
    assert metrics.summarize_performance(empty) == {
        "cumulative_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "annualized_volatility": 0.0}
    assert metrics.sharpe_ratio(flat) == 0.0          # zero volatility -> 0, not a division error


# ----- 5. workflow integration ----------------------------------------------------
def test_workflow_runs_analysts_then_trader_then_risk(no_ml_model):
    llm = RecordingLLM()
    workflow = TradingWorkflow(FakeConfig(llm), FakeMarketData())
    result = workflow.run_with_details("TCS.NS", "2025-06-30")

    # Fundamentals and news are empty -> those analysts answer "no view" WITHOUT calling the LLM.
    assert llm.calls == ["AnalystOpinion", "AnalystOpinion", "TradeProposal", "RiskAssessment"]
    reports = {r.analyst: r for r in result["analyst_reports"]}
    assert set(reports) == {"fundamental_analyst", "technical_analyst", "sentiment_analyst", "news_analyst"}
    assert reports["fundamental_analyst"].confidence == 0.0 and reports["news_analyst"].confidence == 0.0

    # Trader asked for 50%, risk manager approved 50% -> guardrail caps at 20% -> final 20%.
    final = result["final_decision"]
    assert isinstance(final, FinalDecision)
    assert (final.action, final.position_size_pct) == ("BUY", pytest.approx(0.20))


# ----- 6. backtester ---------------------------------------------------------------
def test_backtester_applies_decisions_from_the_next_day():
    class ScriptedWorkflow:


        def decide(self, ticker, as_of_date, run_config=None):

            return FinalDecision(ticker=ticker, as_of_date=as_of_date, action="BUY", position_size_pct=1.0,
                                 rationale="r", risk_level="low", risk_comments="c")

    market_data = FakeMarketData()
    result = Backtester(ScriptedWorkflow(), market_data, 0.0).run("TCS.NS", "2025-01-01", "2025-06-30", 21)
    prices = market_data.get_price_history("TCS.NS", "2025-01-01", "2025-06-30")
    daily = prices["Close"].pct_change().fillna(0.0)

    # Always 100% invested from day 2: same as buy-and-hold minus the first day's return.
    expected = metrics.cumulative_return(daily.iloc[1:])
    assert result.strategy_metrics["cumulative_return"] == pytest.approx(expected)
    assert list(result.decisions["action"].unique()) == ["BUY"]

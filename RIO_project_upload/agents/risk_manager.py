"""Risk manager: reviews the trade proposal before it becomes a decision.

Two layers protect the portfolio:
  1. The LLM reads real risk numbers (volatility, VaR, drawdown) next to the
     trade proposal and decides APPROVE / ADJUST / REJECT.
  2. Hard-coded guardrails then enforce the position cap, whatever the LLM said.
"""
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.schemas import RiskAssessment
from core.state import TradingState
from evaluation.metrics import annualized_volatility, max_drawdown, value_at_risk


class RiskManager(BaseAgent):
    name = "risk_manager"

    SYSTEM_PROMPT = (
        "You are the risk manager of a trading team covering {market_name}. Review the trader's proposal against the risk data "
        "and the analysts' disagreement.\n"
        "Decide: APPROVE (keep as is), ADJUST (change the size, set adjusted_position_size_pct) or "
        "REJECT (do not trade, set adjusted_position_size_pct to 0). Capital preservation comes first: "
        "reduce the size when volatility or drawdown is high, or when analysts conflict. "
        "adjusted_position_size_pct must never exceed {max_position_pct}."
    )

    def __init__(self, llm, market_data, price_lookback_days: int, max_position_pct: float) -> None:
        super().__init__(llm)
        self.market_data = market_data
        self.price_lookback_days = price_lookback_days
        self.max_position_pct = max_position_pct

    def run(self, state: TradingState) -> Dict[str, Any]:
        ticker, as_of_date = state["ticker"], state["as_of_date"]
        self.logger.info("Reviewing risk for %s", ticker)

        risk_metrics = self.compute_risk_metrics(ticker, as_of_date)
        user_prompt = (
            f"Stock: {ticker}\nDate: {as_of_date}\n\n"
            f"Trade proposal:\n{self.to_json(state['trade_proposal'])}\n\n"
            f"Analyst reports:\n{self.to_json([r.model_dump() for r in state['analyst_reports']])}\n\n"
            f"Risk metrics (last year of daily returns):\n{self.to_json(risk_metrics)}"
        )
        system_prompt = self.SYSTEM_PROMPT.format(
            max_position_pct=self.max_position_pct, market_name=self.market_data.market["name"]
        )
        assessment = self.ask_llm(system_prompt, user_prompt, RiskAssessment)
        return {"risk_assessment": self.apply_guardrails(assessment), "risk_metrics": risk_metrics}

    def compute_risk_metrics(self, ticker: str, as_of_date: str) -> Dict[str, float]:
        prices = self.market_data.get_recent_prices(ticker, as_of_date, lookback_days=365)
        if len(prices) < 30:
            return {}
        returns = prices["Close"].pct_change().dropna()
        return {
            "annualized_volatility": round(annualized_volatility(returns), 3),
            "daily_value_at_risk_95": round(value_at_risk(returns, 0.95), 4),
            "max_drawdown": round(max_drawdown(returns), 3),
        }

    def apply_guardrails(self, assessment: RiskAssessment) -> RiskAssessment:
        """Rules that hold even if the LLM ignores its instructions."""
        capped_size = min(assessment.adjusted_position_size_pct, self.max_position_pct)
        if assessment.decision == "REJECT":
            capped_size = 0.0
        return assessment.model_copy(update={"adjusted_position_size_pct": capped_size})

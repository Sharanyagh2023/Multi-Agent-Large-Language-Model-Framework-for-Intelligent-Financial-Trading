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
        "You are the risk manager of a trading team covering {market_name}. "
    "Review the trader's proposal against the risk data and analyst disagreement.\n"
    "Decide: APPROVE (keep as is), ADJUST (reduce the size) or "
    "REJECT (do not trade).\n\n"

    "POSITION SIZE CONVENTION:\n"
    "Position sizes are decimal fractions of the total portfolio, between 0 and 1.\n"
    "Examples:\n"
    "- 0.01 means 1% of the portfolio.\n"
    "- 0.02 means 2% of the portfolio.\n"
    "- 0.03 means 3% of the portfolio.\n"
    "- 0.20 means 20% of the portfolio.\n\n"

    "The adjusted_position_size_pct field must contain the decimal fraction, "
    "not the percentage number.\n"
    "For example, if the intended allocation is 2%, return 0.02, never 0.2.\n\n"

    "The adjusted position must never exceed the trader's proposed position "
    "or the maximum configured position limit of {max_position_pct}.\n"
    "Reduce exposure when volatility or drawdown is high, or when analysts conflict.\n"
    "For REJECT, set adjusted_position_size_pct to 0.0.\n"
    "Capital preservation comes first."
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
        return {"risk_assessment": self.apply_guardrails(assessment,state["trade_proposal"].position_size_pct,)}

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

    def apply_guardrails(self, assessment: RiskAssessment,proposed_position_size_pct: float,) -> RiskAssessment:
        """Enforce position limits independently of the LLM."""
        capped_size = min(assessment.adjusted_position_size_pct,proposed_position_size_pct, self.max_position_pct)
        if assessment.decision == "REJECT":
            capped_size = 0.0
        return assessment.model_copy(update={"adjusted_position_size_pct": capped_size})

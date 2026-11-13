"""Risk manager: reviews the trade proposal before it becomes a decision.

Two layers protect the portfolio:
  1. The LLM reads real risk numbers (volatility, VaR, drawdown) next to the trade
     proposal and decides whether to let it through or REJECT it.
  2. The approved size is computed in code by volatility targeting:
         size = proposed x min(1, target_volatility / annualized_volatility)
     capped at max_position_pct. Analyst disagreement is NOT penalised again here;
     the trader's size already reflects it.
"""
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.schemas import RiskAssessment, RiskView, TradeProposal
from core.state import TradingState
from evaluation.metrics import annualized_volatility, max_drawdown, value_at_risk

MIN_POSITION_PCT = 0.01   # an approved size smaller than this becomes REJECT
SIZE_STEP = 0.005         # sizes are rounded to 0.5%


class RiskManager(BaseAgent):
    name = "risk_manager"

    SYSTEM_PROMPT = (
        "You are the risk manager of a trading team covering {market_name}. Review the trader's proposal "
        "against the market risk data: volatility, value at risk and drawdown.\n"
        "Decide REJECT only when the market risk is unacceptable for this trade (for example extreme volatility, "
        "a severe drawdown in progress or missing risk data). Otherwise answer APPROVE.\n"
        "Do NOT choose a position size: it is computed in code by scaling the proposal down when annual "
        "volatility is above the {target_volatility:.0%} target, capped at {max_position_pct:.0%}.\n"
        "Do NOT penalise analyst disagreement or low conviction again; the trader's size already accounts for it. "
        "Focus your comments on market risk. Capital preservation comes first."
    )

    def __init__(
        self, llm, market_data, price_lookback_days: int, max_position_pct: float, target_volatility: float
    ) -> None:
        super().__init__(llm)
        self.market_data = market_data
        self.price_lookback_days = price_lookback_days
        self.max_position_pct = max_position_pct
        self.target_volatility = target_volatility

    def run(self, state: TradingState) -> Dict[str, Any]:
        ticker, as_of_date = state["ticker"], state["as_of_date"]
        self.logger.info("Reviewing risk for %s", ticker)

        risk_metrics = self.compute_risk_metrics(ticker, as_of_date)
        user_prompt = (
            f"Stock: {ticker}\nDate: {as_of_date}\n\n"
            f"Trade proposal:\n{self.to_json(state['trade_proposal'])}\n\n"
            f"Risk metrics (last year of daily returns):\n{self.to_json(risk_metrics)}"
        )
        system_prompt = self.SYSTEM_PROMPT.format(
            market_name=self.market_data.market["name"],
            target_volatility=self.target_volatility,
            max_position_pct=self.max_position_pct,
        )
        view = self.ask_llm(system_prompt, user_prompt, RiskView)
        assessment = self.size_assessment(view, state["trade_proposal"], risk_metrics)
        return {"risk_assessment": assessment, "risk_metrics": risk_metrics}

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

    def volatility_scale(self, risk_metrics: Dict[str, float]) -> float:
        """1.0 at or below the target volatility, smaller above it. 1.0 when volatility is unknown."""
        volatility = risk_metrics.get("annualized_volatility")
        if not volatility:
            return 1.0
        return min(1.0, self.target_volatility / volatility)

    def size_assessment(
        self, view: RiskView, proposal: TradeProposal, risk_metrics: Dict[str, float]
    ) -> RiskAssessment:
        """Approved size from the volatility rule; the LLM can only block the trade."""
        proposed = proposal.position_size_pct
        if view.decision == "REJECT":
            return RiskAssessment(
                decision="REJECT", risk_level=view.risk_level, adjusted_position_size_pct=0.0, comments=view.comments
            )

        scale = self.volatility_scale(risk_metrics)
        size = min(round(proposed * scale / SIZE_STEP) * SIZE_STEP, self.max_position_pct)
        if proposed == 0:
            note = "No position proposed."
        elif not risk_metrics.get("annualized_volatility"):
            note = f"Volatility unavailable; size kept at {size:.1%}."
        elif scale < 1.0:
            note = (
                f"Size {size:.1%} = proposed {proposed:.1%} x volatility scale {scale:.2f} "
                f"(target {self.target_volatility:.0%} / annual volatility {risk_metrics['annualized_volatility']:.1%})."
            )
        else:
            note = f"Volatility is within the {self.target_volatility:.0%} target; size kept at {size:.1%}."

        if proposed > 0 and size < MIN_POSITION_PCT:
            decision, size = "REJECT", 0.0
            note += f" Below the {MIN_POSITION_PCT:.0%} minimum, so the trade is rejected."
        else:
            decision = "ADJUST" if size < proposed else "APPROVE"

        return RiskAssessment(
            decision=decision,
            risk_level=view.risk_level,
            adjusted_position_size_pct=size,
            comments=f"{view.comments} ({note})",
        )

"""Trader: combines the four analyst reports into one trade proposal."""
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from core.schemas import AnalystReport, TradeProposal
from core.state import TradingState


class Trader(BaseAgent):
    name = "trader"

    SYSTEM_PROMPT = (
        "You are the trader on a professional trading team covering {market_name}. Four analysts have given you structured reports. "
        "Weigh them by their confidence, notice where they agree or conflict, and propose ONE action for the "
        "stock: BUY, SELL or HOLD, with a target position size as a fraction of the portfolio.\n"
        "Rules: position_size_pct is the target weight AFTER the trade (0 means no position). "
        "It must never exceed {max_position_pct}. Use a small size when analysts disagree or confidence is low. "
        "Reports with confidence 0 had no data and must be ignored."
    )

    def __init__(self, llm, max_position_pct: float, market_name: str) -> None:
        super().__init__(llm)
        self.max_position_pct = max_position_pct
        self.market_name = market_name

    def run(self, state: TradingState) -> Dict[str, Any]:
        ticker = state["ticker"]
        self.logger.info("Forming trade proposal for %s", ticker)

        system_prompt = self.SYSTEM_PROMPT.format(
            max_position_pct=self.max_position_pct, market_name=self.market_name
        )
        user_prompt = (
            f"Stock: {ticker}\nDate: {state['as_of_date']}\n\n"
            f"Analyst reports:\n{self.format_reports(state['analyst_reports'])}"
        )
        proposal = self.ask_llm(system_prompt, user_prompt, TradeProposal)
        return {"trade_proposal": proposal}

    def format_reports(self, reports: List[AnalystReport]) -> str:
        return self.to_json([report.model_dump() for report in reports])

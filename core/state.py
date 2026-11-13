"""The shared state that flows through the LangGraph workflow."""
import operator
from typing import Annotated, Dict, List, TypedDict

from core.schemas import AnalystReport, FinalDecision, RiskAssessment, TradeProposal


class TradingState(TypedDict, total=False):
    """Every node reads from this dictionary and returns the keys it wants to update.

    `analyst_reports` uses `operator.add` as a reducer: the four analysts run in
    parallel and each one appends its report instead of overwriting the others.
    """

    ticker: str
    as_of_date: str  # YYYY-MM-DD, the "today" the agents are allowed to know about
    analyst_reports: Annotated[List[AnalystReport], operator.add]
    trade_proposal: TradeProposal
    risk_metrics: Dict[str, float]  # the numbers the risk manager reviewed (shown in the UI)
    risk_assessment: RiskAssessment
    final_decision: FinalDecision

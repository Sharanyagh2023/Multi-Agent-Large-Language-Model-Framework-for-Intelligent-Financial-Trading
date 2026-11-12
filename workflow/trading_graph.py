"""Builds the LangGraph workflow that connects the agents.

    START ─┬─> fundamental_analyst ─┐
           ├─> technical_analyst   ─┤
           ├─> sentiment_analyst   ─┼─> trader ─> risk_manager ─> finalize ─> END
           └─> news_analyst        ─┘

The four analysts run in parallel; `trader` starts only after all four finish.
"""
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base_agent import BaseAgent
from agents.fundamental_analyst import FundamentalAnalyst
from agents.news_analyst import NewsAnalyst
from agents.risk_manager import RiskManager
from agents.sentiment_analyst import SentimentAnalyst
from agents.technical_analyst import TechnicalAnalyst
from agents.trader import Trader
from config.loader import AppConfig
from core.schemas import FinalDecision
from core.state import TradingState
from data.market_data import MarketDataService


class TradingWorkflow:
    """Creates the agents from the config and wires them into a compiled graph."""

    def __init__(self, config: AppConfig, market_data: MarketDataService) -> None:
        self.config = config
        self.market_data = market_data
        self.analysts: List[BaseAgent] = self._create_analysts()
        self.trader: BaseAgent = self._create_trader()
        self.risk_manager: BaseAgent = self._create_risk_manager()
        self.graph: CompiledStateGraph = self._build_graph()

    # ----- public API -------------------------------------------------------
    def decide(self, ticker: str, as_of_date: str) -> FinalDecision:
        """Runs the whole team for one stock on one date and returns its decision."""
        final_state = self.graph.invoke({"ticker": ticker, "as_of_date": as_of_date, "analyst_reports": []})
        return final_state["final_decision"]

    def run_with_details(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        """Same as `decide` but returns the full state (analyst reports, proposal, risk review)."""
        return self.graph.invoke({"ticker": ticker, "as_of_date": as_of_date, "analyst_reports": []})

    # ----- agent creation ---------------------------------------------------
    def _llm_for(self, agent_name: str):
        return self.config.get_llm_for_agent(agent_name)

    def _create_analysts(self) -> List[BaseAgent]:
        trading = self.config.trading
        return [
            FundamentalAnalyst(self._llm_for(FundamentalAnalyst.name), self.market_data),
            TechnicalAnalyst(self._llm_for(TechnicalAnalyst.name), self.market_data, trading["price_lookback_days"]),
            SentimentAnalyst(self._llm_for(SentimentAnalyst.name), self.market_data, trading["news_limit"]),
            NewsAnalyst(self._llm_for(NewsAnalyst.name), self.market_data, trading["news_limit"]),
        ]

    def _create_trader(self) -> BaseAgent:
        return Trader(
            self._llm_for(Trader.name), self.config.trading["max_position_pct"], self.config.market["name"]
        )

    def _create_risk_manager(self) -> BaseAgent:
        trading = self.config.trading
        return RiskManager(
            self._llm_for(RiskManager.name),
            self.market_data,
            trading["price_lookback_days"],
            trading["max_position_pct"],
        )

    # ----- graph wiring -----------------------------------------------------
    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(TradingState)

        for analyst in self.analysts:
            graph.add_node(analyst.name, analyst.run)
            graph.add_edge(START, analyst.name)  # all analysts start together

        graph.add_node(self.trader.name, self.trader.run)
        graph.add_node(self.risk_manager.name, self.risk_manager.run)
        graph.add_node("finalize", self._finalize_decision)

        # Passing a list waits until EVERY analyst has finished before the trader starts.
        graph.add_edge([analyst.name for analyst in self.analysts], self.trader.name)
        graph.add_edge(self.trader.name, self.risk_manager.name)
        graph.add_edge(self.risk_manager.name, "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    @staticmethod
    def _finalize_decision(state: TradingState) -> Dict[str, Any]:
        """Plain Python (no LLM): merges the trade proposal with the risk verdict."""
        proposal = state["trade_proposal"]
        risk = state["risk_assessment"]

        if risk.decision == "REJECT":
            action, size = "HOLD", 0.0
        elif risk.decision == "ADJUST":
            action = proposal.action
            size = min(
                proposal.position_size_pct,
                risk.adjusted_position_size_pct,
            )
        else:  # APPROVE
            action = proposal.action
            size = min(
                proposal.position_size_pct,
                risk.adjusted_position_size_pct,
            )

        decision = FinalDecision(
            ticker=state["ticker"],
            as_of_date=state["as_of_date"],
            action=action,
            position_size_pct=size,
            rationale=proposal.rationale,
            risk_level=risk.risk_level,
            risk_comments=risk.comments,
        )
        return {"final_decision": decision}

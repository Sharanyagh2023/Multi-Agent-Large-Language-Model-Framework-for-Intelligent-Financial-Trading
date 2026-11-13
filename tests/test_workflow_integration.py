
from workflow.trading_graph import TradingWorkflow
from core.schemas import AnalystReport, TradeProposal, RiskAssessment
from agents.risk_manager import RiskManager

class FakeAnalyst:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def run(self, state):
        self.events.append(f"analyst:{self.name}")

        report = AnalystReport(
            analyst=self.name,
            ticker=state["ticker"],
            signal="bullish",
            confidence=0.8,
            summary=f"Mock report from {self.name}",
            key_points=["Test evidence"],
        )

        return {"analyst_reports": [report]}


class FakeTrader:
    name = "trader"

    def __init__(self, events):
        self.events = events
        self.received_report_count = 0

    def run(self, state):
        reports = state["analyst_reports"]
        self.received_report_count = len(reports)
        self.events.append(f"trader:{len(reports)}")

        return {
            "trade_proposal": TradeProposal(
                action="BUY",
                position_size_pct=0.15,
                rationale="Mock proposal based on analyst reports",
            )
        }


class FakeRiskManager:
    name = "risk_manager"

    def __init__(self, events):
        self.events = events
        self.saw_proposal = False

    def run(self, state):
        self.saw_proposal = state.get("trade_proposal") is not None
        self.events.append("risk_manager")

        return {
            "risk_assessment": RiskAssessment(
                decision="APPROVE",
                risk_level="medium",
                adjusted_position_size_pct=0.20,
                comments="Mock risk review",
            )
        }


def test_workflow_runs_agents_in_correct_order():
    events = []

    analyst_names = [
        "fundamental_analyst",
        "technical_analyst",
        "sentiment_analyst",
        "news_analyst",
    ]

    analysts = [
        FakeAnalyst(name, events)
        for name in analyst_names
    ]
    trader = FakeTrader(events)
    risk_manager = FakeRiskManager(events)

    # Construct the real workflow graph without calling its
    # constructor, which would initialize real LLM providers.
    workflow = TradingWorkflow.__new__(TradingWorkflow)
    workflow.analysts = analysts
    workflow.trader = trader
    workflow.risk_manager = risk_manager
    workflow.graph = workflow._build_graph()

    state = workflow.run_with_details(
        ticker="RELIANCE.NS",
        as_of_date="2025-06-30",
    )

    # All four analysts must contribute reports.
    reports = state["analyst_reports"]
    assert len(reports) == 4
    assert {report.analyst for report in reports} == set(analyst_names)

    # The trader must receive every analyst report.
    assert trader.received_report_count == 4

    # Risk management must happen after the trader.
    assert risk_manager.saw_proposal is True

    # Verify execution order despite parallel analyst execution.
    trader_index = events.index("trader:4")
    risk_index = events.index("risk_manager")
    analyst_indices = [
        events.index(f"analyst:{name}")
        for name in analyst_names
    ]

    assert max(analyst_indices) < trader_index
    assert trader_index < risk_index

    # Verify the finalized trading decision.
    decision = state["final_decision"]
    assert decision.action == "BUY"
    assert decision.position_size_pct == 0.15

def test_adjusted_position_cannot_exceed_trader_proposal():

    state = {
        "ticker": "RELIANCE.NS",
        "as_of_date": "2026-11-12",
        "trade_proposal": TradeProposal(
            action="BUY",
            position_size_pct=0.03,
            rationale="Test proposal",
        ),
        "risk_assessment": RiskAssessment(
            decision="ADJUST",
            risk_level="high",
            adjusted_position_size_pct=0.20,
            comments="Risk adjustment test",
        ),
    }

    result = TradingWorkflow._finalize_decision(state)

    decision = result["final_decision"]

    assert decision.position_size_pct == 0.03
    assert decision.position_size_pct <= state["trade_proposal"].position_size_pct

def test_risk_guardrail_never_exceeds_trader_proposal():

    manager = RiskManager.__new__(RiskManager)
    manager.max_position_pct = 0.20

    assessment = RiskAssessment(
        decision="ADJUST",
        risk_level="high",
        adjusted_position_size_pct=0.20,
        comments="Reduce position to 2%.",
    )

    result = manager.apply_guardrails(
        assessment,
        proposed_position_size_pct=0.02,
    )

    assert result.adjusted_position_size_pct == 0.02

    assert (
        result.adjusted_position_size_pct
        <= 0.02
    )

def test_rejected_trade_has_zero_position_size():

    manager = RiskManager.__new__(RiskManager)
    manager.max_position_pct = 0.20

    assessment = RiskAssessment(
        decision="REJECT",
        risk_level="high",
        adjusted_position_size_pct=0.10,
        comments="Risk is too high.",
    )

    result = manager.apply_guardrails(
        assessment,
        proposed_position_size_pct=0.02,
    )

    assert result.adjusted_position_size_pct == 0.0
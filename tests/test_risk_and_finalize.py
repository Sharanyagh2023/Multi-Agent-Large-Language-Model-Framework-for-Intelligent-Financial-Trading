import pytest

from agents.risk_manager import RiskManager
from core.schemas import RiskAssessment, TradeProposal
from workflow.trading_graph import TradingWorkflow


@pytest.fixture
def risk_manager():
    # These tests exercise deterministic methods, not the LLM.
    return RiskManager(
        llm=None,
        market_data=None,
        price_lookback_days=365,
        max_position_pct=0.20,
    )


def test_risk_manager_caps_position_size(risk_manager):
    assessment = RiskAssessment(
        decision="APPROVE",
        risk_level="medium",
        adjusted_position_size_pct=0.40,
        comments="Test assessment",
    )

    result = risk_manager.apply_guardrails(
        assessment,
        proposed_position_size_pct=0.80,
    )

    assert result.adjusted_position_size_pct == pytest.approx(0.20)


def test_risk_manager_reject_forces_zero_position(risk_manager):
    assessment = RiskAssessment(
        decision="REJECT",
        risk_level="high",
        adjusted_position_size_pct=0.15,
        comments="Test rejection",
    )

    result = risk_manager.apply_guardrails(
        assessment,
        proposed_position_size_pct=0.80,
    )

    assert result.adjusted_position_size_pct == 0.0


def test_approved_trade_cannot_exceed_proposed_size():
    state = {
        "ticker": "RELIANCE.NS",
        "as_of_date": "2025-06-30",
        "trade_proposal": TradeProposal(
            action="BUY",
            position_size_pct=0.15,
            rationale="Test proposal",
        ),
        "risk_assessment": RiskAssessment(
            decision="APPROVE",
            risk_level="low",
            adjusted_position_size_pct=0.20,
            comments="Approved",
        ),
    }

    result = TradingWorkflow._finalize_decision(state)

    assert result["final_decision"].action == "BUY"
    assert result["final_decision"].position_size_pct == pytest.approx(0.15)


def test_adjusted_trade_uses_risk_approved_size():
    state = {
        "ticker": "RELIANCE.NS",
        "as_of_date": "2025-06-30",
        "trade_proposal": TradeProposal(
            action="BUY",
            position_size_pct=0.18,
            rationale="Test proposal",
        ),
        "risk_assessment": RiskAssessment(
            decision="ADJUST",
            risk_level="high",
            adjusted_position_size_pct=0.08,
            comments="Reduce exposure",
        ),
    }

    result = TradingWorkflow._finalize_decision(state)

    assert result["final_decision"].action == "BUY"
    assert result["final_decision"].position_size_pct == pytest.approx(0.08)


def test_rejected_trade_becomes_hold():
    state = {
        "ticker": "RELIANCE.NS",
        "as_of_date": "2025-06-30",
        "trade_proposal": TradeProposal(
            action="BUY",
            position_size_pct=0.15,
            rationale="Test proposal",
        ),
        "risk_assessment": RiskAssessment(
            decision="REJECT",
            risk_level="high",
            adjusted_position_size_pct=0.0,
            comments="Reject trade",
        ),
    }

    result = TradingWorkflow._finalize_decision(state)

    assert result["final_decision"].action == "HOLD"
    assert result["final_decision"].position_size_pct == 0.0

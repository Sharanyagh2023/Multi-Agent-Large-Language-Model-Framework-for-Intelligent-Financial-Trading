"""Structured messages exchanged between agents.

Agents do not pass free text to each other. Each one returns one of these
validated objects, which keeps information intact and makes decisions easy to
explain and log (the "structured communication" idea from the project brief).
"""
from typing import List, Literal

from pydantic import BaseModel, Field

Signal = Literal["bullish", "neutral", "bearish"]
TradeAction = Literal["BUY", "SELL", "HOLD"]
RiskDecision = Literal["APPROVE", "ADJUST", "REJECT"]
RiskLevel = Literal["low", "medium", "high"]


class AnalystOpinion(BaseModel):
    """What the LLM must return for every analyst."""

    signal: Signal = Field(description="Overall view of the stock from this analyst's angle.")
    confidence: float = Field(ge=0.0, le=1.0, description="0 = no conviction, 1 = very high conviction.")
    summary: str = Field(description="Two or three sentences explaining the view.")
    key_points: List[str] = Field(description="The 3 to 5 most important facts behind the view.")


class AnalystReport(AnalystOpinion):
    """An opinion plus who produced it and for which stock."""

    analyst: str
    ticker: str


class TradeProposal(BaseModel):
    """The trader's recommendation, before risk review."""

    action: TradeAction
    position_size_pct: float = Field(ge=0.0, le=1.0, description="Target share of the portfolio in this stock.")
    rationale: str = Field(description="Why this action, referring to the analyst reports.")


class RiskAssessment(BaseModel):
    """The risk manager's verdict on the trade proposal."""

    decision: RiskDecision
    risk_level: RiskLevel
    adjusted_position_size_pct: float = Field(ge=0.0, le=1.0, description="Position size after risk review.")
    comments: str = Field(description="Main risks and why the decision was taken.")


class FinalDecision(BaseModel):
    """The decision the whole system outputs."""

    ticker: str
    as_of_date: str
    action: TradeAction
    position_size_pct: float
    rationale: str
    risk_level: RiskLevel
    risk_comments: str

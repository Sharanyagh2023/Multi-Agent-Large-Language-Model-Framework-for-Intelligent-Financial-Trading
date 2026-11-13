"""Base classes shared by every agent.

BaseAgent     : owns the LLM and knows how to ask it for a structured answer.
AnalystAgent  : the common recipe for the four analysts (template method):
                    1. collect_data()    - get facts from the data layer   (differs per analyst)
                    2. build_prompt()    - turn the facts into text         (differs per analyst)
                    3. ask the LLM       - same for everyone
                    4. wrap the result as an AnalystReport                  (same for everyone)
                A concrete analyst only fills in steps 1 and 2.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from core.schemas import AnalystOpinion, AnalystReport
from core.state import TradingState
from data.market_data import MarketDataService

SchemaType = TypeVar("SchemaType", bound=BaseModel)


class BaseAgent(ABC):
    """An agent is a named LLM with a role. `run` is what LangGraph calls as a node."""

    name: str = "base_agent"  # also the key used in config.yaml -> agents

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def run(self, state: TradingState) -> Dict[str, Any]:
        """Reads the shared state and returns only the keys this agent updates."""

    def ask_llm(self, system_prompt: str, user_prompt: str, output_schema: Type[SchemaType]) -> SchemaType:
        """Sends the prompts and returns the answer validated as `output_schema`."""
        structured_llm = self.llm.with_structured_output(output_schema)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        return structured_llm.invoke(messages)

    @staticmethod
    def to_json(data: Any) -> str:
        """Readable JSON for prompts; pydantic models are converted first."""
        if isinstance(data, BaseModel):
            data = data.model_dump()
        return json.dumps(data, indent=2, default=str)


class AnalystAgent(BaseAgent):
    """Base class for the fundamental, technical, sentiment and news analysts."""

    role_description: str = ""  # one sentence describing what this analyst looks at

    def __init__(self, llm: BaseChatModel, market_data: MarketDataService) -> None:
        super().__init__(llm)
        self.market_data = market_data

    SYSTEM_PROMPT_TEMPLATE = (
        "You are the {role} on a professional trading team covering {market_name}. {role_description}\n"
        "All money amounts are in {currency}.\n"
        "Use ONLY the data provided. Do not invent numbers or events. "
        "If the data is weak or contradictory, say so and lower your confidence.\n"
        "Return a signal (bullish / neutral / bearish), a confidence between 0 and 1, "
        "a short summary and the key supporting points."
    )

    @abstractmethod
    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        """Gathers the facts this analyst needs. Return an empty dict if nothing is available."""

    @abstractmethod
    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        """Turns the collected facts into the user prompt for the LLM."""

    def run(self, state: TradingState) -> Dict[str, Any]:
        ticker, as_of_date = state["ticker"], state["as_of_date"]
        self.logger.info("Analysing %s as of %s", ticker, as_of_date)

        data = self.collect_data(ticker, as_of_date)
        if not data:
            # No facts means no opinion. Skipping the LLM call also avoids hallucinated analysis.
            return {"analyst_reports": [self._no_data_report(ticker)]}

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            role=self.name.replace("_", " "),
            role_description=self.role_description,
            market_name=self.market_data.market["name"],
            currency=self.market_data.market["currency"],
        )
        opinion = self.ask_llm(system_prompt, self.build_prompt(ticker, as_of_date, data), AnalystOpinion)
        report = AnalystReport(analyst=self.name, ticker=ticker, **opinion.model_dump())
        return {"analyst_reports": [report]}

    def _no_data_report(self, ticker: str) -> AnalystReport:
        return AnalystReport(
            analyst=self.name,
            ticker=ticker,
            signal="neutral",
            confidence=0.0,
            summary="No data was available for this analyst, so no view is given.",
            key_points=[],
        )

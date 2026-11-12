"""Base classes shared by every agent.

BaseAgent     : owns the LLM and knows how to ask it for a structured answer.
AnalystAgent  : the common recipe for the four analysts (template method):
                    1. collect_data()    - get facts from the data layer   (differs per analyst)
                    2. build_prompt()    - turn the facts into text         (differs per analyst)
                    3. ask the LLM       - same for everyone
                    4. wrap the result as an AnalystReport                  (same for everyone)
                A concrete analyst only fills in steps 1 and 2.
"""

import os
import re

from langchain_ollama import ChatOllama

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

    def __init__(self, llm: BaseChatModel,fallback_llm: BaseChatModel | None = None,) -> None:
        self.llm = llm
        self._fallback_llm = fallback_llm
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def run(self, state: TradingState) -> Dict[str, Any]:
        """Reads the shared state and returns only the keys this agent updates."""

    
    def ask_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[SchemaType],
    ) -> SchemaType:
        """Return validated structured output, falling back to Ollama
        when credentials are missing or a recognized API limit is reached.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        if self._cloud_credentials_missing():
            self.logger.warning(
                "Cloud credentials are missing; using local Ollama."
            )
            return self._invoke_ollama(messages, output_schema)

        try:
            structured_llm = self.llm.with_structured_output(output_schema)
            return structured_llm.invoke(messages)
        except Exception as exc:
            if not self._is_fallback_error(exc):
                raise

            if self._is_ollama_model(self.llm):
                # Do not retry the same local model as its own fallback.
                raise

            self.logger.warning(
                "Recognized cloud quota/rate/token limit (%s); "
                "falling back to Ollama.",
                type(exc).__name__,
            )
            return self._invoke_ollama(messages, output_schema)

    def _cloud_credentials_missing(self) -> bool:
        """Check credentials for the actual provider used by this agent.

        The provider is inferred from the configured LLM's class.
        Unknown providers are not assumed to require a particular key.
        """
        module = type(self.llm).__module__.lower()
        class_name = type(self.llm).__name__.lower()

        if "ollama" in module or "ollama" in class_name:
            return False

        if "anthropic" in module or "anthropic" in class_name:
            return not os.getenv("ANTHROPIC_API_KEY", "").strip()

        if "azure" in module or "azure" in class_name:
            required = (
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
                "AZURE_OPENAI_API_VERSION",
            )
            return any(not os.getenv(key, "").strip() for key in required)

        if "openai" in module or "openai" in class_name:
            return not os.getenv("OPENAI_API_KEY", "").strip()

        # Do not trigger fallback for providers whose credentials we
        # cannot reliably identify.
        return False

    @staticmethod
    def _is_ollama_model(llm: BaseChatModel) -> bool:
        """Identify an Ollama model to prevent recursive fallback."""
        module = type(llm).__module__.lower()
        class_name = type(llm).__name__.lower()
        return "ollama" in module or "ollama" in class_name

    @staticmethod
    def _is_fallback_error(exc: Exception) -> bool:
        """Recognize quota, rate-limit and context/token-limit errors."""
        message_parts = [str(exc).lower()]

        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)

        if status is None and response is not None:
            status = getattr(response, "status_code", None)

        error_code = getattr(exc, "code", None)
        if error_code is not None:
            message_parts.append(str(error_code).lower())

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                for key in ("code", "type", "message"):
                    if error.get(key) is not None:
                        message_parts.append(str(error[key]).lower())
            else:
                message_parts.append(str(error).lower())

        message = " ".join(message_parts)

        quota_or_rate_limit = (
            "insufficient_quota",
            "quota exceeded",
            "billing hard limit",
            "rate_limit_exceeded",
            "rate limit reached",
            "too many requests",
        )

        token_or_context_limit = (
            "context_length_exceeded",
            "maximum context length",
            "max_tokens",
            "token limit",
            "tokens exceed",
            "prompt is too long",
            "context window exceeded",
            "maximum input length",
        )

        # HTTP 429 may mean either rate limiting or exhausted quota.
        if status == 429:
            return True

        return any(
            indicator in message
            for indicator in quota_or_rate_limit + token_or_context_limit
        )

    def _get_ollama_fallback(self) -> BaseChatModel:
        """Create the local model once per agent."""
        if self._fallback_llm is None:
            self._fallback_llm = ChatOllama(
                model="qwen3:4b",
                temperature=0,
            )
        return self._fallback_llm

    def _invoke_ollama(
        self,
        messages: list,
        output_schema: Type[SchemaType],
    ) -> SchemaType:
        """Run the structured-output request against local Ollama."""
        fallback = self._get_ollama_fallback()
        structured_llm = fallback.with_structured_output(output_schema)
        return structured_llm.invoke(messages)

    @staticmethod
    def to_json(data: Any) -> str:
        """Convert data, including Pydantic models, into readable JSON."""
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
        opinion = self.ask_llm(
            system_prompt,
            self.build_prompt(ticker, as_of_date, data),
            AnalystOpinion,
        )

        print("\nDEBUG - ORIGINAL FUNDAMENTAL OPINION:")
        print(opinion.model_dump_json(indent=2))

        opinion = self.validate_opinion(opinion, data)
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

    def validate_opinion(
        self,
        opinion: AnalystOpinion,
        data: Dict[str, Any],
    ) -> AnalystOpinion:
        """Default validation hook for analyst opinions."""
        return opinion


from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from agents.base_agent import BaseAgent


class ExampleOutput(BaseModel):
    answer: str


class DummyAgent(BaseAgent):
    name = "test_agent"

    def run(self, state):
        return {}


@pytest.fixture
def cloud_llm():
    return MagicMock(spec=BaseChatModel)


@pytest.fixture
def agent(cloud_llm):
    return DummyAgent(llm=cloud_llm)


def test_quota_error_falls_back_to_ollama(agent, cloud_llm):
    cloud_llm.with_structured_output.side_effect = Exception(
        "insufficient_quota: quota exceeded"
    )

    expected = ExampleOutput(answer="local response")
    mock_ollama = MagicMock()
    mock_ollama.with_structured_output.return_value.invoke.return_value = expected

    with patch.object(agent, "_get_ollama_fallback", return_value=mock_ollama):
        result = agent.ask_llm("system", "user", ExampleOutput)

    assert result == expected
    mock_ollama.with_structured_output.assert_called_once_with(ExampleOutput)


def test_unrelated_error_does_not_fall_back(agent, cloud_llm):
    cloud_llm.with_structured_output.side_effect = ValueError(
        "Invalid application configuration"
    )

    with patch.object(agent, "_get_ollama_fallback") as fallback:
        with pytest.raises(ValueError, match="Invalid application configuration"):
            agent.ask_llm("system", "user", ExampleOutput)

    fallback.assert_not_called()


def test_ollama_failure_propagates(agent, cloud_llm):
    cloud_llm.with_structured_output.side_effect = Exception(
        "rate_limit_exceeded"
    )

    mock_ollama = MagicMock()
    mock_ollama.with_structured_output.return_value.invoke.side_effect = (
        RuntimeError("Ollama server unavailable")
    )

    with patch.object(agent, "_get_ollama_fallback", return_value=mock_ollama):
        with pytest.raises(RuntimeError, match="Ollama server unavailable"):
            agent.ask_llm("system", "user", ExampleOutput)


def test_ollama_model_does_not_fall_back_to_itself():
    ollama_llm = MagicMock(spec=BaseChatModel)
    agent = DummyAgent(llm=ollama_llm)

    # Simulate an Ollama-backed model class.
    with patch.object(agent, "_is_ollama_model", return_value=True):
        ollama_llm.with_structured_output.side_effect = Exception(
            "rate_limit_exceeded"
        )

        with patch.object(agent, "_get_ollama_fallback") as fallback:
            with pytest.raises(Exception, match="rate_limit_exceeded"):
                agent.ask_llm("system", "user", ExampleOutput)

        fallback.assert_not_called()


def test_http_429_is_recognized_as_fallback_error():
    error = Exception("Too many requests")
    error.status_code = 429

    assert BaseAgent._is_fallback_error(error)


def test_context_limit_is_recognized_as_fallback_error():
    error = Exception("maximum context length exceeded")

    assert BaseAgent._is_fallback_error(error)


def test_ordinary_error_is_not_a_fallback_error():
    error = ValueError("Unexpected application bug")

    assert not BaseAgent._is_fallback_error(error)

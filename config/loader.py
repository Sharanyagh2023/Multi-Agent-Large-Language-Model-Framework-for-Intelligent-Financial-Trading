"""Loads config/config.yaml and builds chat models from it.

Instead of one Python class per LLM provider, every model is described in the
YAML file. `AppConfig.create_llm("claude")` reads that description and asks
LangChain's `init_chat_model` to build the matching chat model.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class AppConfig:
    """Typed access to config.yaml plus a factory for the configured LLMs."""

    def __init__(self, raw_config: Dict[str, Any]) -> None:
        self._raw = raw_config
        self._llm_cache: Dict[str, BaseChatModel] = {}

    # ----- settings sections ------------------------------------------------
    @property
    def trading(self) -> Dict[str, Any]:
        return self._raw["trading"]

    @property
    def market(self) -> Dict[str, Any]:
        """Settings of the market selected by `active_market`."""
        active_market = self._raw["active_market"]
        available = self._raw["markets"]
        if active_market not in available:
            raise KeyError(f"active_market '{active_market}' is not defined under markets. Available: {list(available)}")
        return available[active_market]

    @property
    def backtest(self) -> Dict[str, Any]:
        return self._raw["backtest"]

    @property
    def storage_paths(self) -> Dict[str, Path]:
        """Absolute paths of the results database and the checkpoint database."""
        storage = {"database": "outputs/rio.sqlite", "checkpoints": "outputs/checkpoints.sqlite"}
        storage.update(self._raw.get("storage") or {})
        return {name: (PROJECT_ROOT / path).resolve() for name, path in storage.items()}

    # ----- LLM access -------------------------------------------------------
    def llm_name_for_agent(self, agent_name: str) -> str:
        """Returns the llm key an agent should use (its override or the default)."""
        return self._raw.get("agents", {}).get(agent_name, self._raw["default_llm"])

    def get_llm_for_agent(self, agent_name: str) -> BaseChatModel:
        return self.create_llm(self.llm_name_for_agent(agent_name))

    def create_llm(self, llm_name: str) -> BaseChatModel:
        """Builds (once) the chat model described under `llms.<llm_name>`."""
        if llm_name in self._llm_cache:
            return self._llm_cache[llm_name]

        available = self._raw["llms"]
        if llm_name not in available:
            raise KeyError(f"LLM '{llm_name}' is not defined in config.yaml. Available: {list(available)}")

        settings = dict(available[llm_name])
        provider = settings.pop("provider")
        model = settings.pop("model")
        extra_params = {key: self._expand_env(value) for key, value in settings.items()}

        llm = init_chat_model(model=self._expand_env(model), model_provider=provider, **extra_params)
        self._llm_cache[llm_name] = llm
        return llm

    # ----- helpers ----------------------------------------------------------
    @staticmethod
    def _expand_env(value: Any) -> Any:
        """Replaces ${NAME} in strings with the environment variable NAME."""
        if not isinstance(value, str):
            return value

        def replace(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in os.environ:
                raise EnvironmentError(f"Environment variable '{name}' (used in config.yaml) is not set.")
            return os.environ[name]

        return ENV_VARIABLE_PATTERN.sub(replace, value)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as file:
        return AppConfig(yaml.safe_load(file))

"""Abstract base class shared by all Atlas agents."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import config
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class holding an LLM client, a name, and prompt-loading behaviour."""

    def __init__(self, llm_client: LLMClient, name: str, prompt_filename: str) -> None:
        """Store the shared LLM client and this agent's name and prompt file."""
        self._llm = llm_client
        self.name = name
        self._prompt_filename = prompt_filename

    def load_prompt(self) -> str:
        """Read this agent's system prompt from the prompts/ directory.

        Raises:
            ConfigError: If the prompt file cannot be read.
        """
        path = config.PROMPTS_DIR / self._prompt_filename
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise config.ConfigError(f"Could not read prompt '{path}': {exc}") from exc

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's task. Implemented by each subclass."""

"""Central configuration: environment loading, constants, and validation.

All tunable constants and secret loading live here so logic modules never
hardcode values or read environment variables directly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
PROMPTS_DIR: Path = BASE_DIR / "prompts"
REPORTS_DIR: Path = BASE_DIR / "reports"

# --- Research tuning constants --------------------------------------------
MIN_SUB_QUESTIONS: int = 2
MAX_SUB_QUESTIONS: int = 5
MAX_EVIDENCE_PER_QUESTION: int = 4
CONFIDENCE_THRESHOLD: float = 0.6
# Fractional confidence reduction applied when sources conflict.
CONFLICT_PENALTY: float = 0.3

# --- Confidence colour thresholds (used by the UI) ------------------------
CONFIDENCE_GREEN: float = 0.7
CONFIDENCE_AMBER: float = 0.4

# --- LLM constants ---------------------------------------------------------
MODEL: str = os.getenv("MODEL", "claude-fable-5")
MAX_TOKENS: int = 8000
MAX_TOOL_ITERATIONS: int = 6
# Backoff schedule (seconds) for the three LLM retry attempts.
RETRY_BACKOFFS: tuple[float, ...] = (2.0, 4.0, 8.0)
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 529})

# --- HTTP constants --------------------------------------------------------
HTTP_TIMEOUT: float = 20.0

# --- Secrets / runtime settings (from .env) -------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def configure_logging() -> None:
    """Configure root logging using the LOG_LEVEL from the environment."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def validate_config() -> None:
    """Validate that all required environment variables are present.

    Raises:
        ConfigError: If a required variable is missing, naming each one.
    """
    missing: list[str] = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in the values."
        )


configure_logging()

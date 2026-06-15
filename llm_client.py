"""All large-language-model access for Atlas lives here.

This is the only module that talks to an LLM provider. It uses the OpenAI
Python SDK against any OpenAI-compatible endpoint (Groq, Ollama, Google Gemini,
etc.), chosen entirely through configuration. It exposes a structured-output
helper used by every agent, wrapped in retry/backoff. Structured JSON output is
used (rather than native function-calling) because it is far more reliable
across free/open models.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around an OpenAI-compatible chat API used by every agent."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Create the async client from configuration (provider-agnostic)."""
        self._client = AsyncOpenAI(
            api_key=api_key or config.LLM_API_KEY or "not-needed",
            base_url=base_url or config.LLM_BASE_URL,
        )
        self._model = model or config.MODEL

    async def _create(self, **kwargs: Any) -> Any:
        """Call chat.completions.create with retry and exponential backoff.

        Retries retryable API errors with waits of 2s/4s/8s between attempts.
        Non-retryable errors (e.g. bad request, auth) propagate immediately.
        """
        last_exc: Exception | None = None
        for delay in (0.0, *config.RETRY_BACKOFFS):
            if delay:
                await asyncio.sleep(delay)
            try:
                return await self._client.chat.completions.create(
                    model=self._model, max_tokens=config.MAX_TOKENS, **kwargs
                )
            except openai.APIConnectionError as exc:
                last_exc = exc
                logger.warning("LLM connection error; will retry: %s", exc)
            except openai.APIStatusError as exc:
                if exc.status_code not in config.RETRYABLE_STATUS_CODES:
                    raise
                last_exc = exc
                logger.warning("LLM HTTP %s; will retry.", exc.status_code)
        assert last_exc is not None  # the loop always assigns before exhausting
        raise last_exc

    async def call_structured(
        self, system_prompt: str, user_message: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Call the model and return JSON parsed against the given schema.

        Requests JSON-object output and embeds the schema in the prompt. Returns
        an empty dict if the model returns no parseable JSON.
        """
        instruction = (
            "Respond with a single json object that matches this schema. "
            "Output only json — no prose and no markdown fences:\n"
            + json.dumps(response_schema)
        )
        response = await self._create(
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\n{instruction}"},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        return _parse_json(response.choices[0].message.content or "")


def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from model text, tolerating surrounding prose or code fences."""
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            logger.warning("Structured response contained no JSON object.")
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from structured response.")
            return {}

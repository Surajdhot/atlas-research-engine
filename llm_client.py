"""All Claude (Anthropic) API access for Atlas lives here.

This is the only module that talks to the Anthropic SDK. It exposes a tool-use
loop for agentic retrieval and a structured-output helper for planning and
synthesis, both wrapped in retry/backoff. The model is read from config and is
called with the parameters supported by ``claude-fable-5`` (no ``thinking`` or
sampling parameters, which that model rejects).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import anthropic

import config

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class ToolSpec:
    """An Anthropic tool definition bound to an async handler that executes it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


class LLMClient:
    """Async wrapper around the Anthropic Messages API used by every agent."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Create the async Anthropic client from configuration."""
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or config.ANTHROPIC_API_KEY
        )
        self._model = model or config.MODEL

    async def _create(self, **kwargs: Any) -> Any:
        """Call messages.create with retry and exponential backoff.

        Retries retryable API errors with waits of 2s/4s/8s between attempts.
        Non-retryable errors (e.g. bad request, auth) propagate immediately.
        """
        last_exc: Exception | None = None
        for delay in (0.0, *config.RETRY_BACKOFFS):
            if delay:
                await asyncio.sleep(delay)
            try:
                return await self._client.messages.create(
                    model=self._model, max_tokens=config.MAX_TOKENS, **kwargs
                )
            except anthropic.APIConnectionError as exc:
                last_exc = exc
                logger.warning("Anthropic connection error; will retry: %s", exc)
            except anthropic.APIStatusError as exc:
                if exc.status_code not in config.RETRYABLE_STATUS_CODES:
                    raise
                last_exc = exc
                logger.warning("Anthropic HTTP %s; will retry.", exc.status_code)
        assert last_exc is not None  # the loop always assigns before exhausting
        raise last_exc

    async def call_with_tools(
        self, system_prompt: str, user_message: str, tools: list[ToolSpec]
    ) -> str:
        """Run a full tool-use loop until the model stops requesting tools.

        The model may call any provided tool repeatedly; each call is executed
        via its handler and the result fed back. Returns the final text answer.
        """
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        handlers = {t.name: t.handler for t in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = await self._create(
                system=system_prompt, messages=messages, tools=api_tools
            )
            if response.stop_reason == "refusal":
                logger.warning("Model refused the tool-use request.")
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return _extract_text(response.content)
            tool_results = await self._run_tool_calls(response.content, handlers)
            messages.append({"role": "user", "content": tool_results})
        logger.info("Tool-use loop ended without a final text answer.")
        return ""

    async def _run_tool_calls(
        self, content: list[Any], handlers: dict[str, ToolHandler]
    ) -> list[dict[str, Any]]:
        """Execute every tool_use block and collect matching tool_result blocks."""
        results: list[dict[str, Any]] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            text, is_error = await _safe_handle(
                handlers.get(block.name), block.name, block.input
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                    "is_error": is_error,
                }
            )
        return results

    async def call_structured(
        self, system_prompt: str, user_message: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Call the model and return JSON parsed against the given schema.

        Uses Anthropic structured outputs to constrain the response. Returns an
        empty dict if the model refuses or returns no parseable JSON.
        """
        response = await self._create(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": response_schema}},
        )
        if response.stop_reason == "refusal":
            logger.warning("Model refused the structured request.")
            return {}
        return _parse_json(_extract_text(response.content))


async def _safe_handle(
    handler: ToolHandler | None, name: str, tool_input: dict[str, Any]
) -> tuple[str, bool]:
    """Execute one tool handler, converting failures into tool error results."""
    if handler is None:
        return f"Unknown tool: {name}", True
    try:
        return await handler(tool_input), False
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Tool '%s' failed: %s", name, exc)
        return f"Tool '{name}' failed: {exc}", True


def _extract_text(content: list[Any]) -> str:
    """Concatenate all text blocks from a response content list."""
    parts = [
        block.text for block in content if getattr(block, "type", None) == "text"
    ]
    return "\n".join(parts).strip()


def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from model text, tolerating surrounding prose or code fences."""
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

"""All large-language-model access for Atlas lives here.

This is the only module that talks to an LLM provider. It uses the OpenAI
Python SDK against any OpenAI-compatible endpoint (Groq, Ollama, Google Gemini,
etc.), chosen entirely through configuration. It exposes a tool-use loop for
agentic retrieval and a structured-output helper for planning and synthesis,
both wrapped in retry/backoff.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import openai
from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class ToolSpec:
    """A tool definition bound to an async handler that executes it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


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

    async def call_with_tools(
        self, system_prompt: str, user_message: str, tools: list[ToolSpec]
    ) -> str:
        """Run a full tool-use loop until the model stops requesting tools.

        The model may call any provided tool repeatedly; each call is executed
        via its handler and the result fed back. Returns the final text answer.
        """
        api_tools = [_to_openai_tool(t) for t in tools]
        handlers = {t.name: t.handler for t in tools}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = await self._create(
                messages=messages, tools=api_tools, tool_choice="auto"
            )
            message = response.choices[0].message
            messages.append(_assistant_message(message))
            if not message.tool_calls:
                return (message.content or "").strip()
            for call in message.tool_calls:
                result = await _dispatch(call, handlers)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        logger.info("Tool-use loop ended without a final text answer.")
        return ""

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


def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into an OpenAI function-tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _assistant_message(message: Any) -> dict[str, Any]:
    """Rebuild an assistant message dict (with any tool calls) for the history."""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return payload


async def _dispatch(call: Any, handlers: dict[str, ToolHandler]) -> str:
    """Parse a tool call's arguments and execute the matching handler."""
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    text, _is_error = await _safe_handle(
        handlers.get(call.function.name), call.function.name, args
    )
    return text


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

"""OpenAI chat adapter.

The only provider-specific code in the system. It maps our `LLMMessage` roles
onto the provider's, translates tool specs into the provider's function-calling
schema, and normalises usage accounting back into our `LLMResponse`.
"""

from __future__ import annotations

import json
from typing import Any

from archivist.types import LLMMessage, LLMResponse, ToolCall

try:  # optional dependency, installed via the `openai` extra
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only without the extra
    OpenAI = None

_ROLE_MAP = {"system": "system", "context": "system", "user": "user", "assistant": "assistant"}


class OpenAIChatLLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._client = client
        self._temperature = temperature

    @property
    def name(self) -> str:
        return self._model

    def _ensure_client(self) -> Any:
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("install the 'openai' extra to use OpenAIChatLLM")
            self._client = OpenAI()
        return self._client

    @staticmethod
    def _to_function_schema(tool: dict[str, object]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": _ROLE_MAP.get(message.role, "user"), "content": message.content}
                for message in messages
            ],
        }
        if tools:
            payload["tools"] = [self._to_function_schema(tool) for tool in tools]

        response = self._ensure_client().chat.completions.create(**payload)
        choice = response.choices[0].message
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "calls": 1,
        }

        tool_call: ToolCall | None = None
        if getattr(choice, "tool_calls", None):
            call = choice.tool_calls[0]
            tool_call = ToolCall(
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )

        return LLMResponse(text=choice.content or "", usage=usage, tool_call=tool_call)

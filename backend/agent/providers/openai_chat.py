# pyright: reportUnknownVariableType=false
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)
from agent.providers.pricing import MODEL_PRICING
from agent.providers.token_usage import TokenUsage
from agent.state import ensure_str
from agent.tools import CanonicalToolDefinition, ToolCall, parse_json_arguments
from llm import Llm, get_openai_chat_api_name


def serialize_openai_chat_tools(
    tools: List[CanonicalToolDefinition],
) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for tool in tools:
        serialized.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        )
    return serialized


@dataclass
class OpenAIChatParseState:
    assistant_text: str = ""
    tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    turn_usage: TokenUsage | None = None


def _extract_chat_usage(chunk: Any) -> TokenUsage | None:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    prompt_details = getattr(usage, "prompt_tokens_details", None) or {}
    cached_tokens = 0
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens", 0) or 0
    elif hasattr(prompt_details, "cached_tokens"):
        cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

    return TokenUsage(
        input=prompt_tokens - cached_tokens,
        output=completion_tokens,
        cache_read=cached_tokens,
        cache_write=0,
        total=total_tokens,
    )


async def _parse_chunk(
    chunk: Any,
    state: OpenAIChatParseState,
    on_event: EventSink,
) -> None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        usage = _extract_chat_usage(chunk)
        if usage is not None:
            state.turn_usage = usage
        return

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None:
        return

    content = getattr(delta, "content", None)
    if content:
        state.assistant_text += content
        await on_event(StreamEvent(type="assistant_delta", text=content))

    tool_calls_delta = getattr(delta, "tool_calls", None)
    if tool_calls_delta:
        for tc_delta in tool_calls_delta:
            index = getattr(tc_delta, "index", None)
            if index is None:
                continue

            entry = state.tool_calls.setdefault(
                index,
                {"id": "", "name": "", "arguments": ""},
            )

            tc_id = getattr(tc_delta, "id", None)
            if tc_id:
                entry["id"] = tc_id

            func = getattr(tc_delta, "function", None)
            if func:
                fname = getattr(func, "name", None)
                if fname:
                    entry["name"] = fname
                fargs = getattr(func, "arguments", None)
                if fargs:
                    entry["arguments"] += fargs

            await on_event(
                StreamEvent(
                    type="tool_call_delta",
                    tool_call_id=entry["id"],
                    tool_name=entry.get("name"),
                    tool_arguments=entry.get("arguments"),
                )
            )

    if getattr(choice, "finish_reason", None) == "stop":
        usage = _extract_chat_usage(chunk)
        if usage is not None:
            state.turn_usage = usage


def _build_chat_provider_turn(state: OpenAIChatParseState) -> ProviderTurn:
    tool_calls: List[ToolCall] = []
    for index in sorted(state.tool_calls.keys()):
        entry = state.tool_calls[index]
        raw_args = entry.get("arguments", "")
        args, error = parse_json_arguments(raw_args)
        if error:
            args = {"INVALID_JSON": ensure_str(raw_args)}
        tool_calls.append(
            ToolCall(
                id=entry.get("id") or f"call-{index}",
                name=entry.get("name") or "unknown_tool",
                arguments=args,
            )
        )

    assistant_tool_calls: Optional[List[Dict[str, Any]]] = None
    if tool_calls:
        assistant_tool_calls = []
        for entry in state.tool_calls.values():
            assistant_tool_calls.append(
                {
                    "id": entry.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": entry.get("name", ""),
                        "arguments": entry.get("arguments", ""),
                    },
                }
            )

    assistant_turn: Dict[str, Any] = {
        "role": "assistant",
        "content": state.assistant_text or None,
    }
    if assistant_tool_calls:
        assistant_turn["tool_calls"] = assistant_tool_calls

    return ProviderTurn(
        assistant_text=state.assistant_text,
        tool_calls=tool_calls,
        assistant_turn=assistant_turn,
    )


class OpenAIChatProviderSession(ProviderSession):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: Llm,
        prompt_messages: List[ChatCompletionMessageParam],
        tools: List[Dict[str, Any]],
        api_name: str | None = None,
    ):
        self._client = client
        self._model = model
        self._api_name = api_name or get_openai_chat_api_name(model)
        self._tools = tools
        self._total_usage = TokenUsage()
        self._messages: List[ChatCompletionMessageParam] = list(prompt_messages)

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        model_name = self._api_name
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": self._messages,
            "tools": self._tools,
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 16384,
        }

        state = OpenAIChatParseState()
        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            await _parse_chunk(chunk, state, on_event)

        if state.turn_usage is not None:
            self._total_usage.accumulate(state.turn_usage)

        return _build_chat_provider_turn(state)

    def append_tool_results(
        self,
        turn: ProviderTurn,
        executed_tool_calls: List[ExecutedToolCall],
    ) -> None:
        assistant_message = turn.assistant_turn
        if isinstance(assistant_message, dict):
            self._messages.append(assistant_message)

        for executed in executed_tool_calls:
            tool_message: Dict[str, Any] = {
                "role": "tool",
                "tool_call_id": executed.tool_call.id,
                "content": json.dumps(executed.result.result),
            }
            self._messages.append(tool_message)

    async def close(self) -> None:
        u = self._total_usage
        model_name = self._api_name
        pricing = MODEL_PRICING.get(model_name)
        cost_str = f" cost=${u.cost(pricing):.4f}" if pricing else ""
        cache_hit_rate_str = f" cache_hit_rate={u.cache_hit_rate_percent():.2f}%"
        print(
            f"[TOKEN USAGE] provider=openai_chat model={model_name} | "
            f"input={u.input} output={u.output} "
            f"cache_read={u.cache_read} cache_write={u.cache_write} "
            f"total={u.total}{cache_hit_rate_str}{cost_str}"
        )
        await self._client.close()

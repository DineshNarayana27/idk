"""
api.py — provider-agnostic streaming client via LiteLLM.

LiteLLM normalizes every provider to the OpenAI format. We normalize
LiteLLM's output to OUR event stream. query.py sees only our events.

Compared to claude.ts (~2000 lines, Anthropic-native):
  ✅ Multi-provider (Groq, Gemini, NVIDIA, OpenRouter, Anthropic)
  ✅ Prompt caching via cache_control markers (LiteLLM forwards)
  ✅ Cache markers on system + last user + last assistant (matches claude.ts)
  ✅ Media stripping (matches stripExcessMediaItems)
  ✅ Thinking via keyword detection + model capability check
  ✅ Fallback on rate limit + non-streaming fallback on stream failure
  ✅ Usage accumulation across chunks

  ❌ Fine-grained tool streaming (eager_input_streaming) — LiteLLM limitation
  ❌ Beta headers — LiteLLM handles its own; use extra_headers per-call if needed
  ❌ 1h cache TTL — LiteLLM only forwards base ephemeral
  ❌ task_budget — Anthropic-native only
  ❌ Request ID chaining — provider-inconsistent
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

import litellm

from messages import to_openai
from tool import to_openai_schema

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.drop_params = True


def _sync_provider_credentials() -> None:
    """Bridge the app's legacy NVIDIA key name to LiteLLM's NIM name."""
    if not os.environ.get("NVIDIA_NIM_API_KEY") and os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_NIM_API_KEY"] = os.environ["NVIDIA_API_KEY"]


_sync_provider_credentials()


# ── Provider catalog (same as before) ────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "groq": {
        "prefix": "groq/",
        "env_key": "GROQ_API_KEY",
        "models": [
            "groq/openai/gpt-oss-120b",
            "groq/openai/gpt-oss-20b",
        ],
    },
    "gemini": {
        "prefix": "gemini/",
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini/gemini-3.8-flash"],
    },
    "nvidia": {
        "prefix": "nvidia_nim/",
        "env_key": "NVIDIA_NIM_API_KEY",
        "legacy_env_key": "NVIDIA_API_KEY",
        "models": [
            "nvidia_nim/meta/llama-3.3-70b-instruct",
            "nvidia_nim/nvidia/nemotron-3.5-bolt-e4b",
        ],
    },
    "openrouter": {
        "prefix": "openrouter/",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            "openrouter/deepseek/deepseek-r1:free",
            "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
        ],
    },
}

def _normalize_model_name(model: str) -> str:
    if model.startswith("nvidia/"):
        return f"nvidia_nim/{model}"
    return model


ACTIVE_MODEL = _normalize_model_name(
    os.environ.get("ACTIVE_MODEL", "groq/openai/gpt-oss-120b")
)


def _configured_fallback_models() -> list[str]:
    configured = os.environ.get("LITELLM_FALLBACK_MODELS", "")
    return [
        _normalize_model_name(model.strip())
        for model in configured.split(",")
        if model.strip()
    ]


FALLBACK_MODELS: list[str] = _configured_fallback_models()


# ── Constants ────────────────────────────────────────────────────────────

MAX_MEDIA_PER_REQUEST = 100


# ── Mode control ─────────────────────────────────────────────────────────

def set_active_model(model: str) -> None:
    global ACTIVE_MODEL
    ACTIVE_MODEL = _normalize_model_name(model)


def get_active_model() -> str:
    return ACTIVE_MODEL


def list_models() -> list[str]:
    _sync_provider_credentials()
    return [
        m for provider in PROVIDERS.values()
        if (
            os.environ.get(provider["env_key"])
            or os.environ.get(provider.get("legacy_env_key", ""))
        )
        for m in provider["models"]
    ]


# ── Thinking detection ──────────────────────────────────────────────────

_THINKING_HARD_KEYWORDS = {"ultrathink", "think harder", "think deeply"}
_THINKING_KEYWORDS = {"think", "plan", "analyze", "why", "reason", "explain"}


def _model_supports_thinking(model: str) -> bool:
    """Check if the model supports thinking/reasoning.

    Check our known-prefix list FIRST (strong signal, we control it).
    Then fall back to LiteLLM's database for newer models. This is
    defense in depth — LiteLLM's model registry can lag new releases,
    and we don't want that to silently disable thinking.
    """
    lower = model.lower()
    # Known thinking-capable model families
    known_thinking_prefixes = (
        "claude", "sonnet", "opus", "haiku",
        "deepseek-r1", "deepseek-reasoner",
        "o1", "o3", "o4",
        "gemini-2.5", "gemini-2.0-flash-thinking",
        "qwen", "qwq",
        "grok-3",
    )
    if any(prefix in lower for prefix in known_thinking_prefixes):
        return True
    # Fall back to LiteLLM's database
    try:
        return bool(litellm.supports_reasoning(model=model))
    except Exception:
        return False


def should_enable_thinking(messages: list, model: str | None = None) -> bool:
    """Enable thinking for hard tasks + on models that support it."""
    if model is None:
        model = ACTIVE_MODEL

    if not _model_supports_thinking(model):
        return False
    if not messages:
        return False
    last_user = next(
        (m for m in reversed(messages)
         if getattr(m, "type", None) == "user" and not getattr(m, "is_meta", False)),
        None,
    )
    if last_user is None:
        return False
    content = getattr(last_user, "content", "")
    text = content if isinstance(content, str) else ""
    lower = text.lower()
    if any(kw in lower for kw in _THINKING_HARD_KEYWORDS):
        return True
    if any(kw in lower for kw in _THINKING_KEYWORDS):
        return True
    if len(text) > 500:
        return True
    return False


# ── System prompt freezing ──────────────────────────────────────────────

def _freeze_system_prompt(system_prompt: str) -> list[dict]:
    if not system_prompt:
        return []
    return [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]


# ── Cache control on user + assistant messages ──────────────────────────

def _apply_cache_control_to_messages(oai_messages: list[dict]) -> list[dict]:
    """Tag last user + last assistant blocks with cache_control.

    Matches userMessageToMessageParam / assistantMessageToMessageParam
    from claude.ts. LiteLLM forwards cache_control to providers that
    support it.
    """
    if not oai_messages:
        return oai_messages

    last_user_idx = None
    last_asst_idx = None
    for i in range(len(oai_messages) - 1, -1, -1):
        role = oai_messages[i].get("role")
        if role == "user" and last_user_idx is None:
            last_user_idx = i
        elif role == "assistant" and last_asst_idx is None:
            last_asst_idx = i
        if last_user_idx is not None and last_asst_idx is not None:
            break

    def _tag(idx):
        if idx is None:
            return
        msg = oai_messages[idx]
        content = msg.get("content")
        if isinstance(content, list) and content:
            last = content[-1]
            if last.get("type") not in ("thinking", "redacted_thinking"):
                content[-1] = {**last, "cache_control": {"type": "ephemeral"}}

    _tag(last_user_idx)
    _tag(last_asst_idx)
    return oai_messages


# ── Media stripping ──────────────────────────────────────────────────────

def _strip_excess_media(oai_messages: list[dict], limit: int = MAX_MEDIA_PER_REQUEST) -> list[dict]:
    """Cap total media blocks per request. Strips oldest first.

    Port of stripExcessMediaItems from claude.ts.
    """
    positions: list[tuple[int, int]] = []
    for i, msg in enumerate(oai_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if block.get("type") in ("image_url", "image", "document"):
                positions.append((i, j))

    excess = len(positions) - limit
    if excess <= 0:
        return oai_messages

    to_remove = set(positions[:excess])
    for i, msg in enumerate(oai_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        oai_messages[i] = {
            **msg,
            "content": [
                {"type": "text", "text": "[media removed: request limit]"}
                if (i, j) in to_remove else block
                for j, block in enumerate(content)
            ],
        }
    return oai_messages


# ── Tool call accumulator ───────────────────────────────────────────────

class ToolCallAccumulator:
    def __init__(self):
        self._calls: dict[int, dict] = {}

    def feed(self, delta_tool_calls: list) -> None:
        for tc in delta_tool_calls:
            idx = getattr(tc, "index", 0)
            if idx not in self._calls:
                self._calls[idx] = {"id": "", "name": "", "arguments": ""}
            if tc.id:
                self._calls[idx]["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    self._calls[idx]["name"] += fn.name
                if getattr(fn, "arguments", None):
                    self._calls[idx]["arguments"] += fn.arguments

    def assemble(self) -> list[dict]:
        result = []
        for idx in sorted(self._calls):
            tc = self._calls[idx]
            if not tc["name"]:
                continue
            try:
                parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning("Malformed tool args for %s", tc["name"])
                parsed = {}
            result.append({
                "id": tc["id"] or f"call_{idx}",
                "name": tc["name"],
                "input": parsed,
            })
        return result


# ── Usage extraction ────────────────────────────────────────────────────

def _extract_usage(chunk) -> dict | None:
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


# ── The stream function ─────────────────────────────────────────────────

async def stream(
    messages: list,
    tools: list | None = None,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 8192,
    enable_thinking: bool | None = None,
    signal: Any = None,
) -> AsyncGenerator[dict, None]:
    """Stream a model response with normalized events.

    Events:
      {"type": "message_start", "model": str}
      {"type": "text_delta", "text": str}
      {"type": "thinking_delta", "thinking": str}
      {"type": "tool_use", "id": str, "name": str, "input": dict}
      {"type": "message_stop", "stop_reason": str, "usage": dict, "model": str}
    """
    target_model = model or ACTIVE_MODEL
    if _signal_aborted(signal):
        yield {"type": "message_start", "model": target_model}
        yield {
            "type": "message_stop", "stop_reason": "interrupted",
            "usage": _empty_usage(), "model": target_model,
        }
        return
    oai_messages = to_openai(messages)

    # Apply cache control + media stripping
    oai_messages = _apply_cache_control_to_messages(oai_messages)
    oai_messages = _strip_excess_media(oai_messages)

    system_blocks = _freeze_system_prompt(system_prompt)
    if system_blocks:
        oai_messages = [{"role": "system", "content": system_blocks}, *oai_messages]

    kwargs: dict = {
        "model": target_model,
        "messages": oai_messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = [to_openai_schema(t) for t in tools]
        kwargs["tool_choice"] = "auto"

    if enable_thinking is None:
        enable_thinking = should_enable_thinking(messages, target_model)

    response = await _call_with_fallback(kwargs, enable_thinking)

    yield {"type": "message_start", "model": target_model}

    text_buf: list[str] = []
    thinking_buf: list[str] = []
    accumulator = ToolCallAccumulator()
    stop_reason: str | None = None
    final_usage: dict | None = None

    try:
        async for chunk in _iterate_response(response, signal):
            usage = _extract_usage(chunk)
            if usage is not None:
                final_usage = usage

            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(choice, "finish_reason", None):
                stop_reason = choice.finish_reason

            content = getattr(delta, "content", None)
            if content:
                text_buf.append(content)
                yield {"type": "text_delta", "text": content}

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                thinking_buf.append(reasoning)
                yield {"type": "thinking_delta", "thinking": reasoning}

            tool_call_deltas = getattr(delta, "tool_calls", None)
            if tool_call_deltas:
                accumulator.feed(tool_call_deltas)
    except litellm.APIConnectionError as e:
        # Stream failed mid-response — retry non-streaming
        logger.warning("Stream failed, retrying non-streaming: %s", e)
        try:
            text = await complete(
                messages, system_prompt=system_prompt,
                model=target_model, max_tokens=max_tokens,
            )
            yield {"type": "text_delta", "text": text}
            yield {
                "type": "message_stop", "stop_reason": "end_turn",
                "usage": final_usage or {"input_tokens": 0, "output_tokens": 0,
                                         "cache_read_input_tokens": 0,
                                         "cache_creation_input_tokens": 0},
                "model": target_model, "text": text, "thinking": None,
            }
            return
        except Exception as e2:
            logger.error("Non-streaming fallback failed: %s", e2)
            raise e

    if _signal_aborted(signal):
        yield {
            "type": "message_stop", "stop_reason": "interrupted",
            "usage": final_usage or _empty_usage(), "model": target_model,
        }
        return

    # Emit tool calls after the stream (fragments are fully assembled)
    for tc in accumulator.assemble():
        yield {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}

    if stop_reason == "tool_calls":
        stop_reason = "tool_use"
    elif stop_reason == "stop":
        stop_reason = "end_turn"
    elif stop_reason is None:
        stop_reason = "end_turn"

    yield {
        "type": "message_stop",
        "stop_reason": stop_reason,
        "usage": final_usage or {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        },
        "model": target_model,
        "text": "".join(text_buf) if text_buf else None,
        "thinking": "".join(thinking_buf) if thinking_buf else None,
    }


def _empty_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def _signal_aborted(signal: Any) -> bool:
    return signal is not None and bool(getattr(signal, "aborted", False))


async def _wait_for_abort(signal: Any) -> None:
    if _signal_aborted(signal):
        return
    wait = getattr(signal, "wait", None)
    if callable(wait):
        await wait()
        return
    while not _signal_aborted(signal):
        await asyncio.sleep(0.05)


async def _iterate_response(response: Any, signal: Any):
    iterator = response.__aiter__()
    while True:
        if _signal_aborted(signal):
            break
        next_task = asyncio.create_task(iterator.__anext__())
        abort_task = (
            asyncio.create_task(_wait_for_abort(signal))
            if signal is not None else None
        )
        if abort_task is None:
            try:
                yield await next_task
            except StopAsyncIteration:
                return
            continue

        done, _ = await asyncio.wait(
            (next_task, abort_task), return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            break
        abort_task.cancel()
        await asyncio.gather(abort_task, return_exceptions=True)
        try:
            yield next_task.result()
        except StopAsyncIteration:
            return

    close = getattr(iterator, "aclose", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


# ── Fallback on rate limit ──────────────────────────────────────────────

async def _call_with_fallback(kwargs: dict, enable_thinking: bool):
    original_model = kwargs["model"]
    fallback_models = [
        model for model in _configured_fallback_models()
        if model != original_model
    ]
    attempts = [original_model, *fallback_models]

    primary_rate_limit: Exception | None = None
    last_fallback_error: Exception | None = None
    for model in attempts:
        try:
            kwargs["model"] = model
            call_kwargs = dict(kwargs)
            if enable_thinking and _model_supports_thinking(model):
                call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            response = await litellm.acompletion(**call_kwargs)
            if model != original_model:
                logger.info("Fell back from %s to %s", original_model, model)
            return response
        except litellm.RateLimitError as e:
            if model == original_model:
                primary_rate_limit = e
            else:
                last_fallback_error = e
            logger.warning("Rate limit on %s", model)
        except litellm.ContextWindowExceededError:
            raise
        except Exception as e:
            if model == original_model:
                raise
            last_fallback_error = e
            logger.warning("Configured fallback %s failed: %s", model, e)

    kwargs["model"] = original_model
    if primary_rate_limit is not None:
        if last_fallback_error is not None:
            raise primary_rate_limit from last_fallback_error
        raise primary_rate_limit
    raise RuntimeError("No configured model could be called")


# ── Non-streaming complete ──────────────────────────────────────────────

async def complete(
    messages: list,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 512,
    response_format: dict | None = None,
) -> str:
    """Non-streaming one-shot completion. Used by compaction and classifier."""
    target_model = model or ACTIVE_MODEL
    oai_messages = to_openai(messages)

    system_blocks = _freeze_system_prompt(system_prompt)
    if system_blocks:
        oai_messages = [{"role": "system", "content": system_blocks}, *oai_messages]

    kwargs: dict = {
        "model": target_model,
        "messages": oai_messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        kwargs["response_format"] = response_format

    try:
        response = await litellm.acompletion(**kwargs)
    except litellm.RateLimitError as primary_rate_limit:
        last_fallback_error: Exception | None = None
        for fallback in _configured_fallback_models():
            try:
                kwargs["model"] = fallback
                response = await litellm.acompletion(**kwargs)
                break
            except Exception as error:
                last_fallback_error = error
        else:
            if last_fallback_error is not None:
                raise primary_rate_limit from last_fallback_error
            raise

    return response.choices[0].message.content or ""
"""test_api.py — mock litellm so we don't hit the network."""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import sys


class MockRateLimitError(Exception):
    pass


class MockContextWindowExceededError(Exception):
    pass


class MockAPIError(Exception):
    pass


class MockAPIConnectionError(Exception):
    pass


class MockBadRequestError(Exception):
    pass

mock_litellm = MagicMock()
mock_litellm.acompletion = AsyncMock()
mock_litellm.RateLimitError = MockRateLimitError
mock_litellm.ContextWindowExceededError = MockContextWindowExceededError
mock_litellm.APIError = MockAPIError
mock_litellm.APIConnectionError = MockAPIConnectionError
mock_litellm.supports_reasoning = MagicMock(return_value=False)
sys.modules["litellm"] = mock_litellm

mock_dotenv = MagicMock()
sys.modules["dotenv"] = mock_dotenv

from api import (
    stream, complete, should_enable_thinking,
    ToolCallAccumulator, _freeze_system_prompt,
    _apply_cache_control_to_messages, _strip_excess_media,
    _model_supports_thinking,
    get_active_model, set_active_model, list_models,
    ACTIVE_MODEL, FALLBACK_MODELS, _sync_provider_credentials,
)
from messages import create_user_message


def mock_chunk(text=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    c = MagicMock()
    c.choices = [MagicMock()]
    d = MagicMock()
    d.content = text
    d.reasoning_content = reasoning
    d.tool_calls = tool_calls
    c.choices[0].delta = d
    c.choices[0].finish_reason = finish
    c.usage = usage
    return c


def mock_usage(in_tok=100, out_tok=20, cache_read=50, cache_create=0):
    u = MagicMock()
    u.prompt_tokens = in_tok
    u.completion_tokens = out_tok
    u.cache_read_input_tokens = cache_read
    u.cache_creation_input_tokens = cache_create
    return u


async def stream_yields(response_chunks):
    for c in response_chunks:
        yield c


async def main():
    # ── Thinking detection (no model arg → defaults to ACTIVE_MODEL) ──
    assert should_enable_thinking([create_user_message("think about this")]) is False  # ACTIVE_MODEL may not support
    # Force a model that supports thinking
    assert should_enable_thinking(
        [create_user_message("think about this")], "claude-sonnet-4-5"
    ) is True
    assert should_enable_thinking(
        [create_user_message("plan the refactor")], "claude-sonnet-4-5"
    ) is True
    assert should_enable_thinking(
        [create_user_message("ultrathink")], "claude-sonnet-4-5"
    ) is True
    assert should_enable_thinking(
        [create_user_message("read the file")], "claude-sonnet-4-5"
    ) is False
    assert should_enable_thinking(
        [create_user_message("x" * 600)], "claude-sonnet-4-5"
    ) is True
    # Model that doesn't support thinking → always False
    assert should_enable_thinking(
        [create_user_message("think about this")], "groq/llama-3.3-70b-versatile"
    ) is False
    print("✅ Thinking detection OK")

    # ── Frozen system prompt ──
    assert _freeze_system_prompt("") == []
    blocks = _freeze_system_prompt("You are helpful")
    assert len(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    print("✅ System prompt freezing OK")

    # ── Cache control on user + assistant messages ──
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    tagged = _apply_cache_control_to_messages([m.copy() for m in msgs])
    assert "cache_control" in tagged[2]["content"][-1]
    assert "cache_control" in tagged[1]["content"][-1]
    assert "cache_control" not in tagged[0]["content"][-1]
    print("✅ Cache control on messages OK")

    # ── Thinking blocks don't get cache_control ──
    msgs = [{"role": "assistant", "content": [
        {"type": "text", "text": "answer"},
        {"type": "thinking", "thinking": "reasoning"},
    ]}]
    tagged = _apply_cache_control_to_messages([m.copy() for m in msgs])
    assert "cache_control" not in tagged[0]["content"][-1]
    print("✅ Thinking block cache exclusion OK")

    # ── Media stripping ──
    big_msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"img{i}"}} for i in range(50)
    ]}]
    stripped = _strip_excess_media([m.copy() for m in big_msgs], limit=20)
    media_count = sum(1 for b in stripped[0]["content"] if b.get("type") == "image_url")
    assert media_count == 20
    print("✅ Media stripping OK")

    # ── Model capability check ──
    assert _model_supports_thinking("claude-sonnet-4-5") is True
    assert _model_supports_thinking("deepseek-r1") is True
    assert _model_supports_thinking("llama-3.3-70b") is False
    print("✅ Model capability check OK")

    # ── Tool call accumulator ──
    acc = ToolCallAccumulator()
    tc1 = MagicMock(index=0, id="call_abc")
    tc1.function = MagicMock(name="Bash", arguments='{"comm')
    tc2 = MagicMock(index=0, id=None)
    tc2.function = MagicMock(name=None, arguments='and": "ls"}')
    acc.feed([tc1])
    acc.feed([tc2])
    assembled = acc.assemble()
    assert len(assembled) == 1
    assert assembled[0]["input"] == {"command": "ls"}
    print("✅ Tool call accumulator OK")

    # ── Malformed args ──
    acc = ToolCallAccumulator()
    tc = MagicMock(index=0, id="call_x")
    tc.function = MagicMock(name="X", arguments='{bad json')
    acc.feed([tc])
    assert acc.assemble()[0]["input"] == {}
    print("✅ Malformed args handled OK")

    # ── Streaming text ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.return_value = stream_yields([
        mock_chunk(text="Hello "),
        mock_chunk(text="world"),
        mock_chunk(finish="stop", usage=mock_usage()),
    ])
    events = []
    async for e in stream([create_user_message("hi")], enable_thinking=False):
        events.append(e)
    text_events = [e for e in events if e["type"] == "text_delta"]
    assert len(text_events) == 2
    stop = next(e for e in events if e["type"] == "message_stop")
    assert stop["stop_reason"] == "end_turn"
    print("✅ Streaming text OK")

    # ── Streaming tool call ──
    mock_litellm.acompletion.reset_mock()
    tc1 = MagicMock(index=0, id="call_x")
    tc1.function = MagicMock(name="Bash", arguments='{"command"')
    tc2 = MagicMock(index=0, id=None)
    tc2.function = MagicMock(name=None, arguments=': "ls"}')
    mock_litellm.acompletion.return_value = stream_yields([
        mock_chunk(tool_calls=[tc1]),
        mock_chunk(tool_calls=[tc2]),
        mock_chunk(finish="tool_calls", usage=mock_usage()),
    ])
    events = []
    async for e in stream([create_user_message("list files")], enable_thinking=False):
        events.append(e)
    tool_events = [e for e in events if e["type"] == "tool_use"]
    assert len(tool_events) == 1
    assert tool_events[0]["input"] == {"command": "ls"}
    stop = next(e for e in events if e["type"] == "message_stop")
    assert stop["stop_reason"] == "tool_use"
    print("✅ Streaming tool call OK")

    # ── Streaming thinking ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.return_value = stream_yields([
        mock_chunk(reasoning="let me think..."),
        mock_chunk(reasoning="done"),
        mock_chunk(text="Answer", finish="stop"),
    ])
    events = []
    async for e in stream([create_user_message("think")], enable_thinking=True):
        events.append(e)
    thinking_events = [e for e in events if e["type"] == "thinking_delta"]
    assert len(thinking_events) == 2
    print("✅ Streaming thinking OK")

    # ── Fallback on rate limit ──
    mock_litellm.acompletion.reset_mock()
    rate_limit = mock_litellm.RateLimitError("rate limited")
    mock_litellm.acompletion.side_effect = [
        rate_limit,
        stream_yields([mock_chunk(text="from fallback", finish="stop")]),
    ]
    events = []
    with patch.dict(os.environ, {
        "LITELLM_FALLBACK_MODELS": "groq/test-approved-fallback",
    }):
        async for e in stream([create_user_message("hi")], enable_thinking=False):
            events.append(e)
    text_events = [e for e in events if e["type"] == "text_delta"]
    assert text_events[0]["text"] == "from fallback"
    print("✅ Fallback on rate limit OK")

    # ── Permanent model errors do not trigger fallback attempts ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = MockBadRequestError("unknown model")
    with patch.dict(os.environ, {
        "LITELLM_FALLBACK_MODELS": "groq/test-approved-fallback",
    }):
        try:
            async for _ in stream([create_user_message("hi")], enable_thinking=False):
                pass
            raise AssertionError("expected the model error to propagate")
        except MockBadRequestError:
            pass
    assert mock_litellm.acompletion.call_count == 1
    print("✅ Permanent errors do not fall back OK")

    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-only"}, clear=True):
        _sync_provider_credentials()
        assert os.environ["NVIDIA_NIM_API_KEY"] == "test-only"
        assert "nvidia_nim/nvidia/nemotron-3.5-bolt-e4b" in list_models()
    print("✅ NVIDIA key alias OK")

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only"}, clear=True):
        models = list_models()
        assert "gemini/gemini-3.8-flash" in models
        assert not any("gemini-2.5" in model for model in models)
    print("✅ Current Gemini catalog OK")

    with patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}, clear=True):
        models = list_models()
        assert "groq/openai/gpt-oss-120b" in models
        assert "groq/openai/gpt-oss-20b" in models
        assert not any("llama-3." in model for model in models)
    print("✅ Current Groq catalog OK")

    # ── Non-streaming complete ──
        # ── Non-streaming complete ──
    mock_litellm.acompletion.reset_mock()
    mock_litellm.acompletion.side_effect = None        # ← clear exhausted list
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "Summary of conversation"
    mock_litellm.acompletion.return_value = resp
    result = await complete([create_user_message("summarize")])
    assert result == "Summary of conversation"
    print("✅ Non-streaming complete OK")

    # ── Model switching ──
    original = get_active_model()
    set_active_model("gemini/gemini-2.5-flash")
    assert get_active_model() == "gemini/gemini-2.5-flash"
    set_active_model("nvidia/nemotron-3.5-bolt-e4b")
    assert get_active_model() == "nvidia_nim/nvidia/nemotron-3.5-bolt-e4b"
    set_active_model(original)
    print("✅ Model switching OK")

    print(f"\n🎉 All api.py tests passed ({len(FALLBACK_MODELS)} fallbacks)")


asyncio.run(main())
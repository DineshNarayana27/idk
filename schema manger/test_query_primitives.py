"""test_query_primitives.py — verifies query_config, query_transitions, query_deps."""
import asyncio
import os

from query_config import build_query_config, QueryConfig, QueryGates
from query_transitions import Continue, Terminal
from query_deps import production_deps, NoopDeps, QueryDeps


async def main():
    # ── QueryConfig gates ──
    os.environ.pop("CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION", None)
    os.environ.pop("CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES", None)
    os.environ.pop("USER_TYPE", None)
    cfg = build_query_config(session_id="test-1")
    assert cfg.session_id == "test-1"
    assert cfg.gates.streaming_tool_execution is True       # default ON
    assert cfg.gates.emit_tool_use_summaries is False       # default OFF
    assert cfg.gates.is_ant is False
    assert cfg.gates.fast_mode_enabled is True

    os.environ["CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION"] = "1"
    os.environ["CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES"] = "1"
    os.environ["USER_TYPE"] = "ant"
    cfg2 = build_query_config()
    assert cfg2.gates.streaming_tool_execution is False
    assert cfg2.gates.emit_tool_use_summaries is True
    assert cfg2.gates.is_ant is True
    for k in ("CLAUDE_CODE_DISABLE_STREAMING_TOOL_EXECUTION",
              "CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES", "USER_TYPE"):
        os.environ.pop(k, None)
    print("✅ QueryConfig gates OK")

    # ── Continue: all 7 reasons construct ──
    for reason in ("next_turn", "collapse_drain_retry", "reactive_compact_retry",
                   "max_output_tokens_escalate", "max_output_tokens_recovery",
                   "token_budget_continuation", "stop_hook_blocking"):
        c = Continue(reason=reason)
        assert c.reason == reason
    c = Continue(reason="collapse_drain_retry", committed=3)
    assert c.committed == 3
    c = Continue(reason="max_output_tokens_recovery", attempt=2)
    assert c.attempt == 2
    print("✅ Continue (7 reasons) OK")

    # ── Terminal: all 10 reasons construct ──
    for reason in ("completed", "aborted_streaming", "aborted_tools",
                   "blocking_limit", "prompt_too_long", "image_error",
                   "model_error", "max_turns", "hook_stopped",
                   "stop_hook_prevented"):
        t = Terminal(reason=reason)
        assert t.reason == reason
    t = Terminal(reason="model_error", error=RuntimeError("boom"))
    assert isinstance(t.error, RuntimeError)
    t = Terminal(reason="max_turns", turn_count=5)
    assert t.turn_count == 5
    print("✅ Terminal (10 reasons) OK")

    # ── production_deps shape ──
    deps = production_deps()
    assert callable(deps.call_model)
    assert callable(deps.microcompact)
    assert callable(deps.autocompact)
    assert callable(deps.uuid)
    assert deps.uuid() != deps.uuid()
    print("✅ production_deps OK")

    # ── NoopDeps yields scripted responses ──
    from messages import create_assistant_message
    r1 = create_assistant_message(content="hello")
    r2 = create_assistant_message(content="world")
    nd = NoopDeps(responses=[r1, r2])
    first = [m async for m in nd.call_model()]
    second = [m async for m in nd.call_model()]
    assert len(first) == len(second) == 1
    assert first[0].content == "hello"
    assert second[0].content == "world"
    assert nd.uuid() == "uuid-1"
    assert nd.uuid() == "uuid-2"
    print("✅ NoopDeps OK")

    # ── microcompact / autocompact pass-through ──
    out = await nd.microcompact([], None, "test")
    assert out["messages"] == []
    out = await nd.autocompact([], None, {}, "test", None, 0)
    assert out["compaction_result"] is None
    assert out["consecutive_failures"] is None
    print("✅ NoopDeps compaction passthrough OK")

    print("\n🎉 All query primitives tests passed")


asyncio.run(main())
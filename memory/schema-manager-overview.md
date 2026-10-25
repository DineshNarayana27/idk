# Schema Manager Overview

## What it is
A Python coding-agent runtime, organized as a port of Claude Code-style components. The core loop is provider-agnostic through LiteLLM; messages are converted to OpenAI chat format at the API boundary.

## Request flow
1. `query.py` maintains the conversation state, shapes context, streams model output, and decides whether to finish, retry, compact, or continue after tools.
2. Model tool calls go through `StreamingToolExecutor` (the default) or `tool_orchestration.py`. Safe tools may run concurrently; results are returned in request order.
3. `execute.py` validates the tool call, checks permissions, asks for approval when needed, executes it, and builds a model-facing result.
4. `permissions.py` applies hard-deny rules, configured rules, tool-specific safety checks, permission modes, and auto-mode classification.

## Main building blocks
- `tool.py`, `registry.py`, `tools/`: shared tool contract, lookup registry, and filesystem/Bash/todo tools.
- `messages.py`, `api.py`: internal message types and provider streaming/normalization.
- `state.py`: per-session in-memory state; `store.py`: separate append-only SQLite history, searchable summaries, and persisted plans.
- `compact.py`, `query_token_budget.py`: token estimates, compaction, and continuation controls.
- `system_context.py`, `memdir.py`, `plans.py`, `todo.py`: prompt context, selectable file memories, an isolated Markdown plan module, and task lists.
- `path_safety.py`, `permission_rules.py`, `denial_tracking.py`, `loop_guard.py`: safety and repeated-action controls.

## Maturity notes
The core query, API, tool execution, permission, file-tool, and persistence paths have implementations and focused tests. Some integrations are intentionally incomplete: stop hooks are no-ops; several overflow/context-collapse functions in `compact.py` are stubs; some filesystem/sandbox checks and tool capabilities are deferred. `test_smoke.py` exercises the wiring with a fake model dependency, not a live provider. The tool registry is explicit, so tool definitions must be registered by the application setup.

The initial architecture review excluded `main.py`; it was inspected and updated during the implementation phase below.

## Implementation Roadmap

The persistence checks passed before implementation: `test_todo.py` and `test_store.py`.

### Completed in this phase
- Hardened `memdir.py`: absolute/local path validation, project-setting override rejection, symlink-escape protection, project-root keyed caching, remote memory directory support, one-day freshness warnings, and SHA-256 no-op writes.
- Added per-turn context assembly for the persisted plan, summary, and up to five relevant file memories. Memory selection is deduplicated across a session and routed through the query's existing single system-reminder injection.
- Added read-only `MemorySearch` and `MemoryExpand` tools for FTS5 search and exact DAG expansion.
- Connected CLI registration and SQLite persistence. Query messages are archived before context shaping; compaction summaries become summary-DAG nodes, and expansion retains chronological order.
- Made microcompacted tool results point to their archived message IDs, so `MemoryExpand` can recover them. Preserved loop-guard state through streaming tool contexts and registered memory reads as safe/idempotent.
- Synchronized the compacted active transcript back to the CLI's message history and connected `CLAUDE_CODE_MAX_TURN_TOKENS` to the existing diminishing-returns guard.
- Fixed two verified execution defects encountered in regression tests: Bash path safety was evaluating whole command strings as paths, and `execute.py` discarded the tool mapper's `is_error` result.

### Remaining before calling the context-window phase complete
1. `Store.get_active_view_messages()` now supplies the query after a summary exists, using a 40K default budget and keeping recent user/tool groups intact. Its estimate is character-based, and one indivisible newest group can exceed the budget; validate with the provider tokenizer before claiming the ~29K target.
2. Before the first persisted summary exists, the full transcript still relies on the existing auto-compaction trigger. Reconcile the cited 87% figure with the repository's current 167K threshold before changing it.
3. Load user-level memory settings in CLI setup. Project-level memory-path settings must remain rejected; the resolver supports an explicit user setting but the CLI does not load one yet.

Team sync, AST indexing, sandboxing, HRR, and automatic memory extraction remain separate deferred projects. The complete offline regression suite passed after implementation; persistence, query compaction, memory injection, memory tools, active views, and recovery pointers also have focused tests.

## Next Session Notes
- `plans.py` is now correctly named but remains unused: no imports or runtime call sites. Keep it deferred until a real Markdown Plan Mode lifecycle (enter/exit/resume/display) is implemented; do not merge it into the active SQLite `TodoWrite` plan path. Remove it only if Plan Mode is abandoned.
- Provider config in `api.py`: short `nvidia/...` names normalize to `nvidia_nim/nvidia/...`; legacy `NVIDIA_API_KEY` maps to LiteLLM's `NVIDIA_NIM_API_KEY`; Gemini catalog uses `gemini-3.8-flash`; cross-provider fallbacks are opt-in through `LITELLM_FALLBACK_MODELS` and only attempted after rate limits.
- API tests and smoke tests passed after those provider changes. The most recent `python main.py -p "hi"` terminal status was exit code 0; inspect the actual response if live provider behavior needs confirmation.
- Provider credentials were exposed in a chat attachment. Rotate those keys; never copy their values into this note or another chat.
- The full offline suite passed after the context adapter and provider changes. No commit was made; this workspace note is intended to be included if the user commits the work.

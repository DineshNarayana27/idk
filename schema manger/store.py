"""
store.py — append-only message log + summary DAG + plan persistence.

The persistent memory layer. Three tables and one invariant:

  messages  — APPEND ONLY. No update. No delete.
  summaries — summary DAG. Each node points to its sources via
              source_msg_ids (leaf nodes) or source_summary_ids (parent nodes).
  plan      — objective + todo list, disk-persisted across sessions.

Views are PROJECTED from the log, never modify it. This is the difference
between this design and Claude Code's destructive compaction: nothing is
ever cleared, only hidden from the current view. Every hidden message is
recoverable via expand() or search().

Usage:
    store = Store("~/.agent/memory.db")
    mid = store.append_message(sid, "user", "hello")
    sid_summary = store.append_summary(sid, 0, "User said hello",
                                        source_msg_ids=[mid])
    view = store.get_view(sid, budget_tokens=40_000)
    store.save_plan(sid, "do X", [{"content": "step 1", "status": "pending"}])
"""
from __future__ import annotations

import json
import os
import sqlite3
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_calls  TEXT,
    tool_use_id TEXT,
    timestamp   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS summaries (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL,
    depth              INTEGER NOT NULL,
    content            TEXT NOT NULL,
    token_count        INTEGER NOT NULL,
    source_msg_ids     TEXT,
    source_summary_ids TEXT,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, depth);

CREATE TABLE IF NOT EXISTS plan (
    session_id  TEXT PRIMARY KEY,
    objective   TEXT NOT NULL,
    items       TEXT NOT NULL,
    focus       TEXT,
    updated_at  REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id'
);
"""


class Store:
    def __init__(self, path: str):
        expanded = os.path.expanduser(path)
        parent = os.path.dirname(expanded)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = sqlite3.connect(expanded, check_same_thread=False)
        try:
            os.chmod(expanded, 0o600)
        except OSError:
            pass
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        # Keep FTS5 synced with the message log
        self.db.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ai
            AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
        """)
        self.db.commit()

    # ── Messages (append-only) ─────────────────────────────────────────

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list | None = None,
        tool_use_id: str | None = None,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, "
            "tool_use_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id, role, content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_use_id, time.time(),
            ),
        )
        self.db.commit()
        return cur.lastrowid

    def get_message(self, msg_id: int) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM messages WHERE id = ?", (msg_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_messages(self, session_id: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0]

    def get_message_id_by_tool_use_id(
        self, session_id: str, tool_use_id: str,
    ) -> int | None:
        row = self.db.execute(
            "SELECT id FROM messages WHERE session_id = ? AND tool_use_id = ? "
            "AND role = 'tool' ORDER BY id DESC LIMIT 1",
            (session_id, tool_use_id),
        ).fetchone()
        return int(row[0]) if row else None

    # ── Summaries (DAG) ────────────────────────────────────────────────

    def append_summary(
        self,
        session_id: str,
        depth: int,
        content: str,
        source_msg_ids: list[int] | None = None,
        source_summary_ids: list[int] | None = None,
    ) -> int:
        if not source_msg_ids and not source_summary_ids:
            raise ValueError(
                "Summary must have at least one source "
                "(source_msg_ids or source_summary_ids)"
            )
        tokens = (len(content) + 3) // 4
        cur = self.db.execute(
            "INSERT INTO summaries (session_id, depth, content, token_count, "
            "source_msg_ids, source_summary_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, depth, content, tokens,
                json.dumps(source_msg_ids) if source_msg_ids else None,
                json.dumps(source_summary_ids) if source_summary_ids else None,
                time.time(),
            ),
        )
        self.db.commit()
        return cur.lastrowid

    def get_summary(self, summary_id: int) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_root_summary(self, session_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM summaries WHERE session_id = ? "
            "ORDER BY depth DESC, created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def expand(
        self,
        node_id: int,
        kind: str = "summary",
        session_id: str | None = None,
    ) -> list[dict]:
        """Recursively expand a summary node down to leaf messages.
        This is the 'decode' path: any summary is reversible to the exact
        originals, byte-for-byte. When supplied, session_id confines reads
        to that session's memory."""
        if kind == "message":
            m = self.get_message(node_id)
            if session_id is not None and m and m["session_id"] != session_id:
                return []
            return [m] if m else []

        s = self.get_summary(node_id)
        if not s or (session_id is not None and s["session_id"] != session_id):
            return []

        out: list[dict] = []
        if s.get("source_summary_ids"):
            for sid in json.loads(s["source_summary_ids"]):
                out.extend(self.expand(sid, kind="summary", session_id=session_id))
        if s.get("source_msg_ids"):
            for mid in json.loads(s["source_msg_ids"]):
                m = self.get_message(mid)
                if m and (session_id is None or m["session_id"] == session_id):
                    out.append(m)
        return out

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self, query: str, session_id: str, limit: int = 20,
    ) -> list[dict]:
        rows = self.db.execute(
            "SELECT m.* FROM messages_fts f "
            "JOIN messages m ON m.id = f.rowid "
            "WHERE f.content MATCH ? AND m.session_id = ? "
            "ORDER BY m.id DESC LIMIT ?",
            (query, session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Plan (persists across sessions) ────────────────────────────────

    def save_plan(
        self,
        session_id: str,
        objective: str,
        items: list[dict],
        focus: str | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO plan (session_id, objective, items, focus, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "objective = excluded.objective, "
            "items = excluded.items, "
            "focus = excluded.focus, "
            "updated_at = excluded.updated_at",
            (session_id, objective, json.dumps(items), focus, time.time()),
        )
        self.db.commit()

    def get_plan(self, session_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM plan WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["items"] = json.loads(d["items"])
        return d

    # ── View assembly (the "projection") ───────────────────────────────

    def get_view(
        self,
        session_id: str,
        budget_tokens: int = 40_000,
        fresh_tail: int = 64,
    ) -> list[dict]:
        """Assemble the active view for the next API call.

        Returns a list of view entries in conversation order:
          1. Plan (if exists) as a system_plan entry
          2. Root summary (if exists) as a system_summary entry
          3. Fresh tail (last N messages verbatim)

        Messages older than the fresh tail are NOT included — but they
        remain in the store and are recoverable via expand() or search().
        """
        view: list[dict] = []

        plan = self.get_plan(session_id)
        if plan:
            view.append({
                "type": "system_plan",
                "content": self._format_plan(plan),
            })

        root = self.get_root_summary(session_id)
        if root:
            view.append({
                "type": "system_summary",
                "summary_id": root["id"],
                "depth": root["depth"],
                "content": root["content"],
            })

        tail = self.db.execute(
            "SELECT * FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, fresh_tail),
        ).fetchall()
        for row in reversed(tail):
            view.append({
                "type": "message",
                "role": row["role"],
                "content": row["content"],
                "tool_calls": (
                    json.loads(row["tool_calls"])
                    if row["tool_calls"] else None
                ),
                "tool_use_id": row["tool_use_id"],
                "id": row["id"],
            })

        return view

    def get_active_view_messages(
        self,
        session_id: str,
        budget_tokens: int = 40_000,
        fresh_tail: int = 64,
    ) -> list | None:
        """Project a summary plus a bounded, recent transcript into messages.

        Returns None until a summary exists, leaving initial history handling
        to query.py's normal automatic compaction. Summary sources remain in
        the append-only log and can still be expanded independently.
        """
        from messages import (
            AssistantMessage, SystemMessage, ToolResultMessage, ToolUseBlock,
            UserMessage,
        )

        root = self.get_root_summary(session_id)
        if root is None:
            return None

        budget = max(1, int(budget_tokens))
        summary_content = root["content"]
        summary_budget = max(0, budget - 12)
        summary_limit = summary_budget * 4
        if len(summary_content) > summary_limit:
            summary_content = summary_content[:summary_limit] + "\n[summary clipped in active view]"
        boundary = SystemMessage(subtype="compact_boundary", content="Conversation compacted")
        summary = UserMessage(content=summary_content, is_compact_summary=True)
        used_tokens = 12 + (len(summary_content) + 3) // 4
        remaining = max(0, budget - used_tokens)

        covered_ids = self._summary_source_message_ids(root["id"])
        last_covered_id = max(covered_ids, default=0)
        rows = self.db.execute(
            "SELECT * FROM messages WHERE session_id = ? AND id > ? "
            "ORDER BY id ASC",
            (session_id, last_covered_id),
        ).fetchall()

        groups: list[list[sqlite3.Row]] = []
        current: list[sqlite3.Row] = []
        for row in rows:
            if row["role"] == "user" and current:
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)

        selected_groups: list[list[sqlite3.Row]] = []
        selected_count = 0
        for group in reversed(groups):
            if fresh_tail > 0 and selected_count >= fresh_tail:
                break
            group_tokens = sum(
                max(1, (len(row["content"]) + 3) // 4)
                + ((len(row["tool_calls"] or "") + 3) // 4)
                for row in group
            )
            if selected_groups and group_tokens > remaining:
                break
            selected_groups.append(group)
            selected_count += len(group)
            remaining = max(0, remaining - group_tokens)

        recent_rows = [row for group in reversed(selected_groups) for row in group]
        active_messages: list = [boundary, summary]
        for row in recent_rows:
            if row["role"] == "user":
                message = UserMessage(content=row["content"])
            elif row["role"] == "assistant":
                calls = json.loads(row["tool_calls"]) if row["tool_calls"] else []
                message = AssistantMessage(
                    content=row["content"] or None,
                    tool_calls=[ToolUseBlock(
                        id=call["id"], name=call["name"], input=call["input"],
                    ) for call in calls],
                )
            elif row["role"] == "tool":
                message = ToolResultMessage(
                    tool_use_id=row["tool_use_id"] or "",
                    content=row["content"],
                )
            else:
                continue
            message._store_message_id = row["id"]
            active_messages.append(message)
        return active_messages

    def _summary_source_message_ids(self, summary_id: int) -> set[int]:
        summary = self.get_summary(summary_id)
        if summary is None:
            return set()
        message_ids = set(json.loads(summary["source_msg_ids"] or "[]"))
        for parent_id in json.loads(summary["source_summary_ids"] or "[]"):
            message_ids.update(self._summary_source_message_ids(parent_id))
        return message_ids

    def _format_plan(self, plan: dict) -> str:
        lines = [f"## Current Plan — {plan['objective']}"]
        in_progress = [
            i for i in plan["items"] if i.get("status") == "in_progress"
        ]
        done = [i for i in plan["items"] if i.get("status") == "completed"]
        pending = [i for i in plan["items"] if i.get("status") == "pending"]

        for item in done:
            lines.append(f"[DONE] {item['content']}")
        for item in in_progress:
            lines.append(f"[IN_PROGRESS] {item['content']}")
        for item in pending:
            lines.append(f"[PENDING] {item['content']}")

        if plan.get("focus"):
            lines.append(f"\nCurrent focus: {plan['focus']}")

        total = len(plan["items"])
        done_count = len(done)
        lines.append(f"\n({done_count}/{total} completed)")
        return "\n".join(lines)

    def close(self) -> None:
        self.db.close()
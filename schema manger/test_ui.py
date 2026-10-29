"""Headless smoke tests for the Textual agent UI."""
import asyncio
import os

os.environ["AGENT_STORE_PATH"] = ":memory:"

from textual.widgets import Header, Input

from messages import ProgressMessage, ToolResultMessage
from registry import find
from ui import (
    AgentApp, PlanPanel, StatusBar, StreamingMarkdown, SystemNotice,
    ToolCard, UserMessage,
)


async def main():
    app = AgentApp()
    async with app.run_test(size=(110, 32)) as pilot:
        assert app.query_one(Header)
        composer = app.query_one("#composer", Input)
        assert app.query_one(StatusBar)
        assert app.query_one(PlanPanel)
        assert find("Read") is not None
        assert find("Write") is not None
        assert find("Bash") is not None
        assert app.ctx._store.count_messages(app.ctx.session_id) == 0
        app._append_widget(UserMessage("hello from user"))
        streaming = StreamingMarkdown()
        app._append_widget(streaming)
        streaming.append_text("assistant response")
        streaming.finish()
        await pilot.pause()
        assert streaming._buffer == "assistant response"
        composer.value = "/tools"
        await pilot.press("enter")
        await pilot.pause()
        assert any("Write" in notice._text for notice in app.query(SystemNotice))

        card = ToolCard("tool-1", "Bash", "echo test")
        app._tool_cards["tool-1"] = card
        app._append_widget(card)
        await pilot.pause()
        await app._dispatch_event(ProgressMessage(
            tool_use_id="tool-1",
            parent_tool_use_id="",
            data={"type": "bash_progress", "output": "still running"},
        ))
        assert card._progress_lines == ["still running"]
        await app._dispatch_event(ToolResultMessage(
            tool_use_id="tool-1", content="done", is_error=False,
        ))
        assert "DONE" in card.title

        app.ctx._store.save_plan(app.ctx.session_id, "UI test", [
            {"content": "verify rendering", "status": "in_progress"},
        ], "verify rendering")
        app._refresh_plan()
        assert app.query_one(PlanPanel).title == "PLAN  0/1"
        await pilot.pause(0.1)
    print("Textual UI mounts with registered tools and session store")


if __name__ == "__main__":
    asyncio.run(main())

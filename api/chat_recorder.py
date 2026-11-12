"""Saves a chat while its AG-UI events stream to the browser.

The recorder sees every event the agent sends and keeps the database in step with what the user
sees: the user's message, each finished assistant message, errors, and the dashboard state
(rebuilt from STATE_SNAPSHOT / STATE_DELTA events). A reopened chat therefore looks exactly as
it did, even after a server restart.
"""
import copy
import uuid
from typing import Any, Dict, List

from ag_ui.core import BaseEvent, EventType, RunAgentInput

from persistence.store import RunStore, mark_unfinished


def apply_patch(document: Any, operations: List[Dict[str, Any]]) -> None:
    """Applies RFC 6902 JSON Patch operations (add / replace / remove) in place."""
    for op in operations:
        if not isinstance(op, dict):  # the AG-UI SDK turns patch operations into Pydantic models
            op = op.model_dump()
        parts = [p.replace("~1", "/").replace("~0", "~") for p in op["path"].split("/")[1:]]
        last = parts.pop()
        target = document
        for part in parts:
            target = target[int(part)] if isinstance(target, list) else target[part]
        if op["op"] in ("add", "replace"):
            if isinstance(target, list):
                if last == "-":
                    target.append(op["value"])
                elif op["op"] == "add":
                    target.insert(int(last), op["value"])
                else:
                    target[int(last)] = op["value"]
            else:
                target[last] = op["value"]
        elif op["op"] == "remove":
            if isinstance(target, list):
                target.pop(int(last))
            else:
                target.pop(last, None)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(getattr(part, "text", "") for part in content)
    return content


class ChatRecorder:
    """Persists one AG-UI run of a chat. Call `start`, then `record(event)` for each event, then `close`."""

    def __init__(self, store: RunStore, run_input: RunAgentInput) -> None:
        self.store = store
        self.chat_id = run_input.thread_id
        self.run_input = run_input
        self.state: Dict[str, Any] = copy.deepcopy(run_input.state) if isinstance(run_input.state, dict) else {}
        self.texts: Dict[str, str] = {}
        self.ended = False  # RUN_FINISHED or RUN_ERROR seen

    def start(self) -> None:
        user_messages = [m for m in self.run_input.messages if m.role == "user"]
        first = _message_text(user_messages[0]) if user_messages else "New chat"
        self.store.ensure_chat(self.chat_id, first)
        if user_messages:
            last = user_messages[-1]
            self.store.add_message(self.chat_id, last.id, "user", _message_text(last))

    def record(self, event: BaseEvent) -> None:
        kind = event.type
        if kind == EventType.STATE_SNAPSHOT:
            self.state = copy.deepcopy(event.snapshot) or {}
            self.store.save_state(self.chat_id, self.state)
        elif kind == EventType.STATE_DELTA:
            apply_patch(self.state, event.delta)
            self.store.save_state(self.chat_id, self.state)
        elif kind == EventType.TEXT_MESSAGE_START:
            self.texts[event.message_id] = ""
        elif kind == EventType.TEXT_MESSAGE_CONTENT:
            self.texts[event.message_id] = self.texts.get(event.message_id, "") + event.delta
        elif kind == EventType.TEXT_MESSAGE_END:
            self.store.add_message(self.chat_id, event.message_id, "assistant", self.texts.pop(event.message_id, ""))
        elif kind == EventType.RUN_ERROR:
            self.ended = True
            self._finish_state("error")
            self.store.add_message(self.chat_id, str(uuid.uuid4()), "error", event.message)
        elif kind == EventType.RUN_FINISHED:
            self.ended = True

    def close(self) -> None:
        """Called when the stream ends. If it ended early (Stop, closed tab), record that."""
        if self.ended:
            return
        for message_id, text in self.texts.items():  # keep half-streamed replies
            if text:
                self.store.add_message(self.chat_id, message_id, "assistant", text)
        self._finish_state("stopped")
        self.store.add_message(self.chat_id, str(uuid.uuid4()), "notice", "Stopped.")

    def _finish_state(self, status: str) -> None:
        self.state = mark_unfinished(self.state, status)
        self.store.save_state(self.chat_id, self.state)

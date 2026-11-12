"""SQLite database for chats, runs and every agent's stage output.

Tables
    chats          one row per conversation (id = the AG-UI thread id)
    messages       chat messages in order (user / assistant / error / notice)
    chat_state     the dashboard state of each chat (JSON), so a reopened chat shows its results
    runs           one row per analysis or backtest: parameters, status, final result
    stage_results  the output of every agent / stage of every run, exactly as the agent returned it

Survives server restarts. Runs still marked "running" at start-up were interrupted by the
restart and are marked "interrupted" (analyses can then be resumed from their LangGraph checkpoint).
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, seq);
CREATE TABLE IF NOT EXISTS chat_state (
    chat_id     TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    state_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    chat_id      TEXT REFERENCES chats(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,              -- analysis | backtest
    status       TEXT NOT NULL,              -- running | done | error | cancelled | interrupted
    params_json  TEXT NOT NULL,
    result_json  TEXT,
    error        TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_chat ON runs(chat_id, started_at);
CREATE TABLE IF NOT EXISTS stage_results (
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,              -- e.g. technical_analyst, trader, risk_manager, finalize
    item_key     TEXT NOT NULL DEFAULT '',   -- e.g. the decision date inside a backtest
    status       TEXT NOT NULL,
    output_json  TEXT,
    recorded_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, item_key)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


UNFINISHED_STATUSES = ("running", "pending")


def mark_unfinished(state: Any, new_status: str) -> Any:
    """Replaces every running / pending `"status"` inside a dashboard state with `new_status`."""
    if isinstance(state, dict):
        return {
            key: (
                new_status
                if key == "status" and value in UNFINISHED_STATUSES
                else mark_unfinished(value, new_status)
            )
            for key, value in state.items()
        }
    if isinstance(state, list):
        return [mark_unfinished(item, new_status) for item in state]
    return state


class RunStore:
    """Thread-safe access to the database (agent runs execute in worker threads)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            self._conn.commit()
            return rows

    # ----- chats and messages ----------------------------------------------------
    def ensure_chat(self, chat_id: str, title: str) -> None:
        now = _now()
        self._execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at",
            (chat_id, title.strip()[:80] or "New chat", now, now),
        )

    def add_message(self, chat_id: str, message_id: str, role: str, content: str) -> None:
        """Inserts a message once (ids are unique; re-sent history is ignored)."""
        with self._lock:
            next_seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE chat_id = ?", (chat_id,)
            ).fetchone()[0]
            self._conn.execute(
                "INSERT OR IGNORE INTO messages (id, chat_id, seq, role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, chat_id, next_seq, role, content, _now()),
            )
            self._conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))
            self._conn.commit()

    def save_state(self, chat_id: str, state: Dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO chat_state (chat_id, state_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at",
            (chat_id, _dumps(state), _now()),
        )

    def list_chats(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        chat = self._execute("SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,))
        if not chat:
            return None
        messages = self._execute(
            "SELECT id, role, content, created_at FROM messages WHERE chat_id = ? ORDER BY seq", (chat_id,)
        )
        state = self._execute("SELECT state_json FROM chat_state WHERE chat_id = ?", (chat_id,))
        runs = self._execute(
            "SELECT id, kind, status, params_json, started_at, finished_at FROM runs WHERE chat_id = ? ORDER BY started_at",
            (chat_id,),
        )
        return {
            **dict(chat[0]),
            "messages": [dict(row) for row in messages],
            "state": json.loads(state[0]["state_json"]) if state else {},
            "runs": [{**dict(row), "params": json.loads(row["params_json"])} for row in runs],
        }

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            deleted = self._conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
            self._conn.commit()
        return deleted > 0

    # ----- runs and stage results ---------------------------------------------------
    def start_run(self, run_id: str, chat_id: Optional[str], kind: str, params: Dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO runs (id, chat_id, kind, status, params_json, started_at) VALUES (?, ?, ?, 'running', ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET status = 'running', finished_at = NULL, error = NULL",
            (run_id, chat_id, kind, _dumps(params), _now()),
        )

    def finish_run(self, run_id: str, status: str, result: Any = None, error: Optional[str] = None) -> None:
        self._execute(
            "UPDATE runs SET status = ?, result_json = COALESCE(?, result_json), error = ?, finished_at = ? WHERE id = ?",
            (status, _dumps(result) if result is not None else None, error, _now(), run_id),
        )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not rows:
            return None
        run = dict(rows[0])
        run["params"] = json.loads(run.pop("params_json"))
        run["result"] = json.loads(run.pop("result_json")) if run.get("result_json") else None
        return run

    def save_stage(self, run_id: str, stage: str, output: Any, status: str = "done", item_key: str = "") -> None:
        self._execute(
            "INSERT INTO stage_results (run_id, stage, item_key, status, output_json, recorded_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, stage, item_key) DO UPDATE SET status = excluded.status, "
            "output_json = excluded.output_json, recorded_at = excluded.recorded_at",
            (run_id, stage, item_key, status, _dumps(output), _now()),
        )

    def get_stages(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT stage, item_key, status, output_json, recorded_at FROM stage_results WHERE run_id = ? ORDER BY recorded_at",
            (run_id,),
        )
        return [
            {**{k: row[k] for k in ("stage", "item_key", "status", "recorded_at")}, "output": json.loads(row["output_json"])}
            for row in rows
        ]

    # ----- start-up recovery ----------------------------------------------------------
    def mark_interrupted(self) -> int:
        """Runs left 'running' by a previous server process were interrupted by its shutdown."""
        with self._lock:
            count = self._conn.execute(
                "UPDATE runs SET status = 'interrupted', finished_at = ? WHERE status = 'running'", (_now(),)
            ).rowcount
            for row in self._conn.execute("SELECT chat_id, state_json FROM chat_state").fetchall():
                state = json.loads(row["state_json"])
                updated = mark_unfinished(state, "interrupted")
                if updated != state:
                    self._conn.execute(
                        "UPDATE chat_state SET state_json = ? WHERE chat_id = ?", (_dumps(updated), row["chat_id"])
                    )
            self._conn.commit()
        return count

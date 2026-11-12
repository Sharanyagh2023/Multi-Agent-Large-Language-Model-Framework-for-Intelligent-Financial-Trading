"""LangGraph checkpointer stored in SQLite.

With a checkpointer, LangGraph saves the graph state after every step of a run under the run's
`thread_id`. If the process stops half-way, the same thread can be resumed later and only the
unfinished stages run again.

The state contains this project's Pydantic messages, so they are registered explicitly as safe
to load (LangGraph will refuse unregistered types in future versions).
"""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from core import schemas

CHECKPOINT_TYPES = [
    schemas.AnalystOpinion,
    schemas.AnalystReport,
    schemas.TradeProposal,
    schemas.RiskAssessment,
    schemas.FinalDecision,
]


def create_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[(cls.__module__, cls.__name__) for cls in CHECKPOINT_TYPES]
    )
    checkpointer = SqliteSaver(connection, serde=serializer)
    checkpointer.setup()
    return checkpointer

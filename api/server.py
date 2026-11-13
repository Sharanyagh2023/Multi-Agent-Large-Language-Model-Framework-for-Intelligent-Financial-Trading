"""Web server: the AG-UI endpoint, the chat-history API and the static frontend.

    POST   /agui                   AG-UI run (request: RunAgentInput JSON, response: Server-Sent Events)
    GET    /api/info               market and model names, shown in the UI header
    GET    /api/chats              saved chats, newest first
    GET    /api/chats/{id}         one chat: messages, dashboard state and its runs
    DELETE /api/chats/{id}         delete a chat with its runs and stage results
    GET    /api/runs/{id}          one run with the output of every stage
    GET    /                       the frontend (frontend/index.html, styles.css, app.js)

Everything is stored in SQLite (paths in config.yaml -> storage), so chats and results survive
a server restart. Start it with:  python main.py serve
"""
import asyncio
import logging
import threading
from pathlib import Path

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.agui_agent import CHAT_AGENT_NAME, RioAgent
from api.chat_recorder import ChatRecorder
from config.loader import AppConfig
from persistence.checkpoints import create_checkpointer
from persistence.store import RunStore

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DISCONNECT_POLL_SECONDS = 0.5
logger = logging.getLogger(__name__)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="RIO", description="Multi-agent LLM trading framework (AG-UI server)")
    paths = config.storage_paths
    store = RunStore(paths["database"])
    interrupted = store.mark_interrupted()
    if interrupted:
        logger.info("%d run(s) were interrupted by the last shutdown; analyses can be resumed", interrupted)
    agent = RioAgent(config, store=store, checkpointer=create_checkpointer(paths["checkpoints"]))
    logger.info("Results database: %s | checkpoints: %s", paths["database"], paths["checkpoints"])

    @app.post("/agui")
    async def run_agent(run_input: RunAgentInput, request: Request) -> StreamingResponse:
        encoder = EventEncoder(accept=request.headers.get("accept"))
        cancelled = threading.Event()  # set when the browser goes away; the agent then stops working
        recorder = ChatRecorder(store, run_input)
        recorder.start()

        async def watch_for_disconnect() -> None:
            # Safety net. Uvicorn/Starlette normally cancel this response as soon as the client leaves,
            # which also sets `cancelled` (see the finally below). If a server only reported the
            # disconnect on the next write, that could be a long wait during a backtest.
            while not cancelled.is_set():
                if await request.is_disconnected():
                    logger.info("Client disconnected from run %s: stopping it", run_input.run_id)
                    cancelled.set()
                    return
                await asyncio.sleep(DISCONNECT_POLL_SECONDS)

        async def event_stream():
            watcher = asyncio.create_task(watch_for_disconnect())
            try:
                async for event in agent.run(run_input, cancelled):
                    try:
                        recorder.record(event)
                    except Exception:  # saving history must never break a live run
                        logger.exception("Could not record %s for chat %s", event.type, run_input.thread_id)
                    if cancelled.is_set():
                        break
                    yield encoder.encode(event)
            finally:
                cancelled.set()
                watcher.cancel()
                recorder.close()

        return StreamingResponse(
            event_stream(),
            media_type=encoder.get_content_type(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/info")
    def info() -> dict:
        market = config.market
        return {
            "market": market["name"],
            "currency": market["currency"],
            "llm": config.llm_name_for_agent(CHAT_AGENT_NAME),
            "maxPositionPct": config.trading["max_position_pct"],
            "backtest": config.backtest,
        }

    @app.get("/api/chats")
    def list_chats() -> list:
        return store.list_chats()

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str) -> dict:
        chat = store.get_chat(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat

    @app.delete("/api/chats/{chat_id}")
    def delete_chat(chat_id: str) -> dict:
        if not store.delete_chat(chat_id):
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"deleted": chat_id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {**run, "stages": store.get_stages(run_id)}

    # Mounted last so it does not shadow the routes above.
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app

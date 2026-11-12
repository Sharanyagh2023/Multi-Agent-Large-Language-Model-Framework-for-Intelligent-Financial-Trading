"""AG-UI adapter: exposes the RIO workflow as a stream of AG-UI events.

AG-UI (https://docs.ag-ui.com) is an open, event-based protocol between an agent
backend and a user interface. One HTTP request = one "run"; the server answers
with Server-Sent Events such as:

    RUN_STARTED                      the run begins
    STATE_SNAPSHOT / STATE_DELTA     the shared dashboard state (left panel)
    STEP_STARTED / STEP_FINISHED     an agent stage begins / ends
    TEXT_MESSAGE_START/CONTENT/END   a chat reply, streamed token by token (right panel)
    RUN_FINISHED / RUN_ERROR         the run ends

The browser owns the conversation and the dashboard state and sends both back on
every request, so this server keeps no session data.

What a chat message can trigger (decided by a small "router" LLM call):
    analyze   -> run the full agent team for one stock and date
    backtest  -> walk-forward backtest over a period
    answer    -> reply in chat, using the current results as context
"""
import asyncio
import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional, Tuple

from ag_ui.core import (
    BaseEvent,
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph.checkpoint.base import BaseCheckpointSaver

from config.loader import AppConfig
from data.market_data import MarketDataService
from evaluation.backtest import Backtester
from persistence.store import RunStore
from workflow.trading_graph import TradingWorkflow

logger = logging.getLogger(__name__)

CHAT_AGENT_NAME = "chat_assistant"  # config.yaml -> agents key (falls back to default_llm)
HISTORY_MESSAGES = 12               # how much of the conversation the LLM sees


# ----- what the router LLM returns ---------------------------------------------
class ChatIntent(BaseModel):
    """How to handle the user's latest chat message."""

    action: Literal["analyze", "backtest", "answer"] = Field(
        description="analyze = trading decision for one stock; backtest = historical evaluation over a period; "
        "answer = anything else (questions, explanations, greetings, or a request without a stock)."
    )
    ticker: Optional[str] = Field(
        description="Stock symbol, e.g. RELIANCE, TCS, HDFCBANK or TCS.BO. Convert company names to their NSE "
        "symbol (Reliance Industries -> RELIANCE, Infosys -> INFY). Null if no stock is meant."
    )
    as_of_date: Optional[str] = Field(description="analyze only: decision date as YYYY-MM-DD, null for today.")
    start_date: Optional[str] = Field(description="backtest only: YYYY-MM-DD, null if not given.")
    end_date: Optional[str] = Field(description="backtest only: YYYY-MM-DD, null if not given.")
    rebalance_days: Optional[int] = Field(description="backtest only: trading days between decisions, null if not given.")


ROUTER_PROMPT = (
    "You route chat messages for RIO, a multi-agent trading assistant for {market_name}. Today is {today}.\n"
    "Read the conversation and classify ONLY the latest user message.\n"
    "- 'analyze' when the user wants a view/decision/analysis of a stock (\"analyze Infosys\", \"should I buy TCS?\").\n"
    "- 'backtest' when the user wants to test the strategy on past data over a period.\n"
    "- 'answer' for everything else, including questions about results already shown.\n"
    "Resolve follow-ups from context (\"same for TCS\" repeats the previous action for TCS). "
    "Convert relative dates (\"end of June 2025\", \"last month\") to YYYY-MM-DD. Leave unknown fields null."
)

ANSWER_PROMPT = (
    "You are the chat assistant of RIO, a multi-agent LLM trading framework for {market_name} "
    "(four analysts -> trader -> risk manager -> final decision). Today is {today}.\n"
    "The user sees a dashboard with the latest results. Its data (JSON):\n{dashboard}\n\n"
    "Sizes, confidences, returns, volatility, VaR and drawdown in the data are FRACTIONS: 0.08 means 8%.\n"
    "Answer concisely (short paragraphs or bullets, Markdown allowed). Use only the numbers in the dashboard data; "
    "do not invent figures. If the user wants a new analysis or backtest, tell them to ask e.g. "
    "\"Analyze RELIANCE\" or \"Backtest TCS for 2025\". This is an educational prototype, not financial advice."
)

EXPLAIN_PROMPT = (
    "You are RIO's chat assistant. The agent team just finished analysing a stock. Explain the outcome to the "
    "user in 3 to 5 short sentences of Markdown: the final action and size, the main reasons, where analysts "
    "disagreed, and what the risk manager changed. Sizes, confidences and risk metrics are fractions "
    "(0.08 means 8%). Use only this data:\n{result}"
)


class RunCancelled(Exception):
    """The browser stopped the run (Stop button, closed tab). Raised to unwind the worker thread."""


class StopBeforeLLMCall(BaseCallbackHandler):
    """LangChain callback that refuses to start any LLM call once the run is cancelled.

    Passed in the run config, it reaches every model call made inside the graph's nodes
    (LangChain propagates callbacks to child runs), so no agent sends a new request after Stop.
    """

    raise_error = True  # let the exception escape instead of being logged and ignored

    def __init__(self, cancelled: threading.Event) -> None:
        self.cancelled = cancelled

    def _check(self) -> None:
        if self.cancelled.is_set():
            raise RunCancelled()

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._check()

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._check()


def cancellable_config(cancelled: threading.Event) -> Dict[str, Any]:
    return {"callbacks": [StopBeforeLLMCall(cancelled)]}


class _Failure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


_DONE = object()
_CANCELLED = object()


async def _iterate_in_thread(
    work: Callable[[Callable[[Any], None]], None], cancelled: threading.Event
) -> AsyncIterator[Any]:
    """Runs blocking code (LangGraph, yfinance, LLM calls) in a thread and yields what it emits.

    `work(emit)` is called in a background thread; every `emit(item)` is delivered here in order.
    Cancellation is cooperative: once `cancelled` is set, the next `emit` (or the next LLM call,
    via StopBeforeLLMCall) raises RunCancelled inside the thread so it stops doing work.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def emit(item: Any) -> None:
        if cancelled.is_set():
            raise RunCancelled()
        put(item)

    def target() -> None:
        try:
            work(emit)
            put(_DONE)
        except RunCancelled:
            logger.info("Worker stopped: run cancelled by the client")
            put(_CANCELLED)
        except BaseException as error:  # forwarded to the async side
            put(_Failure(error))

    threading.Thread(target=target, daemon=True).start()
    finished = False
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                finished = True
                return
            if item is _CANCELLED:
                finished = True
                raise RunCancelled()
            if isinstance(item, _Failure):
                finished = True
                raise item.error
            yield item
    finally:
        if not finished:  # the consumer went away (client disconnected): stop the thread too
            cancelled.set()


def _json_safe(value: Any) -> Any:
    """Pydantic models -> dicts and NaN/inf -> None, so the browser can JSON.parse the events."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # numpy scalars
        return _json_safe(value.item())
    return value


def _replace(path: str, value: Any) -> Dict[str, Any]:
    return {"op": "replace", "path": path, "value": _json_safe(value)}


def _add(path: str, value: Any) -> Dict[str, Any]:
    return {"op": "add", "path": path, "value": _json_safe(value)}


def _valid_date(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


@dataclass
class RunContext:
    """Per-request bookkeeping: which chat, whether the browser left, which run is being recorded."""

    chat_id: str
    cancelled: threading.Event
    run_id: Optional[str] = None  # analysis / backtest run currently recorded in the database
    run_closed: bool = True


class RioAgent:
    """Turns one AG-UI `RunAgentInput` into a stream of AG-UI events.

    With a `store`, every analysis / backtest run and every stage output is saved to the database.
    With a `checkpointer`, LangGraph checkpoints each analysis after every stage, so an interrupted
    analysis can be resumed (AG-UI forwardedProps: {"command": "resume", "runId": "..."}).
    """

    def __init__(
        self,
        config: AppConfig,
        store: Optional[RunStore] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.checkpointer = checkpointer
        self._workflows: Dict[bool, TradingWorkflow] = {}
        self._lock = threading.Lock()

    # ----- shared helpers -------------------------------------------------------
    @property
    def market_name(self) -> str:
        return self.config.market["name"]

    def workflow(self, point_in_time: bool) -> TradingWorkflow:
        """Live analysis uses today's fundamentals/news and is checkpointed.

        Backtests (point_in_time=True) are not checkpointed: they consist of many short runs whose
        results are stored in the database instead.
        """
        with self._lock:
            if point_in_time not in self._workflows:
                market_data = MarketDataService(self.config.market, point_in_time=point_in_time)
                checkpointer = None if point_in_time else self.checkpointer
                self._workflows[point_in_time] = TradingWorkflow(self.config, market_data, checkpointer)
            return self._workflows[point_in_time]

    @property
    def chat_llm(self):
        return self.config.get_llm_for_agent(CHAT_AGENT_NAME)

    # ----- database bookkeeping -----------------------------------------------------
    def _start_run(self, ctx: RunContext, run_id: str, kind: str, params: Dict[str, Any]) -> None:
        ctx.run_id, ctx.run_closed = run_id, False
        if self.store:
            self.store.start_run(run_id, ctx.chat_id, kind, params)

    def _close_run(self, ctx: RunContext, status: str, result: Any = None, error: Optional[str] = None) -> None:
        if ctx.run_closed or ctx.run_id is None:
            return
        ctx.run_closed = True
        if self.store:
            self.store.finish_run(ctx.run_id, status, _json_safe(result), error)

    def _save_stage(self, ctx: RunContext, stage: str, output: Any, item_key: str = "") -> None:
        if self.store and ctx.run_id:
            self.store.save_stage(ctx.run_id, stage, _json_safe(output), item_key=item_key)

    @staticmethod
    def _history(run_input: RunAgentInput) -> List[BaseMessage]:
        history: List[BaseMessage] = []
        for message in run_input.messages[-HISTORY_MESSAGES:]:
            content = message.content
            if isinstance(content, list):  # multimodal user content: keep the text parts
                content = " ".join(getattr(part, "text", "") for part in content)
            if not content:
                continue
            if message.role == "user":
                history.append(HumanMessage(content=content))
            elif message.role == "assistant":
                history.append(AIMessage(content=content))
        return history

    async def _say(self, text: str) -> AsyncIterator[BaseEvent]:
        """A complete assistant chat message sent as one chunk."""
        message_id = str(uuid.uuid4())
        yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant")
        yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=text)
        yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)

    async def _stream_llm(self, messages: List[BaseMessage], cancelled: threading.Event) -> AsyncIterator[BaseEvent]:
        """An assistant chat message streamed token by token from the chat LLM."""
        message_id = str(uuid.uuid4())
        yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant")

        def work(emit: Callable[[Any], None]) -> None:
            # Leaving this loop early (emit raises RunCancelled) closes the HTTP stream to the provider.
            for chunk in self.chat_llm.stream(messages, config=cancellable_config(cancelled)):
                # `.text` works for every provider: OpenAI/Azure/Ollama stream plain strings, while
                # Anthropic streams a list of content blocks ([{"type": "text", "text": ...}]).
                if chunk.text:
                    emit(chunk.text)

        async for delta in _iterate_in_thread(work, cancelled):
            yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=delta)
        yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)

    # ----- entry point ----------------------------------------------------------
    async def run(
        self, run_input: RunAgentInput, cancelled: Optional[threading.Event] = None
    ) -> AsyncIterator[BaseEvent]:
        """`cancelled` is set by the server when the browser disconnects; all work then stops."""
        ctx = RunContext(chat_id=run_input.thread_id, cancelled=cancelled or threading.Event())
        thread_id, run_id = run_input.thread_id, run_input.run_id
        yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id)
        try:
            state = run_input.state if isinstance(run_input.state, dict) else {}
            forwarded = run_input.forwarded_props if isinstance(run_input.forwarded_props, dict) else {}
            history = self._history(run_input)

            if forwarded.get("command") == "resume":
                handler = self._resume(str(forwarded.get("runId", "")), state, ctx)
            elif not history or not isinstance(history[-1], HumanMessage):
                handler = self._say("Ask me to analyze a stock, e.g. **Analyze RELIANCE**.")
            else:
                intent = await asyncio.to_thread(self._route, history, ctx.cancelled)
                logger.info("Chat intent: %s", intent.model_dump())
                if intent.action == "analyze":
                    handler = self._analyze(state, ctx, intent=intent)
                elif intent.action == "backtest":
                    handler = self._backtest(intent, state, ctx)
                else:
                    handler = self._answer(history, state, ctx.cancelled)
            async for event in handler:
                yield event
            yield RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id)
        except RunCancelled:
            logger.info("Run %s cancelled by the client", run_id)  # nobody is listening any more
            self._close_run(ctx, "cancelled")
        except Exception as error:
            logger.exception("Run %s failed", run_id)
            message = f"{type(error).__name__}: {error}"
            self._close_run(ctx, "error", error=message)
            yield RunErrorEvent(type=EventType.RUN_ERROR, message=message)
        finally:
            self._close_run(ctx, "cancelled")  # no-op unless the stream was closed mid-run

    def _route(self, history: List[BaseMessage], cancelled: threading.Event) -> ChatIntent:
        system = ROUTER_PROMPT.format(market_name=self.market_name, today=date.today().isoformat())
        router = self.chat_llm.with_structured_output(ChatIntent)
        return router.invoke([SystemMessage(content=system), *history], config=cancellable_config(cancelled))

    # ----- answer: plain chat with the dashboard as context -----------------------
    async def _answer(
        self, history: List[BaseMessage], state: Dict[str, Any], cancelled: threading.Event
    ) -> AsyncIterator[BaseEvent]:
        dashboard = dict(state)
        if isinstance(dashboard.get("backtest"), dict):
            dashboard["backtest"] = {k: v for k, v in dashboard["backtest"].items() if k != "curve"}
        system = ANSWER_PROMPT.format(
            market_name=self.market_name,
            today=date.today().isoformat(),
            dashboard=json.dumps(dashboard, default=str)[:20000] if dashboard else "(no results yet)",
        )
        async for event in self._stream_llm([SystemMessage(content=system), *history], cancelled):
            yield event

    # ----- resume an interrupted analysis ------------------------------------------
    async def _resume(self, run_id: str, state: Dict[str, Any], ctx: RunContext) -> AsyncIterator[BaseEvent]:
        run = self.store.get_run(run_id) if self.store else None
        if not run or run["kind"] != "analysis" or run["chat_id"] != ctx.chat_id:
            async for event in self._say("I could not find that analysis to resume."):
                yield event
            return
        if self.checkpointer is None:
            async for event in self._say("Checkpointing is not enabled, so this analysis cannot be resumed."):
                yield event
            return
        async for event in self._analyze(state, ctx, resume_run=run):
            yield event

    # ----- analyze: one full run of the agent team -------------------------------
    @staticmethod
    def _stage_view(node: str, values: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        """What the dashboard shows for a finished stage, plus any extra state patches."""
        if "analyst_reports" in values:
            return values["analyst_reports"][0], []
        if "trade_proposal" in values:
            return values["trade_proposal"], []
        if "risk_assessment" in values:
            return values["risk_assessment"], [_replace("/analysis/riskMetrics", values.get("risk_metrics") or {})]
        if "final_decision" in values:
            return values["final_decision"], [_replace("/analysis/status", "done")]
        return values, []

    async def _analyze(
        self,
        state: Dict[str, Any],
        ctx: RunContext,
        intent: Optional[ChatIntent] = None,
        resume_run: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[BaseEvent]:
        workflow = self.workflow(point_in_time=False)
        analysts = [analyst.name for analyst in workflow.analysts]
        stage_names = [*analysts, workflow.trader.name, workflow.risk_manager.name, "finalize"]

        # Stages already finished (only when resuming), read back from the database.
        finished: Dict[str, Dict[str, Any]] = {}
        if resume_run:
            run_id = resume_run["id"]
            ticker, as_of_date = resume_run["params"]["ticker"], resume_run["params"]["asOfDate"]
            for row in self.store.get_stages(run_id):
                if row["stage"] in stage_names:
                    finished[row["stage"]] = row["output"]
        else:
            if not intent or not intent.ticker:
                async for event in self._say("Which stock should I analyze? For example: **Analyze INFY**."):
                    yield event
                return
            try:
                ticker = workflow.market_data.normalize_ticker(intent.ticker)
            except ValueError as error:
                async for event in self._say(f"{error}. Phase 1 covers NSE (`.NS`) and BSE (`.BO`) stocks only."):
                    yield event
                return
            today = date.today().isoformat()
            as_of_date = min(_valid_date(intent.as_of_date) or today, today)
            run_id = str(uuid.uuid4())

        self._start_run(ctx, run_id, "analysis", {"ticker": ticker, "asOfDate": as_of_date})

        stages: Dict[str, Any] = {name: {"status": "pending", "output": None} for name in stage_names}
        risk_metrics = None
        result: Dict[str, Any] = {}
        for node, values in finished.items():
            output, _ = self._stage_view(node, values)
            stages[node] = {"status": "done", "output": output}
            risk_metrics = values.get("risk_metrics", risk_metrics)
            for key, value in values.items():
                if key == "analyst_reports":
                    result.setdefault(key, []).extend(value)
                else:
                    result[key] = value
        done = set(finished)
        all_done = done.issuperset(stage_names)

        yield StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=_json_safe({
                **state,
                "view": "analysis",
                "analysis": {
                    "runId": run_id,
                    "ticker": ticker,
                    "asOfDate": as_of_date,
                    "status": "done" if all_done else "running",
                    "stages": stages,
                    "riskMetrics": risk_metrics,
                },
            }),
        )
        if resume_run:
            remaining = [name for name in stage_names if name not in done]
            message = (
                f"Resuming **{ticker}** (as of {as_of_date}) from its last checkpoint. "
                f"{len(done)} of {len(stage_names)} stages were already finished and are not run again."
                if remaining else f"The analysis of **{ticker}** had already finished."
            )
        else:
            message = f"Running the agent team on **{ticker}** as of **{as_of_date}**. Follow each stage in the left panel."
        async for event in self._say(message):
            yield event

        started: set = set(done)

        def start(stage: str) -> List[BaseEvent]:
            started.add(stage)
            return [
                StepStartedEvent(type=EventType.STEP_STARTED, step_name=stage),
                StateDeltaEvent(
                    type=EventType.STATE_DELTA, delta=[_replace(f"/analysis/stages/{stage}/status", "running")]
                ),
            ]

        def next_starts() -> List[BaseEvent]:
            """STEP_STARTED for the stage(s) the graph will run next."""
            pending_analysts = [a for a in analysts if a not in done]
            if pending_analysts:
                return [e for a in pending_analysts if a not in started for e in start(a)]
            for stage in stage_names[len(analysts):]:
                if stage not in done:
                    return start(stage) if stage not in started else []
            return []

        if not all_done:
            for event in next_starts():
                yield event

            def work(emit: Callable[[Any], None]) -> None:
                # Checkpointed under thread_id = run_id; a resumed run continues where it stopped.
                # The callback blocks new LLM calls after Stop; emit() stops the loop between nodes.
                for update in workflow.stream_updates(ticker, as_of_date, run_id, cancellable_config(ctx.cancelled)):
                    emit(update)

            async for update in _iterate_in_thread(work, ctx.cancelled):
                for node, values in update.items():
                    if node not in stages or not values:
                        continue
                    self._save_stage(ctx, node, values)
                    for key, value in values.items():
                        if key == "analyst_reports":
                            if node not in done:
                                result.setdefault(key, []).extend(value)
                        else:
                            result[key] = value
                    output, extra = self._stage_view(node, values)
                    done.add(node)
                    yield StateDeltaEvent(
                        type=EventType.STATE_DELTA,
                        delta=[_replace(f"/analysis/stages/{node}", {"status": "done", "output": output}), *extra],
                    )
                    yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=node)
                    for event in next_starts():
                        yield event

        if "final_decision" not in result:
            raise RuntimeError("The workflow ended without a final decision.")
        self._close_run(ctx, "done", result=result["final_decision"])

        prompt = EXPLAIN_PROMPT.format(result=json.dumps(_json_safe(result), indent=1))
        async for event in self._stream_llm([SystemMessage(content=prompt)], ctx.cancelled):
            yield event

    # ----- backtest: repeated decisions over a period ----------------------------
    async def _backtest(self, intent: ChatIntent, state: Dict[str, Any], ctx: RunContext) -> AsyncIterator[BaseEvent]:
        workflow = self.workflow(point_in_time=True)
        if not intent.ticker:
            async for event in self._say("Which stock should I backtest? For example: **Backtest TCS for 2025**."):
                yield event
            return
        try:
            ticker = workflow.market_data.normalize_ticker(intent.ticker)
        except ValueError as error:
            async for event in self._say(f"{error}. Phase 1 covers NSE (`.NS`) and BSE (`.BO`) stocks only."):
                yield event
            return

        defaults = self.config.backtest
        today = date.today().isoformat()
        start_date = _valid_date(intent.start_date) or defaults["start_date"]
        end_date = min(_valid_date(intent.end_date) or defaults["end_date"], today)
        rebalance_days = intent.rebalance_days if intent.rebalance_days and intent.rebalance_days > 0 else defaults[
            "rebalance_days"
        ]
        if start_date >= end_date:
            async for event in self._say(f"The start date {start_date} must be before the end date {end_date}."):
                yield event
            return

        run_id = str(uuid.uuid4())
        params = {"ticker": ticker, "startDate": start_date, "endDate": end_date, "rebalanceDays": rebalance_days}
        self._start_run(ctx, run_id, "backtest", params)
        new_state = {
            **state,
            "view": "backtest",
            "backtest": {
                "runId": run_id,
                **params,
                "status": "running",
                "total": None,
                "decisions": [],
                "strategyMetrics": None,
                "buyAndHoldMetrics": None,
                "curve": None,
            },
        }
        yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=_json_safe(new_state))
        async for event in self._say(
            f"Backtesting **{ticker}** from **{start_date}** to **{end_date}**, a new decision every "
            f"{rebalance_days} trading days. Each decision is a full agent run, so this takes a few minutes. "
            "Fundamentals and news are switched off in backtests to avoid look-ahead."
        ):
            yield event
        yield StepStartedEvent(type=EventType.STEP_STARTED, step_name="backtest")

        backtester = Backtester(workflow, workflow.market_data, self.config.market["risk_free_rate"])

        def work(emit: Callable[[Any], None]) -> None:
            outcome = backtester.run(
                ticker,
                start_date,
                end_date,
                rebalance_days,
                on_decision=lambda row, index, total: emit(("decision", row, index, total)),
                run_config=cancellable_config(ctx.cancelled),
            )
            emit(("result", outcome))

        outcome = None
        async for item in _iterate_in_thread(work, ctx.cancelled):
            if item[0] == "decision":
                _, row, index, total = item
                self._save_stage(ctx, "decision", row, item_key=row["date"])
                delta = [_add("/backtest/decisions/-", row)]
                if index == 0:
                    delta.append(_replace("/backtest/total", total))
                yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=delta)
            else:
                outcome = item[1]

        strategy_curve = (1 + outcome.strategy_returns).cumprod()
        benchmark_curve = (1 + outcome.buy_and_hold_returns).cumprod()
        curve = {
            "dates": [day.strftime("%Y-%m-%d") for day in strategy_curve.index],
            "strategy": [round(float(v), 5) for v in strategy_curve],
            "buyAndHold": [round(float(v), 5) for v in benchmark_curve],
        }
        metrics = {"strategy": outcome.strategy_metrics, "buyAndHold": outcome.buy_and_hold_metrics}
        self._save_stage(ctx, "metrics", metrics)
        self._close_run(ctx, "done", result=metrics)
        yield StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=[
                _replace("/backtest/strategyMetrics", outcome.strategy_metrics),
                _replace("/backtest/buyAndHoldMetrics", outcome.buy_and_hold_metrics),
                _replace("/backtest/curve", curve),
                _replace("/backtest/status", "done"),
            ],
        )
        yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name="backtest")

        strategy, benchmark = outcome.strategy_metrics, outcome.buy_and_hold_metrics
        async for event in self._say(
            f"Backtest finished for **{ticker}**.\n\n"
            f"- RIO strategy: return **{strategy['cumulative_return']:.1%}**, Sharpe {strategy['sharpe_ratio']:.2f}, "
            f"max drawdown {strategy['max_drawdown']:.1%}\n"
            f"- Buy & hold: return **{benchmark['cumulative_return']:.1%}**, Sharpe {benchmark['sharpe_ratio']:.2f}, "
            f"max drawdown {benchmark['max_drawdown']:.1%}\n\n"
            "Remember: no transaction costs, and the LLM may partly remember how this period ended."
        ):
            yield event

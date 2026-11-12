# Multi-Agent LLM Framework for Explainable, Risk-Aware Trading

**Phase 1 covers the Indian market only (NSE/BSE).**

RIO is a team of AI agents built with LangGraph. Four analysts (fundamental, technical,
sentiment, news) study a stock in parallel, a trader turns their reports into a
BUY / SELL / HOLD proposal with a position size, and a risk manager reviews it. The output is a
decision with a written explanation. This is the **base version (~30%)**: memory and
bull/bear debate are planned for later.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the workflow diagram, design decisions and roadmap.

## How it works (in one minute)

```
ticker + date -> 4 analysts (parallel) -> trader -> risk manager -> final decision
```

* Analysts first fetch **real data** (ratios, indicators, an ML prediction, sentiment scores) and only then ask the LLM to interpret it.
* Agents pass each other **structured objects** (see `core/schemas.py`), not free text.
* The risk manager's position cap is enforced in plain Python, even if the LLM ignores it.

## Project structure

```
rio/
├── main.py                      command line entry point
├── requirements.txt             Python dependencies
├── .env.example                 template for API keys (copy to .env)
├── config/
│   ├── config.yaml              ALL settings: LLMs, market, limits, backtest dates
│   └── loader.py                reads config.yaml and builds the chat models
├── core/
│   ├── schemas.py               the messages agents exchange (Pydantic)
│   └── state.py                 the shared state that flows through the graph
├── data/
│   ├── market_data.py           prices, fundamentals, news, ticker handling (yfinance)
│   ├── indicators.py            RSI, MACD, Bollinger, ATR, moving averages (pandas)
│   └── ml_models.py             RandomForest direction model + VADER news sentiment
├── agents/
│   ├── base_agent.py            BaseAgent and AnalystAgent (shared behaviour)
│   ├── fundamental_analyst.py   valuation, profitability, growth, debt
│   ├── technical_analyst.py     trend, momentum, volatility + ML direction model
│   ├── sentiment_analyst.py     news tone (VADER) + India VIX
│   ├── news_analyst.py          reads headlines for catalysts and risks
│   ├── trader.py                combines the reports into a trade proposal
│   └── risk_manager.py          reviews the proposal, applies position limits
├── workflow/
│   └── trading_graph.py         creates the agents and wires the LangGraph workflow
├── evaluation/
│   ├── metrics.py               cumulative return, Sharpe ratio, max drawdown, VaR
│   └── backtest.py              walk-forward backtest versus buy-and-hold
├── api/
│   ├── server.py                FastAPI app: POST /agui (AG-UI events) + serves the frontend
│   └── agui_agent.py            turns chat messages into agent runs and streams AG-UI events
└── frontend/
    ├── index.html               layout: results panel (left) + chat (right)
    ├── styles.css               light / dark theme
    └── app.js                   AG-UI client, chat, stage cards, backtest chart
```

### What each file does

| File | Purpose |
|---|---|
| `main.py` | Two commands: `analyze` (one decision, with all reports printed) and `backtest` (historical test). Normalises tickers and prints results. |
| `config/config.yaml` | The single place to configure the system (see [Configuration](#configuration)). |
| `config/loader.py` | `AppConfig` class: loads the YAML, replaces `${ENV_VAR}` with values from `.env`, returns the active market and creates the LLM each agent needs. |
| `core/schemas.py` | `AnalystReport`, `TradeProposal`, `RiskAssessment`, `FinalDecision`: the validated messages between agents. |
| `core/state.py` | `TradingState`: the dictionary every graph node reads and updates. Analyst reports are appended so parallel analysts do not overwrite each other. |
| `data/market_data.py` | `MarketDataService`: the only file that talks to Yahoo Finance. Turns `RELIANCE` into `RELIANCE.NS`, rejects non-Indian tickers, never returns data after the as-of date. |
| `data/indicators.py` | `TechnicalIndicatorCalculator`: computes technical indicators from a price table. |
| `data/ml_models.py` | `DirectionPredictor` (scikit-learn: probability the stock rises in 5 days, with its validation accuracy) and `NewsSentimentScorer` (VADER). |
| `agents/base_agent.py` | `BaseAgent` holds the LLM and asks it for structured answers. `AnalystAgent` is the common recipe for all analysts: collect data, build prompt, ask the LLM, wrap as a report. |
| `agents/*_analyst.py` | Each analyst only defines *what data it collects* and *how it phrases the prompt*. |
| `agents/trader.py` | Weighs the four reports by confidence and proposes an action and size. |
| `agents/risk_manager.py` | Calculates volatility, value-at-risk and drawdown, asks the LLM to APPROVE / ADJUST / REJECT, then enforces the maximum position size. |
| `workflow/trading_graph.py` | `TradingWorkflow`: builds all agents from the config and defines the graph (analysts in parallel, then trader, risk manager, finalize). **Best file to read first.** |
| `evaluation/metrics.py` | Pure functions for performance and risk numbers. |
| `evaluation/backtest.py` | `Backtester`: makes a decision every N trading days, applies it from the next day, and compares with buy-and-hold. |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

Then open `.env` and fill in the keys for the LLM provider(s) you will use. You only need the
provider you select in the config (next section).

## Configuration

Everything is in [config/config.yaml](config/config.yaml). Nothing in the Python code needs editing.

### 1. Choose the LLM

`llms` lists the available models: `azure_gpt` (Azure OpenAI), `openai_gpt` (OpenAI) and `claude` (Anthropic).

* Set `default_llm` to the one every agent should use.
* To use a different model for one agent, list it under `agents`, for example `trader: claude`.
* To add another model, copy a block under `llms`, give it a new name, and set `provider`, `model` and its keys.
* Values like `${OPENAI_API_KEY}` are read from your `.env` file, so secrets never go in the YAML.

| Provider | `.env` variables needed |
|---|---|
| `azure_gpt` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION` |
| `openai_gpt` | `OPENAI_API_KEY` |
| `claude` | `ANTHROPIC_API_KEY` |

### 2. Market

`markets.india` holds the market settings: exchange suffixes (`.NS`, `.BO`), benchmark index
(Nifty 50), fear index (India VIX) with its thresholds, currency (INR) and risk-free rate. `active_market: india`
selects it. Adding a market later means adding another block with the same keys.

### 3. Trading limits and backtest

| Setting | Meaning |
|---|---|
| `trading.max_position_pct` | Hard cap on one stock's share of the portfolio (0.20 = 20%). |
| `trading.price_lookback_days` | How much price history the technical analyst and ML model use. |
| `trading.news_limit` | Maximum number of headlines per run. |
| `backtest.start_date`, `end_date` | Default backtest period. |
| `backtest.rebalance_days` | A new decision is taken every this many trading days. |

## Run

Always run from the `rio` folder.

```bash
# Decision for today. RELIANCE is read as RELIANCE.NS (NSE); use TCS.BO for BSE.
python main.py analyze RELIANCE

# Decision as of a past date
python main.py analyze INFY --date 2025-06-30

# Backtest with the dates from config.yaml
python main.py backtest TCS

# Backtest with your own dates
python main.py backtest HDFCBANK --start 2024-01-01 --end 2024-12-31 --rebalance-days 21
```

### Web UI

```bash
python main.py serve              # then open http://127.0.0.1:8000
```

The chat on the right accepts plain language ("Analyze Infosys as of end of June 2025",
"Backtest TCS for H1 2025", "Why did the risk manager cut the size?"). The left panel shows every stage
as it runs: the four analyst reports (signal, confidence, key points), the trader's proposal, the risk
review with its metrics, and the final decision; backtests show progress, a metrics table, equity curves
and every decision.

**Saved history and checkpoints.** The server stores everything in SQLite (paths under `storage` in
`config.yaml`, inside the git-ignored `outputs/` folder), so it survives restarts:

* `outputs/rio.sqlite`: chats, messages, the dashboard state of each chat, every run, and the output of
  every agent stage (`stage_results`). The sidebar lists past chats; click one to reopen it, as in ChatGPT.
  `GET /api/runs/<run id>` returns a run with all its stage outputs.
* `outputs/checkpoints.sqlite`: LangGraph checkpoints taken after every stage of an analysis. If the
  server stops mid-analysis, the chat shows **Interrupted** and a **Resume from checkpoint** button that
  continues from the last finished stage instead of starting again. (Backtests are not checkpointed;
  each decision is stored in the database.)

Delete the `outputs/` folder to clear all history.

The browser and server talk over [AG-UI](https://docs.ag-ui.com), an open event protocol for agent UIs
(`api/agui_agent.py` streams `STEP_*`, `STATE_SNAPSHOT/STATE_DELTA` and `TEXT_MESSAGE_*` events;
`frontend/app.js` is a small client for them). A short "router" LLM call decides whether a message is
an analysis, a backtest or a question; it uses `default_llm` unless `agents.chat_assistant` is set.

**`analyze` prints:** each analyst's signal, confidence and key points, the trader's proposal, the risk review
and the final decision (action and position size).

**`backtest` prints:** the decision on every rebalance date, then cumulative return, Sharpe ratio, maximum
drawdown and volatility for the strategy next to a buy-and-hold benchmark. Each rebalance date is one full
agent run, so a year at 21-day steps is about 12 runs of LLM calls.

### Common problems

| Message | Fix |
|---|---|
| `Environment variable '...' is not set` | Fill that variable in `.env` (it is the one your selected LLM needs). |
| `'AAPL' ... is not listed on Indian equities` | Phase 1 accepts only NSE (`.NS`) and BSE (`.BO`) tickers. |
| An analyst says "no data was available" | Expected when the data source has nothing, see the limitations below. |

## Known limitations (base version)

* One stock at a time, long-only, no transaction costs.
* Yahoo Finance returns little or no news for Indian tickers, so the news analyst often reports "no view" and the sentiment analyst relies mostly on India VIX. A better Indian news source is a priority for the next phase.
* Backtests turn off fundamentals and news (the free data source has no history for them), so only the technical and sentiment (India VIX) analysts have data. The LLM may also "remember" how past periods ended, which can flatter results.
* Sharpe ratio and volatility assume 252 trading days a year (NSE has about 248).
* Not financial advice; an educational research prototype.

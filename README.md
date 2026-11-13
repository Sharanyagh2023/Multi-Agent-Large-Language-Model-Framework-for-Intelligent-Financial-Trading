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
│   ├── indicators.py            SMA, RSI, MACD, stochastic, Bollinger, ATR, ADX/DI, OBV, 52-week levels (pandas)
│   ├── technical_snapshot.py    labelled technical picture (trend, momentum, volatility, volume, levels)
│   └── ml_models.py             VADER news sentiment (+ the old RandomForest, no longer used)
├── ml/                          technical analyst's LightGBM direction model (see ml/README.md)
│   ├── settings.py              reads config.yaml -> technical_model
│   ├── features.py              29 features + 3-class labels (down / flat / up over 5 days)
│   ├── dataset.py               pooled Nifty 50 training table, cut at the cutoff date + price cache
│   ├── model.py                 DirectionModel: LightGBM + temperature calibration + SHAP reasons
│   ├── validation.py            purged walk-forward validation against baselines
│   ├── registry.py              picks a saved model valid for the date, or trains one
│   ├── predictor.py             MLDirectionService: prediction -> MLDirectionEvidence
│   ├── schemas.py               MLDirectionEvidence, ModelQuality, FeatureDriver
│   └── train.py                 CLI: python -m ml.train
├── agents/
│   ├── base_agent.py            BaseAgent and AnalystAgent (shared behaviour)
│   ├── fundamental_analyst.py   valuation, profitability, growth, debt
│   ├── technical_analyst.py     labelled snapshot + ML evidence + evidence tally + confidence guardrail
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
├── frontend/
│   ├── index.html               layout: results panel (left) + chat (right)
│   ├── styles.css               light / dark theme
│   └── app.js                   AG-UI client, chat, stage cards, backtest chart
└── tests/
    ├── test_technical_analyst.py  indicators, labels, no look-ahead, ML model, registry, agent, guardrail
    └── test_system.py             validation, schemas, risk guardrails, finalizer, metrics, workflow, backtester
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
| `data/indicators.py` | `TechnicalIndicatorCalculator`: scale-free indicators from a price table; every value on day t uses only prices up to day t. |
| `data/technical_snapshot.py` | `TechnicalSnapshotBuilder`: groups indicators into trend, momentum, volatility, volume, price levels and relative strength vs Nifty 50, each with a rule-based label (uptrend, overbought, high volatility, ...). |
| `data/ml_models.py` | `NewsSentimentScorer` (VADER), used by the sentiment analyst. The old `DirectionPredictor` (RandomForest) is kept but no longer used. |
| `ml/` | The technical analyst's direction model: LightGBM trained on pooled Nifty 50 history, validated walk-forward, calibrated, explained with SHAP. See [Technical analyst and ML model](#technical-analyst-and-ml-model) and [ml/README.md](ml/README.md). |
| `tests/` | Offline pytest suite (synthetic prices, fake LLM): `python -m pytest tests -q`. |
| `agents/base_agent.py` | `BaseAgent` holds the LLM and asks it for structured answers. `AnalystAgent` is the common recipe for all analysts: collect data, build prompt, ask the LLM, wrap as a report. |
| `agents/*_analyst.py` | Each analyst only defines *what data it collects* and *how it phrases the prompt*. |
| `agents/trader.py` | Weighs the four reports by confidence and proposes an action and size. |
| `agents/risk_manager.py` | Calculates volatility, value-at-risk and drawdown, asks the LLM to APPROVE / ADJUST / REJECT, then enforces the maximum position size. |
| `workflow/trading_graph.py` | `TradingWorkflow`: builds all agents from the config and defines the graph (analysts in parallel, then trader, risk manager, finalize). **Best file to read first.** |
| `evaluation/metrics.py` | Pure functions for performance and risk numbers. |
| `evaluation/backtest.py` | `Backtester`: makes a decision every N trading days, applies it from the next day, and compares with buy-and-hold. |

## Technical analyst and ML model

The technical analyst ([agents/technical_analyst.py](agents/technical_analyst.py)) returns the same
`AnalystReport` as every other analyst, so the trader and risk manager need nothing special. Inside, it
works in four steps:

1. **Clean prices.** It fetches about two years of prices for the stock and the Nifty 50, up to the
   as-of date only. While NSE is open (before 15:30 IST), today's unfinished bar is dropped.
2. **Labelled snapshot.** Indicators are grouped by theme and given rule-based labels in Python, so the
   LLM reads `"trend": "uptrend", "rsi_zone": "overbought"` instead of applying thresholds itself.
3. **ML evidence.** A LightGBM model gives the probability of the stock going **down / flat / up over the
   next 5 trading days**, the top-5 SHAP reasons, and its own measured quality (`useful`, `weak` or `no_edge`).
   "Flat" means a move smaller than half the stock's normal 5-day move.
4. **LLM with a guardrail.** The prompt is three JSON blocks (snapshot, ML evidence, evidence tally) plus
   reading and output rules. Python computes an allowed confidence range from how much the evidence
   agrees, and clips the LLM's confidence to it. Key points are tagged (`[Trend]`, `[ML model]`, ...) and
   cite numbers.

**How the model is kept honest**

* **Pooled training data.** About 47 Nifty 50 stocks over 8 years (~92k rows), because one stock alone
  has too little history.
* **Purged walk-forward validation.** The model is tested only on later, unseen dates, with a 5-day gap
  so training labels cannot overlap the test period. It is compared with a class-prior baseline, logistic
  regression and random forest.
* **Point-in-time models.** A saved model is used only if its cutoff date is not after the analysis date,
  and it is re-used for up to 90 days. Otherwise a new one is trained and saved under `ml/artifacts/`.
  Backtests therefore never use a model that has seen the future (the 2025 backtest trained its own
  models with cutoffs 2025-01-01, 07-04 and 10-06).

**Results** (cutoff 2026-11-12, test period 2023-08 to 2026-11, 37,026 unseen predictions):

| Model | Balanced accuracy | Log loss | Up/down calls correct |
|---|---|---|---|
| **LightGBM (used)** | **0.377** | **1.079** | **51.6%** |
| Logistic regression | 0.377 | 1.090 | 48.6% |
| Random forest (old model) | 0.369 | 1.094 | 48.3% |
| Chance (class prior) | 0.333 | 1.085 | - |

LightGBM is the only model whose log loss beats chance. The edge is small, so the model is rated
**weak**: the LLM mentions it but gives it little weight. Five-day moves of large-cap stocks are mostly
noise, and the agent is told that honestly. Full details are in [ml/README.md](ml/README.md).

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

`llms` lists the available models: `ollama_qwen` (local, the default), `azure_gpt` (Azure OpenAI), `openai_gpt` (OpenAI) and `claude` (Anthropic).

* Set `default_llm` to the one every agent should use.
* To use a different model for one agent, list it under `agents`, for example `trader: claude`.
* To add another model, copy a block under `llms`, give it a new name, and set `provider`, `model` and its keys.
* Values like `${OPENAI_API_KEY}` are read from your `.env` file, so secrets never go in the YAML.

| Provider | `.env` variables needed |
|---|---|
| `ollama_qwen` | none: install [Ollama](https://ollama.com), run `ollama pull qwen2.5:7b` and keep Ollama running |
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

### 4. Technical analyst ML model

All keys under `technical_model` are optional (defaults in `ml/settings.py`); an unknown key is an error.

| Setting | Meaning |
|---|---|
| `horizon_days` | Days ahead the model predicts (5). |
| `flat_threshold_vol_multiple` | A move smaller than this many "normal moves" is labelled flat (0.5). |
| `training_years` | Years of history used for training (8). |
| `max_model_age_days` | Retrain when the newest valid model is older than this (90). |
| `min_edge` | Balanced accuracy above chance (0.333) needed for "weak"; twice this for "useful" (0.03). |
| `validation_folds`, `validation_share` | Walk-forward test blocks, and the share of recent dates they cover. |
| `calibration_share` | Share of the most recent training dates used to calibrate probabilities. |
| `top_drivers` | Number of SHAP reasons given to the LLM. |
| `artifacts_dir` | Where models, model cards and the price cache are stored (`ml/artifacts`, git-ignored). |
| `universe` | Optional list of training tickers; leave it out to use ~47 Nifty 50 large caps. |

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

# Train the technical analyst's ML model and print the comparison with the baselines
python -m ml.train                        # cutoff = today
python -m ml.train --cutoff 2025-01-01    # a model that only knows data up to that date

# Run the offline tests (no internet or LLM needed)
python -m pytest tests -q
```

You do not have to run `ml.train` first: the first analysis trains a model automatically if none
is saved (about 1 minute, mostly downloading prices; later runs load it in under a second).

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
| Technical report says the ML model is "unavailable" | The reason is in the report: too little history (needs about 200 trading days) or training failed (check the internet connection). The agent still reports from the indicators. |
| The first analysis or backtest pauses for about a minute | The direction model is being trained for that date; the model and prices are cached in `ml/artifacts/` afterwards. |

## Known limitations (base version)

* One stock at a time, long-only, no transaction costs.
* Yahoo Finance returns little or no news for Indian tickers, so the news analyst often reports "no view" and the sentiment analyst relies mostly on India VIX. A better Indian news source is a priority for the next phase.
* Backtests turn off fundamentals and news (the free data source has no history for them), so only the technical and sentiment (India VIX) analysts have data. The LLM may also "remember" how past periods ended, which can flatter results.
* Sharpe ratio and volatility assume 252 trading days a year (NSE has about 248).
* The technical ML model has a real but small edge (balanced accuracy 0.377 vs 0.333 chance), so it is evidence, never the decision.
* The ML training universe is today's Nifty 50 large caps (survivorship bias). Yahoo leaves a few corporate actions unadjusted (e.g. TRENT 2026-01-01, TMPV demerger), which distorts those stocks' indicators; a data-quality guard is planned for Phase 2.
* Yahoo prices match NSE's official bhavcopy for `.NS` tickers, but most `.BO` (BSE) tickers return almost no history, so use NSE symbols.
* Not financial advice; an educational research prototype.

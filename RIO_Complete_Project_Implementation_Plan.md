# RIO Project --- Complete Implementation Plan

**Project title:** A Multi-Agent Large Language Model Framework for
Intelligent, Explainable, and Risk-Aware Financial Trading\
**Planning basis:** The current implementation ZIP and the previously
supplied project documents, including the project description,
objectives, literature survey, requirement analysis, and system-design
materials.\
**Plan status:** Objective 1 is treated as completed according to the
project team's current status. Objective 2 is the next implementation
priority. Objectives 3 and 4 remain planned work unless verified in the
codebase.

------------------------------------------------------------------------

## 1. Project overview

The project aims to develop a multi-agent Large Language Model (LLM)
framework that simulates the collaborative structure of a professional
financial trading team. Specialized agents analyze different financial
information sources, communicate their findings through structured
reports, develop trading recommendations, debate opposing market views,
and evaluate decisions through risk management.

The proposed system integrates:

-   Fundamental analysis
-   Technical analysis
-   News analysis
-   Market and investor sentiment analysis
-   Trader decision-making
-   Bullish and bearish research debate
-   Risk assessment
-   Historical backtesting and performance evaluation
-   Explainable trading recommendations

The current project scope is focused on Indian equities. The existing
implementation uses Python, LangGraph/LangChain, Pydantic, yfinance,
Pandas/NumPy, scikit-learn, VADER, and a configurable LLM provider with
local Ollama support.

## 2. Project objectives and completion status

  ------------------------------------------------------------------------
  Objective               Description             Current status
  ----------------------- ----------------------- ------------------------
  Objective 1             To develop a            **Completed according to
                          multi-agent LLM         current project
                          framework that          status.** The
                          simulates the           orchestration
                          collaborative structure foundation, agent
                          of a professional       structure, shared state,
                          financial trading team. and trading workflow are
                                                  implemented.

  Objective 2             To integrate            **Partially
                          specialized agents for  implemented.** Four
                          fundamental, technical, analyst modules exist
                          sentiment, and news     and are connected to the
                          analysis to support     workflow. Their data
                          informed trading        coverage, analytical
                          decisions.              quality, validation, and
                                                  reliability need
                                                  improvement.

  Objective 3             To implement structured **Planned.** Structured
                          agent communication and analyst schemas exist as
                          multi-agent debate for  a foundation; formal
                          efficient and           debate and
                          explainable             evidence-aware
                          decision-making.        communication need
                                                  implementation and
                                                  validation.

  Objective 4             To incorporate risk     **Partially implemented
                          management mechanisms   foundation.** A risk
                          and evaluate the        manager, finalization
                          framework using         logic, and
                          financial performance   backtesting/evaluation
                          metrics such as         modules exist. Their
                          cumulative return,      completeness and
                          Sharpe ratio, and       correctness must be
                          maximum drawdown.       tested and enhanced.
  ------------------------------------------------------------------------

**Status interpretation:** A module being present in the repository does
not automatically mean its objective is complete. Completion should be
declared only after the acceptance criteria and tests in this plan pass.

------------------------------------------------------------------------

## 3. Existing implementation baseline

### 3.1 Existing architecture

The current codebase contains the following major areas:

  -----------------------------------------------------------------------
  Area                                Existing modules / responsibilities
  ----------------------------------- -----------------------------------
  Entry point                         `main.py` --- CLI commands for
                                      analysis and backtesting

  Configuration                       `config/config.yaml`,
                                      `config/loader.py` --- model,
                                      market, risk, and backtest
                                      configuration

  Agents                              `agents/` --- base agent, four
                                      analysts, trader, and risk manager

  Schemas                             `core/schemas.py` --- Pydantic
                                      contracts for analyst reports,
                                      proposals, risk assessments, and
                                      final decisions

  Shared state                        `core/state.py` --- LangGraph
                                      trading state and accumulated
                                      analyst reports

  Workflow                            `workflow/trading_graph.py` ---
                                      parallel analyst execution followed
                                      by trader, risk manager, and
                                      finalization

  Market data                         `data/market_data.py` --- ticker
                                      normalization and market data
                                      retrieval

  Indicators                          `data/indicators.py` --- technical
                                      indicator calculations

  ML and sentiment                    `data/ml_models.py` --- Random
                                      Forest direction predictor and
                                      VADER sentiment functionality

  Evaluation                          `evaluation/metrics.py`,
                                      `evaluation/backtest.py` ---
                                      performance metrics and historical
                                      backtesting

  Tests                               `tests/` --- risk/finalization,
                                      workflow integration, and LLM
                                      fallback tests
  -----------------------------------------------------------------------

### 3.2 Existing workflow

The current graph is organized around the following flow:

1.  Receive a ticker and analysis date.
2.  Retrieve relevant market information.
3.  Run Fundamental, Technical, Sentiment, and News Analysts in
    parallel.
4.  Collect the four analyst reports in shared state.
5.  Pass the reports to the Trader Agent.
6.  Generate a structured trade proposal.
7.  Pass the proposal to the Risk Manager.
8.  Approve, adjust, or reject the proposal.
9.  Produce the final trading decision.

### 3.3 Known baseline issues to address

The earlier live execution demonstrated that the workflow can run
end-to-end, but it also exposed analytical quality issues that must be
corrected:

-   News analysis may receive no usable articles for a ticker.
-   Sentiment explanations may misinterpret the India VIX value or its
    threshold.
-   Technical explanations may contradict the numerical RSI value.
-   Technical output may contain malformed comparisons or unsupported
    claims.
-   Market-data availability and quality can vary by ticker and date.
-   Data-source limitations must be made visible rather than hidden by
    generated explanations.

These issues should become explicit regression tests.

------------------------------------------------------------------------

# 4. Objective 1 --- Multi-agent framework foundation

**Objective statement:** To develop a multi-agent LLM framework that
simulates the collaborative structure of a professional financial
trading team.

**Current status:** Completed according to the project team's status.
This section records the foundation that subsequent objectives must
preserve.

### 4.1 Existing capabilities to preserve

-   Modular agent architecture.
-   Common base-agent functionality.
-   LangGraph workflow orchestration.
-   Parallel execution of the four specialist analyst agents.
-   Shared trading state.
-   Pydantic data contracts.
-   Trader and Risk Manager workflow stages.
-   Configurable LLM provider and local Ollama fallback.
-   CLI entry points for analysis and backtesting.

### 4.2 Maintenance and regression requirements

Although Objective 1 is considered complete, later work must not break
its functionality.

Tasks:

1.  Preserve existing public interfaces unless a migration is necessary.
2.  Keep configuration centralized.
3.  Keep agent outputs validated through Pydantic.
4.  Keep the graph state and reducer behavior correct when analysts run
    in parallel.
5.  Retain provider fallback behavior.
6.  Run existing tests after each significant change.
7.  Document any schema or workflow migration.

**Acceptance criteria**

-   Existing analysis workflow still executes.
-   Parallel analyst reports are all retained in shared state.
-   Trader and risk manager receive the expected structured data.
-   Existing risk/finalization and LLM fallback tests continue to pass.
-   CLI behavior remains compatible.

------------------------------------------------------------------------

# 5. Objective 2 --- Specialized financial-analysis agents

**Objective statement:** To integrate specialized agents for
fundamental, technical, sentiment, and news analysis to support informed
trading decisions.

**Current status:** Partially implemented. The four agent classes and
parallel workflow integration already exist. Work should focus on
improving the evidence, analysis, structured outputs, and tests rather
than recreating the agents.

## 5.1 Common design for all specialist agents

Each analyst should follow a consistent pipeline:

1.  Receive ticker, analysis date, and relevant configuration.
2.  Request role-specific information from the data layer.
3.  Validate the data and identify missing fields.
4.  Calculate deterministic metrics where appropriate.
5.  Prepare a concise evidence package for the LLM.
6.  Ask the LLM to interpret only the supplied evidence.
7.  Parse the response into the common analyst schema.
8.  Validate signal, confidence, summary, and evidence.
9.  Record sources, data availability, and limitations.
10. Return a structured report to the shared workflow state.

### 5.1.1 Recommended common report contract

Retain the current `AnalystOpinion` / `AnalystReport` fields and extend
them compatibly where useful.

  Field              Meaning
  ------------------ --------------------------------------------
  `analyst`          Specialist agent identifier
  `ticker`           Asset being analyzed
  `signal`           `bullish`, `neutral`, or `bearish`
  `confidence`       Numeric confidence from 0 to 1
  `summary`          Concise analytical conclusion
  `key_points`       Evidence supporting the conclusion
  `data_available`   Whether sufficient evidence was available
  `data_sources`     Data providers or source identifiers
  `limitations`      Missing data, stale data, or known caveats
  `as_of_date`       Date to which the analysis applies

The exact schema migration should be backward-compatible with the Trader
Agent and tests. Do not invent data-source values; record only sources
actually used.

## 5.2 Fundamental Analyst

**Existing module:** `agents/fundamental_analyst.py`\
**Supporting module:** `data/market_data.py`

### Responsibilities

Evaluate the company's financial condition, operating performance,
growth, valuation, and financial health.

### Data to retrieve where available

  -----------------------------------------------------------------------
  Category                            Metrics / evidence
  ----------------------------------- -----------------------------------
  Company profile                     Company name, sector, industry,
                                      market capitalization

  Valuation                           P/E, forward P/E where available,
                                      P/B

  Profitability                       ROE, profit margin, operating
                                      margin

  Growth                              Revenue growth, earnings growth

  Financial health                    Debt-to-equity, current ratio, free
                                      cash flow

  Financial statements                Income statement, balance sheet,
                                      cash-flow statement, historical
                                      periods
  -----------------------------------------------------------------------

The availability and meaning of each metric depend on the data provider
and the company. Missing metrics must remain missing; they must not be
replaced with guessed values.

### Implementation tasks

1.  Audit the existing Yahoo Finance fundamental retrieval.
2.  Separate raw provider data retrieval from metric normalization.
3.  Normalize units, periods, and field names.
4.  Handle missing, null, infinite, and invalid values.
5.  Add ratio calculations only when the required source values are
    available.
6.  Add historical comparisons when enough periods exist.
7.  Interpret valuation and financial ratios in company/sector context
    where possible.
8.  Build an evidence package for the LLM.
9.  Improve the fundamental-analysis prompt to prohibit unsupported
    numerical claims.
10. Return the common structured analyst report.
11. Add unit tests for complete, partial, and unavailable financial
    data.

### Expected output

A fundamental signal with supporting financial metrics,
company-performance observations, and explicit limitations.

### Acceptance criteria

-   Metrics are traceable to retrieved data or deterministic
    calculations.
-   Missing metrics are clearly identified.
-   The LLM does not invent financial figures.
-   The signal and explanation do not contradict the evidence.
-   Tests cover different data-availability conditions.

## 5.3 Technical Analyst

**Existing modules:** `agents/technical_analyst.py`,
`data/indicators.py`, `data/ml_models.py`

### Responsibilities

Analyze historical price and volume behavior, trends, momentum, and
volatility.

### Indicators in scope

  Indicator         Purpose
  ----------------- ---------------------------------------------
  SMA 20            Short-term trend
  SMA 50            Medium-term trend
  SMA 200           Long-term trend
  RSI 14            Momentum and overbought/oversold conditions
  MACD              Trend and momentum confirmation
  Bollinger Bands   Relative price position and volatility
  ATR               Volatility measurement
  Volume ratio      Relative trading activity

### Implementation tasks

1.  Validate OHLCV history, timestamps, missing rows, and duplicate
    rows.
2.  Check adjusted/unadjusted price handling and document the selected
    convention.
3.  Verify indicator calculations against deterministic test data.
4.  Handle insufficient lookback windows.
5.  Implement deterministic interpretation rules for each indicator.
6.  Ensure RSI thresholds and other rules are applied consistently.
7.  Combine indicators into an evidence package without treating
    correlated indicators as independent proof.
8.  Review the existing Random Forest direction predictor.
9.  Use chronological training/validation splits; avoid random shuffling
    for time-series evaluation.
10. Prevent target leakage and ensure features are available at
    prediction time.
11. Report model evaluation separately from the LLM's confidence.
12. Ask the LLM to explain the evidence, not recalculate or invent
    indicator values.
13. Add tests for trend, momentum, volatility, missing data, and
    insufficient history.

### Acceptance criteria

-   Indicator calculations match expected values on test fixtures.
-   Interpretations match the numerical thresholds.
-   The agent does not describe one RSI value as both overbought and
    oversold.
-   Model evaluation uses time-ordered data and is reproducible.
-   The report identifies the analysis date and data limitations.

## 5.4 News Analyst

**Existing module:** `agents/news_analyst.py`\
**Supporting module:** `data/market_data.py`\
**Potential new module:** `data/news_provider.py`

### Responsibilities

Identify relevant company-specific and market-wide events and explain
their potential impact on the company.

### News categories

-   Earnings and financial results
-   Mergers and acquisitions
-   Management changes
-   Regulatory and legal announcements
-   RBI and government policy developments
-   Industry developments
-   Global economic events
-   Other material company announcements

### Implementation tasks

1.  Audit the existing Yahoo Finance news integration.
2.  Define a common news-provider interface.
3.  Evaluate accessible news sources suitable for Indian equities.
4.  Retrieve company-specific and relevant market-wide articles.
5.  Normalize headline, summary, publisher, publication time, URL, and
    ticker relevance.
6.  Remove duplicate or near-duplicate articles.
7.  Filter irrelevant articles and respect the requested analysis date.
8.  Classify events into defined categories.
9.  Estimate event direction and impact level with evidence.
10. Preserve source and publication details for traceability.
11. Handle empty results, unavailable providers, and provider errors.
12. Add fixtures and tests for relevant, irrelevant, duplicated, stale,
    and unavailable news.

### Acceptance criteria

-   Relevant articles are associated with the correct ticker or market
    context.
-   Duplicates are not counted as separate independent events.
-   Publication dates are checked against the analysis date.
-   Impact explanations refer to retrieved article content.
-   No-news cases are explicitly reported instead of generating
    fictional headlines.

## 5.5 Sentiment Analyst

**Existing module:** `agents/sentiment_analyst.py`\
**Supporting module:** `data/ml_models.py`\
**Potential new module:** `data/sentiment.py`

### Responsibilities

Measure the tone of available market commentary and distinguish
company-specific sentiment from broader market conditions.

### Existing foundation

-   VADER sentiment scoring.
-   Yahoo Finance news-derived text.
-   India VIX as a market-wide volatility indicator.

### Implementation tasks

1.  Audit the current VADER scoring and aggregation.
2.  Normalize and clean article text before scoring.
3.  Calculate positive, negative, and neutral proportions.
4.  Aggregate article-level scores with a documented method.
5.  Consider source quality, recency, and duplicate removal.
6.  Separate company-specific sentiment from broad market mood.
7.  Interpret India VIX as a volatility indicator, not as a direct
    bullish/bearish sentiment score.
8.  Validate any VIX thresholds against documented rules and avoid
    unsupported labels.
9.  Ask the LLM to summarize the measured sentiment and limitations.
10. Add tests for positive, negative, neutral, mixed, empty, and
    duplicated input.

### Acceptance criteria

-   Sentiment values are reproducible from the input text and scoring
    method.
-   The report explains article coverage and sample size.
-   India VIX is not misrepresented as a direct sentiment score.
-   Empty or weak coverage reduces confidence and is disclosed.
-   Sentiment conclusions are consistent with the aggregate scores.

## 5.6 Distinguishing News and Sentiment responsibilities

The two agents may use some of the same articles, but their tasks must
remain distinct:

  -----------------------------------------------------------------------
  News Analyst                        Sentiment Analyst
  ----------------------------------- -----------------------------------
  Identifies what happened            Measures the tone of available
                                      commentary

  Classifies events and catalysts     Aggregates positive, negative, and
                                      neutral signals

  Explains possible company impact    Explains overall company-specific
                                      or market sentiment

  Emphasizes event relevance and      Emphasizes sentiment distribution,
  source details                      coverage, and confidence
  -----------------------------------------------------------------------

The system must not treat two reports based on the same underlying
article as two independent confirmations.

## 5.7 Objective 2 integration and validation

Tasks:

1.  Ensure all four analysts receive the same ticker and analysis date.
2.  Ensure each analyst receives only the data relevant to its role.
3.  Validate each output against the shared schema.
4.  Preserve all four reports in the LangGraph state.
5.  Confirm that the Trader Agent receives all reports.
6.  Add integration tests for partial and failed analyst outputs.
7.  Run end-to-end tests for multiple Indian tickers.
8.  Inspect report quality manually against the underlying evidence.

**Objective 2 acceptance criteria**

-   All four analyst modules execute through the workflow.
-   Each report is structured and validated.
-   Each conclusion is supported by the data supplied to the agent.
-   Missing or unreliable data is disclosed.
-   No agent invents unavailable financial values, headlines, or
    indicators.
-   All four reports reach the Trader Agent.
-   Unit and integration tests pass.

------------------------------------------------------------------------

# 6. Objective 3 --- Structured communication and multi-agent debate

**Objective statement:** To implement structured agent communication and
multi-agent debate for efficient and explainable decision-making.

**Current status:** Planned. Existing Pydantic schemas and shared state
provide the foundation, but a formal debate process and evidence-aware
synthesis must be implemented.

## 6.1 Structured agent communication

### Purpose

Reduce information loss and context overload caused by passing long,
unstructured natural-language messages between agents.

### Implementation tasks

1.  Review current schemas in `core/schemas.py`.
2.  Define versioned contracts for analyst reports, debate arguments,
    rebuttals, and final synthesis.
3.  Include ticker, analysis date, signal, confidence, evidence, source,
    and limitations.
4.  Validate every message at the workflow boundary.
5.  Keep messages concise and role-specific.
6.  Define behavior for invalid, missing, or conflicting reports.
7.  Preserve the original analyst reports so the decision can be traced
    back to its evidence.

### Proposed debate message

  Field                   Meaning
  ----------------------- ------------------------------------------
  `agent_role`            Bullish researcher or bearish researcher
  `ticker`                Asset under discussion
  `position`              Bullish or bearish
  `claim`                 Main argument
  `supporting_evidence`   References to analyst evidence
  `counter_evidence`      Evidence that weakens the argument
  `uncertainties`         Missing or ambiguous information
  `confidence`            Confidence from 0 to 1

## 6.2 Bullish Researcher Agent

### Responsibilities

-   Build the strongest evidence-based bullish case.
-   Identify positive fundamental developments.
-   Identify supportive technical trends.
-   Consider positive news and sentiment.
-   Identify assumptions and risks that could invalidate the bullish
    view.
-   Reference the original analyst evidence.

### Output

A structured bullish argument with claims, supporting evidence,
counter-evidence, uncertainties, and confidence.

## 6.3 Bearish Researcher Agent

### Responsibilities

-   Build the strongest evidence-based bearish case.
-   Identify financial weaknesses and valuation concerns.
-   Identify negative technical conditions.
-   Consider negative news and sentiment.
-   Identify downside catalysts and assumptions.
-   Reference the original analyst evidence.

### Output

A structured bearish argument with claims, supporting evidence,
counter-evidence, uncertainties, and confidence.

## 6.4 Debate and rebuttal process

Proposed sequence:

1.  Four specialist analysts submit their reports.
2.  Bullish Researcher receives the reports and creates a bullish
    argument.
3.  Bearish Researcher receives the reports and creates a bearish
    argument.
4.  Each researcher receives the opposing argument.
5.  Each researcher produces a concise rebuttal.
6.  A synthesis or Trader stage reviews the original reports, both
    arguments, and rebuttals.
7.  The system records unresolved disagreements and uncertainty.
8.  The Trader generates the proposal.

The debate must not allow an agent to introduce unsupported facts.
Arguments should cite the report evidence they rely on.

## 6.5 LangGraph workflow changes

The planned graph should evolve from:

`Analysts → Trader → Risk Manager → Finalize`

to:

`Analysts → Bullish/Bearish Research → Rebuttals → Synthesis/Trader → Risk Manager → Finalize`

Implementation tasks:

1.  Add debate state fields in `core/state.py`.
2.  Add structured debate schemas in `core/schemas.py`.
3.  Implement bullish and bearish researcher modules.
4.  Add debate nodes and edges in `workflow/trading_graph.py`.
5.  Define retry and failure behavior.
6.  Keep the original analyst reports available to the Trader.
7.  Add workflow tests for agreement, disagreement, and missing analyst
    reports.

## 6.6 Explainability

Every final recommendation should be traceable to:

-   Specialist analyst reports.
-   Evidence and source information.
-   Bullish and bearish arguments.
-   Rebuttals and unresolved disagreements.
-   Trader rationale.
-   Risk manager decision and comments.

**Objective 3 acceptance criteria**

-   Agent communication uses validated structured contracts.
-   Bullish and bearish arguments are both generated.
-   Rebuttals refer to the opposing arguments and original evidence.
-   The synthesis considers both sides rather than merely selecting the
    more confident LLM response.
-   Unsupported claims are rejected or marked as unverified.
-   The final explanation can be traced to the underlying analyst
    evidence.
-   Tests cover agreement, disagreement, invalid messages, and agent
    failure.

------------------------------------------------------------------------

# 7. Objective 4 --- Risk management and financial evaluation

**Objective statement:** To incorporate risk management mechanisms and
evaluate the framework using financial performance metrics such as
cumulative return, Sharpe ratio, and maximum drawdown.

**Current status:** A risk manager, finalization logic, evaluation
metrics, and backtesting modules already exist. Their correctness,
completeness, and integration with the expanded workflow must be
validated.

## 7.1 Risk Manager

**Existing module:** `agents/risk_manager.py`\
**Related modules:** `core/schemas.py`, `workflow/trading_graph.py`,
`config/config.yaml`

### Existing decision contract

The current risk assessment uses:

-   Decision: `APPROVE`, `ADJUST`, or `REJECT`
-   Risk level: `low`, `medium`, or `high`
-   Adjusted position size
-   Comments

The finalizer maps this risk decision to the final action and position
size.

### Risk controls to implement or validate

1.  Maximum position size.
2.  Maximum portfolio exposure.
3.  Volatility-based risk assessment.
4.  Maximum drawdown monitoring.
5.  Risk/reward evaluation where required data exists.
6.  Concentration limits where portfolio holdings are available.
7.  Missing-data and stale-data safeguards.
8.  Explicit rejection or adjustment rules.
9.  Deterministic validation of position-size limits.
10. Audit trail of the risk decision.

Do not claim that a portfolio-level constraint is implemented until the
system has the required portfolio data and tests.

## 7.2 Risk management perspectives

The project description proposes a risk management team with three
viewpoints:

  -----------------------------------------------------------------------
  Perspective                         Responsibility
  ----------------------------------- -----------------------------------
  Risk-seeking                        Evaluate potential opportunity
                                      while identifying downside exposure

  Neutral                             Balance expected opportunity and
                                      risk

  Conservative                        Emphasize capital preservation and
                                      downside protection
  -----------------------------------------------------------------------

Implementation tasks:

1.  Define the three risk perspectives as separate analysis roles or
    clearly separated evaluation stages.
2.  Give each perspective the same trade proposal and relevant risk
    evidence.
3.  Require structured outputs with supporting evidence.
4.  Add a risk synthesis stage.
5.  Produce a final `APPROVE`, `ADJUST`, or `REJECT` decision.
6.  Enforce hard risk limits deterministically after LLM analysis.

The LLM may explain risk, but hard position-size and exposure limits
should be enforced by ordinary validated code.

## 7.3 Final decision logic

Preserve and test the existing finalization behavior:

-   `REJECT`: final action becomes `HOLD` with zero position size.
-   `ADJUST`: retain the proposal action but use the risk-adjusted
    position size.
-   `APPROVE`: retain the proposal action and ensure the position size
    does not exceed the permitted risk-adjusted size.

Add tests for boundary values, invalid sizes, missing risk assessments,
and each decision branch.

## 7.4 Backtesting

**Existing modules:** `evaluation/backtest.py`, `evaluation/metrics.py`

### Purpose

Evaluate the framework using historical market data and compare its
performance with an appropriate benchmark.

### Implementation tasks

1.  Review the existing walk-forward backtesting implementation.
2.  Define the evaluation universe and historical date range.
3.  Define when a signal is generated and when a simulated trade is
    executed.
4.  Ensure only information available at the decision time is used.
5.  Prevent look-ahead bias and data leakage.
6.  Account for transaction costs and slippage where the required
    assumptions are available.
7.  Define cash, position sizing, and portfolio accounting rules.
8.  Compare the strategy with a suitable buy-and-hold benchmark.
9.  Save trades, decisions, equity curve, and evaluation results.
10. Test calculations using small deterministic price series.

### Required evaluation metrics

  -----------------------------------------------------------------------
  Metric                              Purpose
  ----------------------------------- -----------------------------------
  Cumulative return                   Total portfolio growth or decline
                                      over the test period

  Sharpe ratio                        Return relative to volatility,
                                      using a documented risk-free rate
                                      and annualization convention

  Maximum drawdown                    Largest peak-to-trough portfolio
                                      decline
  -----------------------------------------------------------------------

Additional metrics may include volatility, Value at Risk, and
benchmark-relative performance if the implementation supports them.

### Evaluation safeguards

-   Use time-ordered walk-forward evaluation.
-   Do not use future prices or future news in earlier decisions.
-   Ensure fundamentals are point-in-time appropriate.
-   Record missing data and skipped periods.
-   Document trading-cost and execution assumptions.
-   Use the same evaluation period and assumptions for strategy and
    benchmark.

## 7.5 Results visualization and reporting

The project requirements include presentation of trading
recommendations, portfolio performance, risk metrics, and historical
trends through charts, tables, and reports.

Planned outputs:

-   Buy/Sell/Hold decision history.
-   Entry and exit records for simulated trades.
-   Portfolio equity curve.
-   Benchmark comparison.
-   Cumulative return.
-   Sharpe ratio.
-   Maximum drawdown.
-   Volatility and other supported risk measures.
-   Analyst report and risk-decision explanations.

A user interface or dashboard can be implemented after the analytical
and backtesting pipeline is reliable. The project requirements mention
Streamlit as a possible interface technology; the final choice should be
made based on the team's scope and time.

**Objective 4 acceptance criteria**

-   Risk manager returns validated decisions.
-   Hard risk limits are enforced deterministically.
-   Finalization handles approve, adjust, and reject correctly.
-   Backtesting uses point-in-time data and avoids look-ahead bias.
-   Cumulative return, Sharpe ratio, and maximum drawdown are tested
    against known examples.
-   Strategy performance is compared with a benchmark.
-   Evaluation assumptions and limitations are documented.

------------------------------------------------------------------------

# 8. Cross-cutting engineering requirements

These activities apply to all four objectives.

## 8.1 Configuration and secrets

-   Keep model, market, risk, and backtest settings in configuration.
-   Keep API keys and credentials in environment variables.
-   Never commit `.env` or secrets.
-   Provide a safe `.env.example`.
-   Validate configuration at startup.
-   Keep local Ollama support and configured provider fallback behavior.

## 8.2 Data quality and source traceability

-   Normalize tickers and timestamps.
-   Validate market-data schemas.
-   Handle missing and duplicate observations.
-   Record source and retrieval metadata.
-   Make data availability visible to agents.
-   Respect analysis-date boundaries.
-   Avoid using current information in historical simulations.

## 8.3 LLM reliability

-   Use structured output parsing.
-   Validate all LLM-generated fields.
-   Set clear role-specific prompts.
-   Require evidence-based explanations.
-   Prevent unsupported numerical claims.
-   Handle rate limits, quota errors, timeouts, and provider failures.
-   Use bounded retries and fallback behavior.
-   Keep deterministic calculations outside the LLM.

## 8.4 Testing strategy

  -----------------------------------------------------------------------
  Test level                          Purpose
  ----------------------------------- -----------------------------------
  Unit tests                          Validate individual calculations,
                                      parsers, schemas, and agent logic

  Data-provider tests                 Validate normalization, empty
                                      results, and provider errors

  Agent tests                         Validate role-specific evidence and
                                      structured reports

  Workflow tests                      Validate graph routing, shared
                                      state, and parallel report
                                      collection

  Integration tests                   Validate all agents working
                                      together

  Regression tests                    Ensure previously working features
                                      remain functional

  Backtest tests                      Validate accounting, metrics, and
                                      time boundaries
  -----------------------------------------------------------------------

Use deterministic fixtures for tests. Tests should not depend on live
market APIs or paid LLM services unless explicitly marked as external
integration tests.

------------------------------------------------------------------------

# 9. Recommended repository organization

The following is a target organization, not a claim that every listed
file already exists. Preserve the current naming and structure where
possible; add modules only when needed.

``` text
project/
│
├── main.py
├── config/
│   ├── config.yaml
│   └── loader.py
│
├── agents/
│   ├── base_agent.py
│   ├── fundamental_analyst.py
│   ├── technical_analyst.py
│   ├── sentiment_analyst.py
│   ├── news_analyst.py
│   ├── bullish_researcher.py
│   ├── bearish_researcher.py
│   ├── trader.py
│   └── risk_manager.py
│
├── core/
│   ├── schemas.py
│   └── state.py
│
├── data/
│   ├── market_data.py
│   ├── indicators.py
│   ├── ml_models.py
│   ├── news_provider.py
│   └── sentiment.py
│
├── workflow/
│   └── trading_graph.py
│
├── evaluation/
│   ├── metrics.py
│   └── backtest.py
│
├── tests/
│   ├── test_fundamental_analyst.py
│   ├── test_technical_analyst.py
│   ├── test_sentiment_analyst.py
│   ├── test_news_analyst.py
│   ├── test_debate.py
│   ├── test_risk_and_finalize.py
│   ├── test_workflow_integration.py
│   └── test_backtest.py
│
├── results/
├── logs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# 10. End-to-end target workflow

The final intended system workflow is:

``` text
User Input
    |
    v
Configuration and Input Validation
    |
    v
Market Data Collection
    |
    +-------------------+-------------------+
    |                   |                   |
    v                   v                   v
Historical Prices   Fundamentals        News Data
    |                   |                   |
    v                   v                   v
Technical Analyst  Fundamental Analyst  News Analyst
    |                                       |
    |                                  Sentiment Analysis
    |                                       |
    +-------------------+-------------------+
                        |
                        v
              Four Structured Reports
                        |
                        v
              Bullish Researcher
                        |
              Bearish Researcher
                        |
                        v
               Debate and Rebuttal
                        |
                        v
              Trader / Synthesis Stage
                        |
                        v
                 Trade Proposal
                        |
                        v
                 Risk Assessment
                        |
                        v
              Approve / Adjust / Reject
                        |
                        v
                Final Decision
                        |
                        v
             Store and Explain Results
                        |
                        v
             Historical Backtesting
                        |
                        v
          Performance Metrics and Reports
```

The four specialist analyst nodes may continue to execute in parallel.
The diagram shows the logical data dependencies, not necessarily a
strictly sequential implementation of every data retrieval operation.

------------------------------------------------------------------------

# 11. Implementation milestones and work sequence

The recommended sequence is designed to minimize disruption to the
existing working framework.

  -----------------------------------------------------------------------
  Milestone               Work                    Main deliverable
  ----------------------- ----------------------- -----------------------
  M0                      Baseline inspection and Verified baseline and
                          regression testing      issue list

  M1                      Fundamental Analyst     Evidence-backed
                          improvements            fundamental report

  M2                      Technical Analyst       Validated indicators
                          improvements            and technical report

  M3                      News Analyst            Relevant, traceable
                          improvements            news report

  M4                      Sentiment Analyst       Reproducible sentiment
                          improvements            report

  M5                      Four-agent integration  Four validated reports
                                                  reach Trader

  M6                      Structured              Versioned analyst and
                          communication contracts debate schemas

  M7                      Bullish and bearish     Evidence-backed
                          researchers             opposing arguments

  M8                      Debate, rebuttal, and   Explainable trader
                          synthesis               input

  M9                      Risk controls and risk  Validated risk decision
                          perspectives            and deterministic
                                                  limits

  M10                     Backtesting and metric  Reproducible strategy
                          validation              and benchmark results

  M11                     Results reporting and   Complete project
                          documentation           demonstration and
                                                  technical documentation
  -----------------------------------------------------------------------

### Suggested development priority

1.  Complete Objective 2 first.
2.  Complete Objective 3 after the analyst reports are reliable.
3.  Validate and enhance Objective 4 in parallel where safe, then
    complete its full evaluation after the decision workflow stabilizes.
4.  Perform final end-to-end regression testing and documentation.

Do not begin by rewriting the whole graph. Improve and test one module
at a time.

------------------------------------------------------------------------

# 12. Final project acceptance checklist

## Objective 1 --- Framework

-   [ ] Modular multi-agent architecture works.
-   [ ] LangGraph orchestration works.
-   [ ] Shared state correctly collects parallel outputs.
-   [ ] Pydantic contracts validate messages.
-   [ ] Trader and Risk Manager stages execute.
-   [ ] Existing provider fallback and tests remain functional.

## Objective 2 --- Specialist analysis

-   [ ] Fundamental Analyst uses validated company financial data.
-   [ ] Technical Analyst uses validated price and indicator data.
-   [ ] News Analyst uses relevant, dated, traceable news.
-   [ ] Sentiment Analyst produces reproducible sentiment measurements.
-   [ ] All four reports disclose missing data and limitations.
-   [ ] All four reports reach the Trader.
-   [ ] Analytical explanations agree with numerical evidence.

## Objective 3 --- Collaboration and explainability

-   [ ] Structured communication contracts are implemented.
-   [ ] Bullish Researcher is implemented.
-   [ ] Bearish Researcher is implemented.
-   [ ] Rebuttal process is implemented.
-   [ ] Synthesis considers both sides.
-   [ ] Final recommendation is traceable to evidence and debate.

## Objective 4 --- Risk and evaluation

-   [ ] Risk Manager returns validated decisions.
-   [ ] Risk limits are enforced deterministically.
-   [ ] Approve, adjust, and reject branches are tested.
-   [ ] Historical backtesting avoids look-ahead bias.
-   [ ] Cumulative return is calculated and tested.
-   [ ] Sharpe ratio is calculated and tested.
-   [ ] Maximum drawdown is calculated and tested.
-   [ ] Results are compared with a benchmark.
-   [ ] Assumptions and limitations are documented.

## Final system validation

-   [ ] End-to-end analysis works for selected Indian equities.
-   [ ] Missing-data and provider-failure scenarios are tested.
-   [ ] Existing regression tests pass.
-   [ ] Backtesting results are reproducible.
-   [ ] Logs and outputs are understandable.
-   [ ] README and technical documentation are updated.
-   [ ] No live trading is claimed unless a separate, explicitly
    authorized execution system is implemented and validated.

------------------------------------------------------------------------

# 13. Risks, constraints, and assumptions

  -----------------------------------------------------------------------
  Risk / constraint                   Mitigation
  ----------------------------------- -----------------------------------
  Incomplete or unavailable financial Validate fields, disclose gaps,
  data                                reduce confidence

  Limited Indian-market news coverage Use a provider abstraction and test
                                      source coverage

  LLM hallucination                   Supply evidence, validate output,
                                      keep calculations deterministic

  LLM latency and local hardware      Use configurable models, bounded
  limits                              context, and controlled parallelism

  API quotas and failures             Retain fallback logic, retries, and
                                      graceful failure handling

  Historical data leakage             Enforce point-in-time data and
                                      walk-forward testing

  Correlated analyst evidence         Track shared sources and avoid
                                      double-counting the same article

  Overfitting technical models        Use chronological validation and
                                      out-of-sample testing

  Unclear portfolio assumptions       Document cash, holdings, execution,
                                      costs, and sizing rules
  -----------------------------------------------------------------------

### Scope assumptions

-   Initial market scope is Indian equities.
-   Existing market-data and LLM integrations are retained where
    reliable.
-   Local Ollama remains a supported inference option.
-   The system produces analytical recommendations and simulated
    results.
-   Real-time deployment, broker integration, and live order execution
    are not assumed to be part of the current implementation unless
    separately approved.
-   Data provider availability, API terms, and coverage must be verified
    during implementation.

------------------------------------------------------------------------

# 14. Definition of project completion

The project can be considered complete when all four objectives are
implemented and validated together:

1.  A working multi-agent framework coordinates the specialist and
    decision-making agents.
2.  Fundamental, technical, sentiment, and news analysts produce
    reliable, evidence-backed structured reports.
3.  Bullish and bearish agents debate the available evidence through
    structured communication, and the final recommendation is
    explainable.
4.  Risk management controls evaluate and constrain proposed trades, and
    historical backtesting evaluates performance using cumulative
    return, Sharpe ratio, and maximum drawdown.

The final demonstration should show a complete analysis for an Indian
equity ticker, the four analyst reports, the bullish and bearish debate,
the Trader's proposal, the Risk Manager's decision, and the resulting
explanation. It should also show a reproducible historical backtest and
benchmark comparison.

**Implementation principle:** Preserve working code, make incremental
changes, test each milestone, and mark an objective complete only when
its acceptance criteria pass.

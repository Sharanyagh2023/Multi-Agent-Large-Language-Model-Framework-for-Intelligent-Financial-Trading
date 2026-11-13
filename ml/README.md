# Technical analyst and ML direction model

This folder holds the machine-learning evidence used by the technical analyst
([agents/technical_analyst.py](../agents/technical_analyst.py)). The analyst still returns the
shared `AnalystReport` (signal, confidence, summary, key_points), so the trader, risk
manager and workflow are unchanged.

## Flow

```
prices (stock + Nifty 50) ──> TechnicalIndicatorCalculator   data/indicators.py
                          ├─> TechnicalSnapshot (labelled)    data/technical_snapshot.py
                          └─> LightGBM direction model        ml/
                                  │
   snapshot + ML evidence + evidence tally (votes, warnings, confidence_range)
                                  │
                          LLM (structured output: AnalystOpinion)
                                  │
                 guardrail: confidence clipped to confidence_range
                                  │
                            AnalystReport  ──> trader
```

**Structured input.** The LLM never gets a flat list of numbers. It gets three JSON blocks:

1. `TechnicalSnapshot`: trend, momentum, volatility, volume, price levels and relative strength,
   each with a label computed by fixed rules (uptrend/downtrend/sideways, overbought/oversold, ...).
2. `MLDirectionEvidence`: probabilities of down/flat/up, SHAP reasons, and walk-forward quality.
3. Evidence tally: a +1/0/-1 vote per evidence source, warnings, and the allowed confidence range.

**Structured output.** `AnalystOpinion`. Each key point starts with a tag (`[Trend]`,
`[Momentum]`, `[ML model]`, ...) and cites a number.

## The model

| Item | Choice | Why |
|---|---|---|
| Algorithm | LightGBM (gradient-boosted trees) | Strongest standard model on tabular indicator features, trains in seconds, handles mixed scales |
| Target | down / flat / up over the next 5 trading days | "Flat" = move smaller than 0.5 × the stock's normal 5-day move, so labels scale with each stock's volatility and tiny moves aren't counted as signals |
| Training data | Pooled history of ~47 Nifty 50 large caps, 8 years (~92k rows) | One stock alone has too few rows; pooled data lets the model learn general patterns |
| Features | 29 scale-free features: returns, distance from 20/50/200-day averages, ADX/DI, RSI, MACD, stochastic, Bollinger %B and bandwidth, ATR, volatility regime, volume ratio, OBV trend, 52-week position, Nifty 50 trend and the stock's excess return | Same meaning for every stock and price level |
| Validation | Purged walk-forward: 4 expanding folds over the most recent 40% of dates, 5-day gap between train and test | No shuffling and no look-ahead; the gap stops training labels from overlapping the test period |
| Calibration | Temperature scaling, fitted on the most recent 15% of training dates | Makes "60% up" mean roughly 60% |
| Explainability | TreeSHAP via LightGBM `pred_contrib`: top 5 features pushing toward the predicted class | Explainable evidence for the LLM and the report |
| Reliability gate | `useful` if balanced accuracy ≥ chance + 2×`min_edge` and log loss beats the class prior; `weak` if ≥ chance + `min_edge`; else `no_edge` | The LLM is told how much to trust the model; with `no_edge` it gets no vote in the tally |

### Point-in-time safety (live analysis and backtests)

A model trained with cutoff C has only seen prices up to C. Labels need prices 5 days
ahead, so the last labelled training day is 5 trading days before C. For an analysis
on date D, `ModelRegistry` uses the newest saved model with C ≤ D. It retrains with
C = D when that model is more than `max_model_age_days` old. A backtest of 2025
therefore trains its own 2025 models and never uses one trained on later data. Prices
are cached in `ml/artifacts/price_cache/`, so retraining doesn't download again.

## Results (cutoff 2026-11-12, test period 2023-08-23 to 2026-11-04, 37,026 predictions)

| Model | Balanced acc. | Accuracy | Macro F1 | Log loss | Directional acc. |
|---|---|---|---|---|---|
| **LightGBM (used)** | **0.377** | **0.411** | 0.358 | **1.079** | **0.516** |
| Logistic regression | 0.377 | 0.372 | 0.360 | 1.090 | 0.486 |
| Random forest (old RIO model) | 0.369 | 0.366 | 0.360 | 1.094 | 0.483 |
| Class prior (chance) | 0.333 | 0.408 | 0.193 | 1.085 | n/a |

- LightGBM is the only model whose log loss beats the class-prior baseline. Its edge over
  chance (+0.04 balanced accuracy) holds in every fold (0.384 / 0.387 / 0.380 / 0.366).
- The edge is small, so the reliability gate rates the model **weak**. This is expected:
  5-day moves of large-cap stocks are mostly noise. The model is used as one piece of
  evidence, never as the decision.
- The most important features are market-wide (Nifty 50 trend) and volatility regime.
  Stock-specific momentum matters less.
- Design experiment, also on the walk-forward folds:
  - Horizons of 10 and 20 days were not clearly better than 5.
  - Class re-weighting (`class_weight="balanced"`) gave the same balanced accuracy but a
    worse log loss, so it is off.

Regenerate the numbers with `python -m ml.train`. They are also saved in the model card
`ml/artifacts/models/*.json`.

## Commands

```bash
python -m ml.train                       # train with cutoff = today, print the comparison table
python -m ml.train --cutoff 2025-01-01   # model that only knows data up to 2025-01-01
python -m pytest tests -q                # offline tests (synthetic prices, fake LLM)
```

The first `analyze` run trains a model automatically if none is saved (about 1 minute,
mostly downloading prices).

## Limitations

- Survivorship bias: the training universe is today's large caps.
- Yahoo prices are adjusted for later splits and dividends.
- Today's bar is dropped while NSE is open (before 15:30 IST), because volume and close aren't final yet.
- Small local LLMs (e.g. qwen2.5:7b) sometimes misread numbers. That is why the snapshot
  carries ready-made labels and confidence is clipped in Python.

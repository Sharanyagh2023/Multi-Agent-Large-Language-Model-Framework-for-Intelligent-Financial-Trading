"""Technical analyst: what do price, trend, momentum and the ML model say?

Structured input -> LLM -> structured output, like every RIO analyst:

  collect_data()  facts only, no LLM
      1. TechnicalSnapshot      indicators grouped by theme, each with a rule-based label
                                (data/technical_snapshot.py)
      2. MLDirectionEvidence    LightGBM probabilities of down / flat / up over the next
                                5 days, its walk-forward quality and SHAP reasons (ml/)
      3. evidence_tally         a deterministic vote count across 1 and 2, so the LLM
                                starts from an objective summary
  build_prompt()  the three blocks as JSON + reading rules + output rules
  run()           LLM returns AnalystOpinion -> confidence clipped to the tally's
                  confidence_range (plain Python guardrail) -> AnalystReport

The output contract (AnalystReport) is unchanged, so the trader and risk manager
need no changes.
"""
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from agents.base_agent import AnalystAgent
from core.schemas import AnalystOpinion, AnalystReport
from core.state import TradingState
from data.technical_snapshot import TechnicalSnapshot, TechnicalSnapshotBuilder
from ml.predictor import MLDirectionService
from ml.registry import ModelRegistry, ModelUnavailable
from ml.schemas import MLDirectionEvidence
from ml.settings import TechnicalModelSettings

logger = logging.getLogger(__name__)

MIN_PRICE_ROWS = 60          # below this no meaningful technical view is possible
ML_PROBABILITY_MARGIN = 0.10  # P(up) must beat P(down) by this much (or vice versa) to count as a vote

# NSE closes at 15:30 IST (UTC+5:30, no daylight saving). Before that, today's bar is incomplete.
MARKET_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE = time(15, 30)

# Confidence range by agreement of the evidence tally; each warning lowers both ends by 0.1.
CONFIDENCE_BY_AGREEMENT = {
    "unanimous": (0.70, 0.85),
    "mostly aligned": (0.50, 0.70),
    "conflicting": (0.25, 0.50),
    "weak / mostly neutral": (0.25, 0.50),
}
CONFIDENCE_PENALTY_PER_WARNING = 0.10
MIN_CONFIDENCE, MAX_CONFIDENCE = 0.10, 0.85


class TechnicalAnalyst(AnalystAgent):
    name = "technical_analyst"
    role_description = (
        "You read price action: trend, momentum, volatility, volume and strength versus the market index, "
        "and you weigh the output of a machine-learning direction model according to its measured accuracy."
    )

    def __init__(
        self,
        llm,
        market_data,
        price_lookback_days: int,
        model_settings: Optional[TechnicalModelSettings] = None,
        ml_service: Optional[MLDirectionService] = None,
    ) -> None:
        super().__init__(llm, market_data)
        self.price_lookback_days = price_lookback_days
        self.settings = model_settings or TechnicalModelSettings.from_config_file()
        self.snapshot_builder = TechnicalSnapshotBuilder()
        self.ml_service = ml_service or MLDirectionService(ModelRegistry(market_data, self.settings), self.settings)

    # ----- the agent node ---------------------------------------------------
    def run(self, state: TradingState) -> Dict[str, Any]:
        """Same recipe as AnalystAgent.run, plus a deterministic guardrail on the LLM's confidence."""
        ticker, as_of_date = state["ticker"], state["as_of_date"]
        self.logger.info("Analysing %s as of %s", ticker, as_of_date)

        data = self.collect_data(ticker, as_of_date)
        if not data:
            return {"analyst_reports": [self._no_data_report(ticker)]}

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            role=self.name.replace("_", " "),
            role_description=self.role_description,
            market_name=self.market_data.market["name"],
            currency=self.market_data.market["currency"],
        )
        opinion = self.ask_llm(system_prompt, self.build_prompt(ticker, as_of_date, data), AnalystOpinion)
        opinion = self.apply_guardrails(opinion, data["evidence_tally"])
        return {"analyst_reports": [AnalystReport(analyst=self.name, ticker=ticker, **opinion.model_dump())]}

    @staticmethod
    def apply_guardrails(opinion: AnalystOpinion, tally: Dict[str, Any]) -> AnalystOpinion:
        """Keeps confidence inside the range the evidence supports, whatever the LLM said."""
        low, high = tally["confidence_range"]
        confidence = min(max(opinion.confidence, low), high)
        return opinion.model_copy(update={"confidence": round(confidence, 2)})

    # ----- step 1: facts ----------------------------------------------------
    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        prices = self._completed_sessions(
            self.market_data.get_recent_prices(ticker, as_of_date, self.price_lookback_days)
        )
        if len(prices) < MIN_PRICE_ROWS:
            return {}

        benchmark_close = self._completed_sessions(
            self.market_data.get_index_series(
                self.market_data.benchmark_index, as_of_date, lookback_days=self.price_lookback_days
            )
        )
        snapshot = self.snapshot_builder.build(prices, benchmark_close, self.market_data.market["benchmark_name"])

        ml_evidence: Optional[MLDirectionEvidence] = None
        try:
            ml_evidence = self.ml_service.evidence(prices, benchmark_close, as_of_date)
            ml_block: Dict[str, Any] = {"available": True, **ml_evidence.model_dump()}
        except ModelUnavailable as reason:
            ml_block = {"available": False, "reason": str(reason)}
        except Exception as error:  # the ML model is supporting evidence; never let it stop the analyst
            logger.exception("ML direction model failed for %s", ticker)
            ml_block = {"available": False, "reason": f"model error: {error}"}

        return {
            "snapshot": snapshot,
            "ml_direction_model": ml_block,
            "evidence_tally": self.evidence_tally(snapshot, ml_evidence),
        }

    @staticmethod
    def evidence_tally(snapshot: TechnicalSnapshot, ml: Optional[MLDirectionEvidence]) -> Dict[str, Any]:
        """Deterministic +1 / 0 / -1 votes. The ML model only votes when validation showed an edge."""
        votes: Dict[str, int] = {
            "trend": {"uptrend": 1, "downtrend": -1}.get(snapshot.trend.label, 0),
            "momentum": {"bullish": 1, "bearish": -1}.get(snapshot.momentum.label, 0),
        }
        if snapshot.relative_strength is not None:
            votes["relative_strength"] = {"outperforming": 1, "underperforming": -1}.get(
                snapshot.relative_strength.label, 0
            )
        if ml is not None and ml.quality.reliability != "no_edge":
            margin = ml.probabilities.up - ml.probabilities.down
            votes["ml_model"] = 1 if margin > ML_PROBABILITY_MARGIN else -1 if margin < -ML_PROBABILITY_MARGIN else 0

        net = sum(votes.values())
        bullish, bearish = sum(v > 0 for v in votes.values()), sum(v < 0 for v in votes.values())
        if bullish and bearish:
            agreement = "conflicting"
        elif abs(net) == len(votes):
            agreement = "unanimous"
        elif abs(net) >= 2:
            agreement = "mostly aligned"
        else:
            agreement = "weak / mostly neutral"

        warnings = []
        if snapshot.momentum.rsi_zone in ("overbought", "oversold"):
            warnings.append(f"RSI is {snapshot.momentum.rsi_zone} ({snapshot.momentum.rsi_14}): risk of a pullback/bounce.")
        if snapshot.volatility.regime == "high":
            warnings.append("Volatility is high versus its 1-year norm: signals are less reliable.")
        if snapshot.trend.strength == "weak":
            warnings.append("ADX below 20: no strong trend, trend signals are weak.")
        if snapshot.volume.confirms_price_trend is False:
            warnings.append("Volume flow (OBV) disagrees with the 20-day price move.")

        return {
            "votes": votes,
            "net_score": net,
            "max_possible": len(votes),
            "agreement": agreement,
            "warnings": warnings,
            "confidence_range": TechnicalAnalyst.confidence_range(agreement, len(warnings)),
        }

    @staticmethod
    def confidence_range(agreement: str, warning_count: int) -> Tuple[float, float]:
        low, high = CONFIDENCE_BY_AGREEMENT[agreement]
        penalty = CONFIDENCE_PENALTY_PER_WARNING * warning_count
        low = max(MIN_CONFIDENCE, low - penalty)
        high = min(MAX_CONFIDENCE, max(low + 0.1, high - penalty))
        return round(low, 2), round(high, 2)

    @staticmethod
    def _completed_sessions(data):
        """Drops today's bar while the NSE session is still open (its volume and close are not final)."""
        if data is None or len(data) == 0:
            return data
        now = datetime.now(MARKET_TIMEZONE)
        if pd.Timestamp(data.index[-1]).date() == now.date() and now.time() < MARKET_CLOSE:
            return data.iloc[:-1]
        return data

    # ----- step 2: prompt ---------------------------------------------------
    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        horizon = self.settings.horizon_days
        return (
            f"Stock: {ticker}\n"
            f"Analysis date: {as_of_date}\n"
            f"Benchmark: {self.market_data.market['benchmark_name']}\n\n"
            "## 1. Technical snapshot (computed in Python; the labels are already correct, do not re-derive them)\n"
            f"{self.to_json(data['snapshot'])}\n\n"
            f"## 2. ML direction model (probability of down / flat / up over the next {horizon} trading days)\n"
            f"{self.to_json(data['ml_direction_model'])}\n\n"
            "## 3. Evidence tally (+1 bullish, -1 bearish, 0 neutral)\n"
            f"{self.to_json(data['evidence_tally'])}\n\n"
            "## How to read the data\n"
            "- Ratios are fractions: 0.05 = 5%. price_vs_sma_50 = 0.05 means price is 5% above its 50-day average; "
            "pct_from_52w_high = -0.17 means 17% BELOW the 52-week high. position_in_52w_range: 0 = at the low, "
            "1 = at the high. Describe the 52-week position using price_levels.range_label exactly.\n"
            "- trend.label combines price, 50-day and 200-day averages; trend.strength comes from ADX.\n"
            "- ML model: quality.reliability is measured on unseen data. 'useful' = real but modest edge; "
            "'weak' = barely better than chance, mention it but give it little weight; 'no_edge' or available=false "
            "= ignore its direction entirely and say so. Never treat the model as certain; daily stock moves are "
            "mostly noise and balanced accuracy around 0.40 is already good for this task.\n"
            "- top_drivers are SHAP reasons: which indicators pushed the model toward its predicted class.\n\n"
            "## Output rules\n"
            "- signal: bullish if the tally net_score is clearly positive, bearish if clearly negative, otherwise "
            "neutral. You may deviate from the tally only for a reason you state in the summary.\n"
            "- confidence: choose a value inside evidence_tally.confidence_range (it already accounts for the "
            "agreement and the warnings; values outside it are clipped).\n"
            "- summary: 2-3 sentences: the trend, the momentum, and what the ML model adds (or that it was ignored).\n"
            "- key_points: 4-6 items, each starting with a tag [Trend], [Momentum], [Volatility], [Volume], "
            "[Relative strength], [ML model] or [Levels], and each citing at least one number from the data. "
            "Include the nearest support and resistance levels."
        )

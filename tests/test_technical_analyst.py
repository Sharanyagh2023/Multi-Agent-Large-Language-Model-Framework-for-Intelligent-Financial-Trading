"""Offline tests for the technical analyst and its ML model (synthetic prices, fake LLM).

Run from the project folder:  python -m pytest tests -q
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.technical_analyst import TechnicalAnalyst  # noqa: E402
from core.schemas import AnalystOpinion, AnalystReport  # noqa: E402
from data.indicators import TechnicalIndicatorCalculator  # noqa: E402
from data.technical_snapshot import TechnicalSnapshotBuilder  # noqa: E402
from ml.features import FEATURE_COLUMNS, build_feature_frame, build_labels  # noqa: E402
from ml.model import DirectionModel  # noqa: E402
from ml.predictor import MLDirectionService  # noqa: E402
from ml.registry import ModelRegistry  # noqa: E402
from ml.settings import TechnicalModelSettings  # noqa: E402

UNIVERSE = ("AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS")


# ----- fakes -----------------------------------------------------------------
def synthetic_prices(seed: int, days: int = 1400, drift: float = 0.0003, end: str = "2024-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=days)
    returns = drift + 0.015 * rng.standard_normal(days)
    close = 100 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0, 0.008, days)) * close
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, days)),
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, days).astype(float),
        },
        index=index,
    )


class FakeMarketData:
    market = {"name": "Test market", "currency": "INR", "benchmark_name": "Nifty 50", "benchmark_index": "^NSEI"}
    benchmark_index = "^NSEI"

    def __init__(self) -> None:
        self.downloads = 0

    def get_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        self.downloads += 1
        prices = synthetic_prices(seed=zlib.crc32(ticker.encode()))
        return prices.loc[start:end]

    def get_recent_prices(self, ticker: str, as_of: str, lookback_days: int) -> pd.DataFrame:
        start = (pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return self.get_price_history(ticker, start, as_of)

    def get_index_series(self, symbol: str, as_of: str, lookback_days: int = 60) -> pd.Series:
        return self.get_recent_prices(symbol, as_of, lookback_days)["Close"]


class FakeStructuredLLM:
    def __init__(self, schema) -> None:
        self.schema = schema
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.schema(signal="neutral", confidence=0.4, summary="Mixed picture.",
                           key_points=["[Trend] sideways", "[ML model] ignored"])


class FakeLLM:
    def __init__(self) -> None:
        self.last = None

    def with_structured_output(self, schema):
        self.last = FakeStructuredLLM(schema)
        return self.last


@pytest.fixture
def settings(tmp_path) -> TechnicalModelSettings:
    return TechnicalModelSettings(
        training_years=3, min_training_rows=500, max_model_age_days=90,
        artifacts_dir=tmp_path / "artifacts", universe=UNIVERSE,
    )


# ----- indicators and snapshot ---------------------------------------------
def test_indicators_are_in_valid_ranges():
    frame = TechnicalIndicatorCalculator().build_indicator_frame(synthetic_prices(1)).dropna()
    assert len(frame) > 500
    assert frame["rsi_14"].between(0, 100).all()
    assert frame["stochastic_k_14"].between(0, 100).all()
    assert frame["adx_14"].between(0, 100).all()
    assert frame["obv_trend_20"].between(-1, 1).all()
    assert np.isfinite(frame.to_numpy()).all()


def test_features_never_use_future_prices():
    prices = synthetic_prices(2)
    cut = 900
    full = build_feature_frame(prices, prices["Close"] * 1.1)
    truncated = build_feature_frame(prices.iloc[:cut], prices["Close"].iloc[:cut] * 1.1)
    pd.testing.assert_series_equal(full.iloc[cut - 1], truncated.iloc[-1], check_names=False)


def test_labels_use_volatility_scaled_flat_band():
    close = pd.Series(100.0 * np.exp(np.cumsum(np.r_[0, np.tile([0.01, -0.01], 30)])))
    close.iloc[-6:] = close.iloc[-7] * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.2])  # big jump at the very end
    labels = build_labels(close, horizon=5, flat_multiple=0.5)
    assert labels["label"].iloc[-5:].isna().all()            # future unknown -> no label
    assert labels["label"].iloc[-6] == 2                     # +20% in 5 days -> up
    assert set(labels["label"].dropna().unique()) <= {0.0, 1.0, 2.0}


def test_snapshot_reads_a_clear_uptrend():
    prices = synthetic_prices(3, drift=0.004)
    snapshot = TechnicalSnapshotBuilder().build(prices, prices["Close"] * 0 + 100, "Nifty 50")
    assert snapshot.trend.label == "uptrend"
    assert snapshot.trend.sma_50_above_sma_200 is True
    assert snapshot.relative_strength.label == "outperforming"


# ----- model -------------------------------------------------------------------
def test_direction_model_learns_calibrates_explains_and_saves(tmp_path):
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.standard_normal((6000, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    signal = features["rsi_14"] + 0.5 * rng.standard_normal(6000)
    labels = pd.Series(np.select([signal > 0.5, signal < -0.5], [2, 0], 1))

    model = DirectionModel().fit(features.iloc[:5000], labels.iloc[:5000])
    temperature = model.calibrate(features.iloc[5000:], labels.iloc[5000:])
    assert 0.25 <= temperature <= 10

    probabilities = model.predict_proba(features.iloc[5000:])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (model.predict(features.iloc[5000:]) == labels.iloc[5000:]).mean() > 0.6

    drivers = model.explain(features.iloc[[0]], class_index=2, top_n=3)
    assert len(drivers) == 3 and drivers[0]["feature"] == "rsi_14"

    model.save(tmp_path / "m.joblib")
    reloaded = DirectionModel.load(tmp_path / "m.joblib")
    assert np.allclose(reloaded.predict_proba(features.iloc[:5]), model.predict_proba(features.iloc[:5]))


# ----- registry: point-in-time model selection -----------------------------------
def test_registry_trains_reuses_and_never_uses_a_future_model(settings):
    market_data = FakeMarketData()
    registry = ModelRegistry(market_data, settings)

    model = registry.get_model("2024-06-28")
    assert model.card["trained_until"] == "2024-06-28"
    assert model.card["last_labelled_date"] < "2024-06-28"
    assert {"lightgbm", "class_prior", "logistic_regression"} <= set(model.card["validation"]["models"])

    downloads = market_data.downloads
    assert registry.get_model("2024-08-15") is model          # 48 days old: re-used, nothing downloaded
    assert market_data.downloads == downloads

    older = registry.get_model("2024-01-31")                   # earlier date: must NOT use the June model
    assert older.card["trained_until"] == "2024-01-31"
    assert len(registry.saved_models()) == 2


# ----- agent -------------------------------------------------------------------
def test_technical_analyst_produces_structured_report(settings):
    market_data = FakeMarketData()
    service = MLDirectionService(ModelRegistry(market_data, settings), settings)
    llm = FakeLLM()
    analyst = TechnicalAnalyst(llm, market_data, price_lookback_days=730, model_settings=settings, ml_service=service)

    data = analyst.collect_data("AAA.NS", "2024-06-28")
    assert data["ml_direction_model"]["available"] is True
    evidence = data["ml_direction_model"]
    assert abs(sum(evidence["probabilities"].values()) - 1) < 0.01
    assert len(evidence["top_drivers"]) == settings.top_drivers
    assert set(data["evidence_tally"]["votes"]) >= {"trend", "momentum", "relative_strength"}

    result = analyst.run({"ticker": "AAA.NS", "as_of_date": "2024-06-28", "analyst_reports": []})
    report = result["analyst_reports"][0]
    assert isinstance(report, AnalystReport) and report.analyst == "technical_analyst"
    prompt = llm.last.messages[1].content
    assert "## 1. Technical snapshot" in prompt and "## 2. ML direction model" in prompt
    assert llm.last.schema is AnalystOpinion


def test_confidence_guardrail_clips_overconfident_llm():
    low, high = TechnicalAnalyst.confidence_range("mostly aligned", warning_count=2)
    assert (low, high) == (0.3, 0.5)
    assert TechnicalAnalyst.confidence_range("unanimous", warning_count=0) == (0.7, 0.85)
    assert TechnicalAnalyst.confidence_range("conflicting", warning_count=4) == (0.1, 0.2)

    opinion = AnalystOpinion(signal="bullish", confidence=0.95, summary="s", key_points=["k"])
    clipped = TechnicalAnalyst.apply_guardrails(opinion, {"confidence_range": (low, high)})
    assert clipped.confidence == 0.5 and clipped.signal == "bullish"


def test_technical_analyst_without_history_gives_no_view(settings):
    class ShortHistory(FakeMarketData):
        def get_recent_prices(self, ticker, as_of, lookback_days):
            return super().get_recent_prices(ticker, as_of, lookback_days).tail(30)

    analyst = TechnicalAnalyst(FakeLLM(), ShortHistory(), 730, model_settings=settings)
    report = analyst.run({"ticker": "AAA.NS", "as_of_date": "2024-06-28", "analyst_reports": []})["analyst_reports"][0]
    assert report.confidence == 0.0 and report.signal == "neutral"

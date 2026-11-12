import math
import logging
from unittest import result
import pytest
from agents.fundamental_analyst import FundamentalAnalyst
from core.schemas import AnalystOpinion

def test_validate_fundamental_data():

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    data = {
        "longName": "Example Ltd",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 500000000,
        "trailingPE": 24.5,
        "forwardPE": 20.3,
        "priceToBook": None,
        "profitMargins": 0.18,
        "operatingMargins": 0.15,
        "returnOnEquity": 0.20,
        "revenueGrowth": "0.125",
        "earningsGrowth": -0.05,
        "debtToEquity": 45.0,
        "currentRatio": 1.8,
        "freeCashflow": 25000000,
        "dividendYield": 0.02,
        "beta": float("nan"),
    }

    result = analyst.validate_data(data)

    assert result["longName"] == "Example Ltd"
    assert result["sector"] == "Technology"

    assert result["trailingPE"] == 24.5
    assert result["revenueGrowth"] == 0.125
    assert result["earningsGrowth"] == -0.05

    assert result["priceToBook"] is None
    assert result["beta"] is None

    assert "priceToBook" in result["_data_quality"]["missing_metrics"]
    assert "beta" in result["_data_quality"]["invalid_metrics"]

    assert result["_data_quality"]["total_metrics"] == 14
    assert result["_data_quality"]["available_metrics"] == 12


#empty-data protection test
def test_empty_fundamental_data_skips_llm():

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    class MockMarketData:
        def get_fundamentals(self, ticker):
            return {}

    analyst.market_data = MockMarketData()
    analyst.logger = logging.getLogger("test_fundamental_analyst")

    # The LLM must never be called when no data is available.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called when fundamental data is empty")

    analyst.ask_llm = fail_if_called

    state = {
        "ticker": "RELIANCE.NS",
        "as_of_date": "2026-11-11",
    }

    result = analyst.run(state)

    report = result["analyst_reports"][0]

    assert report.signal == "neutral"
    assert report.confidence == 0.0
    assert report.key_points == []
    assert "No data" in report.summary

#partial-data test
def test_partial_fundamental_data_is_preserved():

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    class MockMarketData:
        def get_fundamentals(self, ticker):
            return {
                "longName": "Example Ltd",
                "sector": "Technology",
                "trailingPE": 22.5,
                "profitMargins": 0.18,
            }

    analyst.market_data = MockMarketData()

    result = analyst.collect_data("EXAMPLE.NS", "2026-11-11")

    assert result["longName"] == "Example Ltd"
    assert result["sector"] == "Technology"
    assert result["trailingPE"] == 22.5
    assert result["profitMargins"] == 0.18

    quality = result["_data_quality"]

    assert "forwardPE" in quality["missing_metrics"]
    assert "returnOnEquity" in quality["missing_metrics"]

    assert quality["available_metrics"] == 2
    assert quality["total_metrics"] == 14

def test_partial_fundamental_data_generates_analyst_report():

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    class MockMarketData:
        market = {
            "name": "Indian Stock Market",
            "currency": "INR",
        }

        def get_fundamentals(self, ticker):
            return {
                "longName": "Example Ltd",
                "sector": "Technology",
                "trailingPE": 22.5,
                "profitMargins": 0.18,
            }

    analyst.market_data = MockMarketData()
    analyst.logger = logging.getLogger("test_fundamental_analyst")

    llm_called = {"value": False}

    def mock_ask_llm(system_prompt, user_prompt, output_schema):
        llm_called["value"] = True

        assert output_schema == AnalystOpinion
        assert "Example Ltd" in user_prompt
        assert "trailingPE" in user_prompt
        assert "profitMargins" in user_prompt
        assert "ONLY the supplied financial data" in user_prompt
        assert "Every observation must be traceable to the supplied data" in user_prompt

        return AnalystOpinion(
            signal="bullish",
            confidence=0.82,
            summary="The available fundamentals indicate positive profitability.",
            key_points=[
                "Profit margin is 18%.",
                "Trailing P/E is 22.5.",
            ],
        )

    analyst.ask_llm = mock_ask_llm

    state = {
        "ticker": "EXAMPLE.NS",
        "as_of_date": "2026-11-11",
    }

    result = analyst.run(state)

    # Verify that the LLM was called.
    assert llm_called["value"] is True

    # Verify the generated analyst report.
    assert len(result["analyst_reports"]) == 1

    report = result["analyst_reports"][0]

    assert report.analyst == "fundamental_analyst"
    assert report.ticker == "EXAMPLE.NS"
    assert report.signal == "bullish"
    assert report.confidence == 0.50
    assert "Example Ltd" in report.summary
    assert "2 numerical financial metrics" in report.summary

    assert any(
        "Trailing P/E: 22.50" in point
        for point in report.key_points
    )

    assert any(
        "Profit margin: 18.00%" in point
        for point in report.key_points
    )

def test_fundamental_prompt_contains_strict_analysis_rules():

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    class MockMarketData:
        market = {
            "name": "Indian Stock Market",
            "currency": "INR",
        }

    analyst.market_data = MockMarketData()

    data = {
        "longName": "Example Ltd",
        "sector": "Technology",
        "trailingPE": 22.5,
        "profitMargins": 0.18,
    }

    prompt = analyst.build_prompt(
        "EXAMPLE.NS",
        "2026-11-12",
        data,
    )

    assert "Configured market currency: INR" in prompt
    assert "supplied data" in prompt.lower()
    assert "do not invent" in prompt.lower()
    assert "market capitalization" in prompt.lower()
    assert "company size" in prompt.lower()
    assert "without appropriate benchmark data" in prompt
    assert "debt-to-equity" in prompt.lower()
    assert "high" in prompt.lower()
    assert "low" in prompt.lower()
    assert "Every interpretation must be supported by supplied data" in prompt


@pytest.mark.parametrize(
    "claim",
    [
        "The P/E and P/B ratios suggest overvaluation.",
        "The stock appears overvalued.",
        "The valuation indicates undervaluation.",
        "The company has high debt.",
        "The debt-to-equity ratio is high.",
        "Positive FCF supports debt servicing.",
        "Market capitalization and low beta indicate stability.",
        "Sector-specific growth expectations justify the valuation.",
    ],
)
def test_unsupported_fundamental_claims_are_removed(claim):

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    opinion = AnalystOpinion(
        signal="bullish",
        confidence=0.85,
        summary="The company has positive fundamentals.",
        key_points=[
            claim,
            "Revenue growth is 10%, based on the supplied data.",
        ],
    )

    data = {
        "trailingPE": 22.5,
        "profitMargins": 0.18,
        "revenueGrowth": 0.10,
        "earningsGrowth": 0.12,
        "_data_quality": {
            "available_metrics": 4,
            "total_metrics": 17,
            "missing_metrics": [],
            "invalid_metrics": [],
        },
    }

    result = analyst.validate_opinion(opinion, data)

    assert claim not in result.key_points

    assert any(
        "Revenue growth" in point and "10.00%" in point
        for point in result.key_points
    )

    assert result.confidence <= 0.40

def test_unsupported_summary_is_replaced():

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    opinion = AnalystOpinion(
        signal="neutral",
        confidence=0.80,
        summary="The company's P/E suggests overvaluation.",
        key_points=[
            "Profit margin is 18%."
        ],
    )

    data = {
        "trailingPE": 22.5,
        "profitMargins": 0.18,
        "revenueGrowth": 0.10,
        "earningsGrowth": 0.12,
    }

    result = analyst.validate_opinion(opinion, data)

    assert "overvaluation" not in result.summary.lower()
    assert result.confidence <= 0.40


@pytest.mark.parametrize(
    "claim",
    [
        "ROE indicates robust profitability.",
        "Debt-to-equity is elevated.",
        "Positive FCF supports financial resilience.",
        "The company has a healthy balance sheet.",
        "The company has a strong financial position.",
    ],
)
def test_unsupported_financial_claims_are_removed(claim):

    analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

    opinion = AnalystOpinion(
        signal="neutral",
        confidence=0.80,
        summary="The company has mixed financial performance.",
        key_points=[claim],
    )

    data = {
        "trailingPE": 22.5,
        "profitMargins": 0.18,
        "returnOnEquity": 0.0876,
        "debtToEquity": 44.325,
        "freeCashflow": 1190000000000,
        "revenueGrowth": 0.0298,
        "earningsGrowth": 0.0407,
    }

    result = analyst.validate_opinion(opinion, data)

    assert claim not in result.key_points
    assert len(result.key_points) > 0
    assert result.confidence <= 0.40

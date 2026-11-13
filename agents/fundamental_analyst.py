"""Fundamental analyst: is the company healthy and fairly priced?"""
import math
from typing import Any, Dict

from core.schemas import AnalystOpinion

from agents.base_agent import AnalystAgent
from data.market_data import (
    METADATA_FIELDS,
    COMMON_FUNDAMENTAL_FIELDS,
    NON_BANK_FUNDAMENTAL_FIELDS,
    BANK_FUNDAMENTAL_FIELDS,
)

class FundamentalAnalyst(AnalystAgent):
    name = "fundamental_analyst"
    role_description = (
        "You are responsible for evaluating the financial health, business quality, "
        "growth potential and valuation of companies. Analyze valuation, profitability, "
        "revenue and earnings growth, balance-sheet strength, liquidity and cash flow. "
        "Consider the company's sector and industry when interpreting financial ratios. "
        "Distinguish between attractive business fundamentals and attractive stock valuation. "
        "Never treat missing financial information as zero or assume that an unavailable "
        "metric indicates poor financial performance."
    )

    
    def get_expected_fields(self, data: Dict[str, Any]) -> list:
        """Return fields applicable to the company's sector."""

        sector = data.get("sector")
        industry = data.get("industry") or ""

        is_bank = (
            sector == "Financial Services"
            and "bank" in industry.lower()
        )

        if is_bank:
            return (
                METADATA_FIELDS
                + COMMON_FUNDAMENTAL_FIELDS
                + BANK_FUNDAMENTAL_FIELDS
            )

        return (
            METADATA_FIELDS
            + COMMON_FUNDAMENTAL_FIELDS
            + NON_BANK_FUNDAMENTAL_FIELDS
        )


    
    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        raw_data = self.market_data.get_fundamentals(ticker)

        print("\nDEBUG - RAW FUNDAMENTAL DATA:")
        print(self.to_json(raw_data))

        if not raw_data:
            return {}

        validated_data = self.validate_data(raw_data)

        expected_fields = self.get_expected_fields(validated_data)
        numeric_fields = [
            field for field in expected_fields
            if field not in METADATA_FIELDS
        ]

        available_numeric_metrics = [
            field for field in numeric_fields
            if validated_data.get(field) is not None
        ]

        if not available_numeric_metrics:
            return {}

        return validated_data
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate only the fundamental fields applicable to the company."""

        validated_data = {}
        missing_metrics = []
        invalid_metrics = []

        expected_fields = self.get_expected_fields(data)

        for metric in expected_fields:
            value = data.get(metric)

            if value is None:
                validated_data[metric] = None
                missing_metrics.append(metric)
                continue

            if metric in METADATA_FIELDS:
                validated_data[metric] = value
                continue

            if isinstance(value, bool):
                validated_data[metric] = None
                invalid_metrics.append(metric)
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                validated_data[metric] = None
                invalid_metrics.append(metric)
                continue

            if not math.isfinite(numeric_value):
                validated_data[metric] = None
                invalid_metrics.append(metric)
                continue

            validated_data[metric] = numeric_value

        numeric_fields = [
            field for field in expected_fields
            if field not in METADATA_FIELDS
        ]

        available_metrics = sum(
            validated_data.get(field) is not None
            for field in numeric_fields
        )

        validated_data["_data_quality"] = {
            "missing_metrics": missing_metrics,
            "invalid_metrics": invalid_metrics,
            "available_metrics": available_metrics,
            "total_metrics": len(numeric_fields),
        }

        return validated_data


    
    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        currency = self.market_data.market["currency"]

        return f"""
FUNDAMENTAL ANALYSIS REQUEST

Company/Ticker: {ticker}
Analysis Date: {as_of_date}
Configured market currency: {currency}

FINANCIAL DATA:
{self.to_json(data)}

YOUR TASK:

Evaluate the company's fundamentals using ONLY the supplied financial data.

Your objective is to produce a factual, evidence-based fundamental
assessment. Every observation must be traceable to the supplied data.

1. VALUATION
- Report available trailing P/E, forward P/E and price-to-book ratios.
- Explain what these ratios measure.
- Do not classify valuation as expensive, cheap, attractive,
  overvalued or undervalued without appropriate benchmark data.
- Do not infer future growth from valuation ratios alone.

2. PROFITABILITY
- Report available profit margins and operating margins.
- Report return on equity.
- Explain what each metric measures.
- Do not classify profitability as strong, weak, robust or poor
  without historical or industry benchmark data.

3. GROWTH
- Report revenue growth and earnings growth separately.
- Identify whether each reported growth value is positive or negative.
- Do not infer operational efficiency or business performance
  from differences between these two values alone.

4. FINANCIAL HEALTH
- Report debt-to-equity, current ratio and free cash flow
  when available.
- Explain what each metric measures.
- Identify possible areas requiring further investigation.
- Do not classify debt as high, low, excessive or manageable
  without appropriate historical or industry benchmarks.
- Do not claim that free cash flow is sufficient to service debt
  without supporting debt obligations and cash-flow information.

5. MARKET CHARACTERISTICS
- Report market capitalization, beta and dividend yield
  when available.
- Explain what each metric measures.
- Market capitalization represents company size.
- Beta represents sensitivity to market movements.
- Dividend yield represents dividend income relative to share price.
- Do not infer company stability, safety or investor confidence
  from these metrics alone.

6. SECTOR CONTEXT
- Identify the supplied sector and industry when available.
- Do not invent sector averages or competitor comparisons.
- Do not assume that a metric is high or low for a sector
  without benchmark data.
- For banks, prioritize ROE and ROA when available.
- Do not interpret debt-to-equity, current ratio or free cash flow
  as ordinary corporate financial-health indicators for banks.
- Do not invent NIM, NPA, CASA or capital adequacy values.
- Discuss bank-specific metrics only when they are supplied.

STRICT DATA INTERPRETATION RULES:

1. NUMERICAL ACCURACY
- Use only numerical values present in the supplied data.
- Do not invent, estimate or modify financial figures.
- Preserve the original numerical precision where practical.
- Do not replace missing values with zero.

2. CURRENCY
- Use the configured market currency: {currency}.
- Never substitute USD or another currency.
- Do not perform currency conversions without an exchange rate.
- If the currency of a monetary value is uncertain,
  explicitly acknowledge the uncertainty.

3. METRIC INTERPRETATION
- Profit margins, operating margins, return on equity,
  return on assets, revenue growth and earnings growth
  are represented as decimal ratios in the supplied data.
- Dividend yield from the tested Yahoo Finance data is
  already expressed as a percentage. Do not multiply it by 100.
- Debt-to-equity is a source-provided ratio. Preserve its value.
- Example: 0.15 represents 15%.
- Debt-to-equity is generally represented on a
  percentage-style scale by Yahoo Finance.
- P/E and P/B are ratios.
- Beta is dimensionless.
- Market capitalization and free cash flow are monetary values.

4. MISSING DATA
- Identify unavailable metrics explicitly when relevant.
- Do not infer missing historical information.
- Reduce confidence when important financial metrics are missing.

5. EVIDENCE
- Distinguish numerical observations from interpretations.
- Every interpretation must be supported by supplied data.
- Do not make claims about historical performance
  when historical data is not supplied.
- Do not infer causation from correlation or numerical differences.

OUTPUT STRUCTURE:

Provide:

1. Signal: bullish, neutral or bearish.
2. Confidence: a value between 0 and 1.
3. Summary: a concise, factual overview of the available fundamentals.
4. Key points: 3–5 concise, independent observations.

IMPORTANT KEY-POINT RULES:

- Each key point should contain one main observation.
- Prefer reporting a metric and its exact value.
- Do not combine supported numerical observations
  with unsupported financial conclusions.
- Avoid promotional or subjective language.
- If an interpretation cannot be established from the supplied data,
  state that additional evidence is required.
- Do not repeat the same observation in multiple key points.

FINAL PRINCIPLE:

A company's business quality and its stock valuation are different
questions. Do not assume that positive financial metrics automatically
make a stock attractive.

Return only the structured AnalystOpinion response.
"""
    
    def validate_opinion(
        self,
        opinion: AnalystOpinion,
        data: Dict[str, Any],
    ) -> AnalystOpinion:
        """Build evidence-based fundamental observations from source data."""

        import math
        import re

        def is_valid_number(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        def format_number(value: float, decimals: int = 2) -> str:
            return f"{value:,.{decimals}f}"

        # Detect unsupported qualitative conclusions in the original LLM response.
        # These patterns are deliberately conservative.
        unsupported_patterns = [
            r"\bovervalu(?:ed|ation)\b",
            r"\bundervalu(?:ed|ation)\b",
            r"\b(?:high|low|elevated|excessive)\s+debt\b",
            r"\bdebt[- ]to[- ]equity\b.{0,30}\b(?:high|low|elevated|excessive)\b",
            r"\b(?:robust|strong)\s+profitability\b",
            r"\bhealthy\s+(?:balance\s+sheet|financial\s+position)\b",
            r"\bstrong\s+financial\s+position\b",
            r"\bfinancial\s+resilience\b",
            r"\b(?:supports?|indicates?)\s+(?:debt servicing|financial resilience|stability)\b",
            r"\b(?:low beta|high beta)\b.{0,60}\b(?:stability|stable|risk)\b",
            r"\b(?:growth expectations|sector growth)\b.{0,60}\b(?:justify|justifies|justified)\b.{0,30}\bvaluation\b",
            r"\b(?:suggests?|indicates?)\s+(?:overvaluation|undervaluation)\b",
            r"\b(?:strong|weak|healthy|poor)\s+(?:fundamentals|financial performance|balance sheet)\b",
        ]

        original_text = " ".join(
            [opinion.summary or ""]
            + [str(point) for point in (opinion.key_points or [])]
        ).lower()

        has_unsupported_claim = any(
            re.search(pattern, original_text, re.IGNORECASE)
            for pattern in unsupported_patterns
        )

        # Fields whose source values are decimal ratios representing percentages.
        percentage_fields = {
            "profitMargins": "Profit margin",
            "operatingMargins": "Operating margin",
            "returnOnEquity": "Return on equity (ROE)",
            "returnOnAssets": "Return on assets (ROA)",
            "revenueGrowth": "Revenue growth",
            "earningsGrowth": "Earnings growth",
        }

        # Ordinary numerical observations: no qualitative classification.
        numeric_fields = {
            "trailingPE": ("Trailing P/E", 2),
            "forwardPE": ("Forward P/E", 2),
            "priceToBook": ("Price-to-book ratio", 2),
            "debtToEquity": ("Debt-to-equity ratio (source value)", 2),
            "currentRatio": ("Current ratio", 3),
            "beta": ("Beta", 3),
        }

        key_points = []
        available_numeric_metrics = 0

        # Percentage metrics.
        for field, label in percentage_fields.items():
            value = data.get(field)

            if is_valid_number(value):
                available_numeric_metrics += 1
                percentage = value * 100
                key_points.append(
                    f"{label}: {format_number(percentage)}%"
                )

        # Other numerical metrics.
        for field, (label, decimals) in numeric_fields.items():
            value = data.get(field)

            if is_valid_number(value):
                available_numeric_metrics += 1
                key_points.append(
                    f"{label}: {format_number(value, decimals)}"
                )

        # Large financial values are shown as full source numbers.
        for field, label in [
            ("marketCap", "Market capitalization (source value)"),
            ("freeCashflow", "Free cash flow (source value)"),
        ]:
            value = data.get(field)

            if is_valid_number(value):
                available_numeric_metrics += 1
                key_points.append(
                    f"{label}: {value:,.0f}"
                )

        # Dividend yield is already expressed as a percentage by Yahoo Finance.
        dividend_yield = data.get("dividendYield")

        if is_valid_number(dividend_yield):
            available_numeric_metrics += 1
            key_points.append(
                f"Dividend Yield: {format_number(dividend_yield)}%"
            )

        # Add available company metadata.
        company = data.get("longName")
        sector = data.get("sector")
        industry = data.get("industry")

        metadata = []
        if company:
            metadata.append(str(company))
        if sector:
            metadata.append(f"Sector: {sector}")
        if industry:
            metadata.append(f"Industry: {industry}")

        company_description = " | ".join(metadata)

        if key_points:
            summary = (
                f"{company_description}. "
                f"{available_numeric_metrics} numerical financial metrics "
                "were available. The observations below are taken directly "
                "from the supplied financial data; qualitative classifications "
                "and investment conclusions are not inferred here."
            )
        else:
            summary = (
                f"{company_description}. "
                "No valid numerical financial metrics were available "
                "for this analysis."
            )

        # Apply confidence limits.
        confidence = opinion.confidence

        if available_numeric_metrics < 4:
            confidence = min(confidence, 0.50)

        if has_unsupported_claim:
            confidence = min(confidence, 0.40)

        if not key_points:
            confidence = min(confidence, 0.30)

        return opinion.model_copy(
            update={
                "summary": summary,
                "key_points": key_points,
                "confidence": confidence,
            }
        )

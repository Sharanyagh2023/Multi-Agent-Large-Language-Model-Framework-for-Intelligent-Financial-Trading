"""Fundamental analyst: is the company healthy and fairly priced?"""
from typing import Any, Dict

from agents.base_agent import AnalystAgent


class FundamentalAnalyst(AnalystAgent):
    name = "fundamental_analyst"
    role_description = (
        "You judge the quality and valuation of the business from its financial ratios: "
        "valuation, profitability, growth and balance-sheet strength."
    )

    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        return self.market_data.get_fundamentals(ticker)

    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        return (
            f"Stock: {ticker}\n"
            f"Date: {as_of_date}\n\n"
            f"Fundamental data (ratios are as reported by the data provider):\n{self.to_json(data)}\n\n"
            "Assess valuation (P/E, P/B), profitability (margins, ROE), growth (revenue, earnings) "
            "and financial risk (debt, liquidity). Note: debtToEquity and dividendYield are percentages (debtToEquity 100 = debt equals equity). "
            "Judge ratios against typical levels for Indian companies in the same sector."
        )

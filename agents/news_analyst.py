"""News analyst: which events could move the stock?

Where the sentiment analyst measures tone with a model, this analyst *reads*
the headlines to find catalysts and risks (earnings, products, regulation,
macro events) and checks how the overall market has been behaving.
"""
from typing import Any, Dict

from agents.base_agent import AnalystAgent


class NewsAnalyst(AnalystAgent):
    name = "news_analyst"
    role_description = (
        "You read recent headlines and market context to identify concrete catalysts and risks."
    )

    def __init__(self, llm, market_data, news_limit: int) -> None:
        super().__init__(llm, market_data)
        self.news_limit = news_limit

    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        articles = self.market_data.get_news(ticker, as_of_date, self.news_limit)
        if not articles:
            return {}

        data: Dict[str, Any] = {"articles": articles}
        benchmark = self.market_data.get_index_series(self.market_data.benchmark_index, as_of_date, lookback_days=45)
        if len(benchmark) >= 21:
            benchmark_name = self.market_data.market["benchmark_name"]
            data[f"{benchmark_name}_return_last_20_days"] = round(float(benchmark.iloc[-1] / benchmark.iloc[-21] - 1), 4)
        return data

    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        return (
            f"Stock: {ticker}\n"
            f"Date: {as_of_date}\n\n"
            f"Recent news and market context:\n{self.to_json(data)}\n\n"
            "Identify the most important catalysts and risks. Separate company-specific news from "
            "general market conditions (RBI policy, FII flows, budget, global cues, etc. where relevant)."
        )

"""Technical analyst: what do price, trend and momentum say?

Uses two evidence sources: classic indicators (RSI, MACD, moving averages ...)
and a scikit-learn model that estimates the 5-day probability of a rise.
"""
from typing import Any, Dict

from agents.base_agent import AnalystAgent
from data.indicators import TechnicalIndicatorCalculator
from data.ml_models import DirectionPredictor


class TechnicalAnalyst(AnalystAgent):
    name = "technical_analyst"
    role_description = (
        "You read price action: trend, momentum, volatility and volume, "
        "and you weigh the output of a statistical direction model."
    )

    def __init__(self, llm, market_data, price_lookback_days: int) -> None:
        super().__init__(llm, market_data)
        self.price_lookback_days = price_lookback_days
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.direction_predictor = DirectionPredictor()

    def collect_data(self, ticker: str, as_of_date: str) -> Dict[str, Any]:
        prices = self.market_data.get_recent_prices(ticker, as_of_date, self.price_lookback_days)
        if len(prices) < 60:
            return {}
        return {
            "indicators": self.indicator_calculator.latest_summary(prices),
            "ml_direction_model": self.direction_predictor.predict(prices),
        }

    def build_prompt(self, ticker: str, as_of_date: str, data: Dict[str, Any]) -> str:
        return (
            f"Stock: {ticker}\n"
            f"Date: {as_of_date}\n\n"
            f"Latest indicators:\n{self.to_json(data['indicators'])}\n\n"
            f"ML direction model (random forest trained on this stock's own history):\n"
            f"{self.to_json(data['ml_direction_model'] or 'not enough history')}\n\n"
            "Guidance: price above sma_50 / sma_200 suggests an uptrend; rsi_14 above 70 is overbought and "
            "below 30 oversold; bollinger_pct_b near 1 / 0 means price is at the upper / lower band. "
            "Trust the ML model only to the extent its validation_accuracy beats always_predict_up_accuracy."
        )

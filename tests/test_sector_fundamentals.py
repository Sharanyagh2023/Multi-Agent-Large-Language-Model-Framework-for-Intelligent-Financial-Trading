from datetime import datetime

from data.market_data import MarketDataService
import yaml

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

market = config["markets"]["india"]

service = MarketDataService(market)

tickers = ["HDFCBANK.NS", "TCS.NS", "RELIANCE.NS"]

for ticker in tickers:
    print("\n" + "=" * 60)
    print(f"FUNDAMENTALS: {ticker}")
    print("=" * 60)

    data = service.get_fundamentals(
    ticker,
    datetime.now().strftime("%Y-%m-%d"),
)

    for key, value in data.items():
        print(f"{key}: {value}")

    print("\nCollected metrics:", len(data))

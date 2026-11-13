from data.market_data import MarketDataService


def test_historical_fundamentals_return_empty():

    market = {
        "name": "Indian Stock Market",
        "currency": "INR",
        "default_suffix": ".NS",
        "exchange_suffixes": [".NS", ".BO"],
        "benchmark_index": "^NSEI",
        "fear_index": "^INDIAVIX",
    }

    service = MarketDataService(
        market=market,
        point_in_time=False,
    )

    result = service.get_fundamentals(
        "RELIANCE.NS",
        "2020-01-01",
    )

    assert result == {}
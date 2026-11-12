
import math

from agents.fundamental_analyst import FundamentalAnalyst


# Create an instance without initializing external LLM dependencies.
analyst = FundamentalAnalyst.__new__(FundamentalAnalyst)

test_data = {
    "longName": "Test Bank",
    "sector": "Financial Services",
    "industry": "Banks - Regional",

    "marketCap": 1000000,
    "trailingPE": float("nan"),
    "forwardPE": float("inf"),
    "priceToBook": True,
    "returnOnEquity": "invalid",
    "dividendYield": 1.5,
    "beta": 0.5,
    "returnOnAssets": 0.02,
}

result = analyst.validate_data(test_data)

print("\nVALIDATION RESULT")
print("=" * 50)

for key, value in result.items():
    print(f"{key}: {value}")

quality = result["_data_quality"]

# Invalid values must be rejected.
assert result["trailingPE"] is None
assert result["forwardPE"] is None
assert result["priceToBook"] is None
assert result["returnOnEquity"] is None

# Valid values must be preserved.
assert result["marketCap"] == 1000000.0
assert result["dividendYield"] == 1.5
assert result["beta"] == 0.5
assert result["returnOnAssets"] == 0.02

# Verify invalid metric tracking.
assert set(quality["invalid_metrics"]) == {
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
}

assert quality["available_metrics"] == 4
assert quality["total_metrics"] == 8

print("\nALL INVALID-DATA TESTS PASSED")

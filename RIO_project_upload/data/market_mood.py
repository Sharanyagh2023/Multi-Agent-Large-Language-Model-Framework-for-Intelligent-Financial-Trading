
"""
Market-wide mood indicators for Indian equities.

Uses the existing MarketDataService for India VIX.
PCR, advance/decline and FII/DII flows are reported as
unavailable until reliable data providers are integrated.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def collect_market_mood(
    market_data: Any,
    as_of: str,
    lookback_days: int = 30,
) -> Dict[str, Any]:
    """
    Collect market-wide mood indicators using data available
    on or before the requested date.

    Returns an empty dictionary when no usable market data exists.
    """

    result: Dict[str, Any] = {
        "as_of": as_of,
        "available_indicators": [],
        "unavailable_indicators": [
            "put_call_ratio",
            "advance_decline",
            "fii_dii_flows",
        ],
    }

    # India VIX
    try:
        vix = market_data.get_index_series(
            market_data.fear_index,
            as_of,
            lookback_days=lookback_days,
        )

        if vix is not None:
            vix = vix.dropna()

        if vix is not None and not vix.empty:
            latest_value = float(vix.iloc[-1])

            thresholds = market_data.market
            fear_threshold = float(
                thresholds.get("fear_index_high", 20)
            )
            complacency_threshold = float(
                thresholds.get("fear_index_low", 13)
            )

            if latest_value > fear_threshold:
                regime = "fear"
            elif latest_value < complacency_threshold:
                regime = "complacency"
            else:
                regime = "normal"

            vix_data = {
                "value": round(latest_value, 2),
                "regime": regime,
                "fear_threshold": fear_threshold,
                "complacency_threshold": complacency_threshold,
            }

            # Calculate change over five trading sessions
            if len(vix) >= 6:
                previous_value = float(vix.iloc[-6])

                vix_data["change_over_5_sessions"] = round(
                    latest_value - previous_value, 2
                )

            # Preserve the actual observation date
            observation_date = vix.index[-1]

            if hasattr(observation_date, "strftime"):
                vix_data["observation_date"] = (
                    observation_date.strftime("%Y-%m-%d")
                )
            else:
                vix_data["observation_date"] = str(
                    observation_date
                )[:10]

            result["india_vix"] = vix_data
            result["available_indicators"].append("india_vix")

    except Exception as error:
        logger.warning(
            "Could not collect India VIX data: %s", error
        )

    # Return no mood block if no indicators were successfully collected.
    if not result["available_indicators"]:
        return {}

    return result

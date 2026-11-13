"""Command line entry point.

    python main.py analyze RELIANCE                 # decision for today (NSE: RELIANCE.NS)
    python main.py analyze TCS --date 2025-06-30    # decision as of a past date
    python main.py backtest INFY                    # walk-forward test (dates from config.yaml)
    python main.py serve                            # web UI at http://127.0.0.1:8000
"""
import argparse
import logging
from datetime import date

from config.loader import load_config
from data.market_data import MarketDataService
from evaluation.backtest import Backtester
from workflow.trading_graph import TradingWorkflow


def print_analysis(result: dict) -> None:
    print("\n=== ANALYST REPORTS ===")
    for report in result["analyst_reports"]:
        print(f"\n[{report.analyst}] {report.signal.upper()} (confidence {report.confidence:.2f})")
        print(f"  {report.summary}")
        for point in report.key_points:
            print(f"   - {point}")

    proposal = result["trade_proposal"]
    print("\n=== TRADER PROPOSAL ===")
    print(f"{proposal.action} at {proposal.position_size_pct:.0%} of portfolio\n  {proposal.rationale}")

    risk = result["risk_assessment"]
    print("\n=== RISK REVIEW ===")
    print(f"{risk.decision} | risk level: {risk.risk_level}\n  {risk.comments}")

    decision = result["final_decision"]
    print("\n=== FINAL DECISION ===")
    print(f"{decision.ticker} on {decision.as_of_date}: {decision.action} {decision.position_size_pct:.0%}")


def print_backtest(result) -> None:
    print(f"\n=== BACKTEST: {result.ticker} ===")
    print(result.decisions.to_string(index=False))
    print(f"\n{'metric':<24}{'RIO strategy':>14}{'buy & hold':>14}")
    for metric, strategy_value in result.strategy_metrics.items():
        buy_and_hold_value = result.buy_and_hold_metrics[metric]
        print(f"{metric:<24}{strategy_value:>14.3f}{buy_and_hold_value:>14.3f}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RIO multi-agent trading framework")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="get a decision for one stock")
    analyze.add_argument("ticker")
    analyze.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")

    backtest = commands.add_parser("backtest", help="evaluate the strategy on historical data")
    backtest.add_argument("ticker")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--rebalance-days", type=int)

    serve = commands.add_parser("serve", help="start the web UI (chat + live agent dashboard)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    # Hide the per-request lines of the HTTP / LLM client libraries; keep their warnings and errors.
    for noisy_logger in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "anthropic"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    args = build_argument_parser().parse_args()
    config = load_config()

    if args.command == "serve":
        import uvicorn

        from api.server import create_app

        print(f"RIO web UI: http://{args.host}:{args.port}")
        uvicorn.run(create_app(config), host=args.host, port=args.port)
        return

    if args.command == "analyze":
        workflow = TradingWorkflow(config, MarketDataService(config.market, point_in_time=False))
        ticker = workflow.market_data.normalize_ticker(args.ticker)
        print_analysis(workflow.run_with_details(ticker, args.date))
        return

    # Backtests must not see today's fundamentals or news, hence point_in_time=True.
    market_data = MarketDataService(config.market, point_in_time=True)
    workflow = TradingWorkflow(config, market_data)
    defaults = config.backtest
    backtester = Backtester(workflow, market_data, config.market["risk_free_rate"])
    result = backtester.run(
        market_data.normalize_ticker(args.ticker),
        args.start or defaults["start_date"],
        args.end or defaults["end_date"],
        args.rebalance_days or defaults["rebalance_days"],
    )
    print_backtest(result)


if __name__ == "__main__":
    main()

"""Train (or re-train) the technical direction model and print its evaluation.

    python -m ml.train                      # cutoff = today
    python -m ml.train --cutoff 2025-01-01  # model that only knows data up to 2025-01-01

Also compares LightGBM with the class-prior, logistic-regression and random-forest
baselines on the same walk-forward folds, so the table can go straight into the
project report. The model and a JSON "model card" are saved under ml/artifacts/models/.
"""
import argparse
import logging
from datetime import date

from config.loader import load_config
from data.market_data import MarketDataService
from ml.registry import ModelRegistry
from ml.settings import TechnicalModelSettings


def print_report(card: dict) -> None:
    validation = card["validation"]
    print(f"\n=== TECHNICAL DIRECTION MODEL (trained until {card['trained_until']}) ===")
    print(f"Training rows: {card['training_rows']:,} from {card['training_stocks']} stocks, "
          f"{card['training_start']} .. {card['last_labelled_date']}")
    print(f"Target: down / flat / up over {card['horizon_days']} trading days "
          f"(flat = within {card['flat_threshold_vol_multiple']} x normal move)")
    print(f"Validation: {validation['method']}, test {validation['test_period']} ({validation['test_rows']:,} rows)")
    print(f"Test class mix: {validation['test_class_share']}")

    header = f"\n{'model':<22}{'bal.acc':>9}{'acc':>8}{'macroF1':>9}{'logloss':>9}{'dir.acc':>9}  folds (bal.acc)"
    print(header)
    print("-" * len(header))
    for name, scores in validation["models"].items():
        folds = " ".join(f"{s:.3f}" for s in scores["fold_balanced_accuracy"])
        print(f"{name:<22}{scores['balanced_accuracy']:>9.3f}{scores['accuracy']:>8.3f}{scores['macro_f1']:>9.3f}"
              f"{scores['log_loss']:>9.3f}{scores['directional_accuracy']:>9.3f}  {folds}")

    print(f"\nEdge over chance (balanced accuracy - 0.333): {card['edge_over_chance']:+.3f}"
          f"  ->  reliability: {card['reliability'].upper()}")
    print(f"Calibration temperature: {card['temperature']}")
    print("\nTop 10 features (split importance):")
    for name, importance in list(card["feature_importance"].items())[:10]:
        print(f"  {name:<26}{importance}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the technical analyst's LightGBM direction model")
    parser.add_argument("--cutoff", default=date.today().isoformat(), help="last date the model may see (YYYY-MM-DD)")
    parser.add_argument("--no-random-forest", action="store_true", help="skip the (slower) random-forest baseline")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    config = load_config()
    settings = TechnicalModelSettings.from_config_file()
    registry = ModelRegistry(MarketDataService(config.market, point_in_time=True), settings)
    model = registry.train(args.cutoff, include_random_forest=not args.no_random_forest)
    print_report(model.card)


if __name__ == "__main__":
    main()

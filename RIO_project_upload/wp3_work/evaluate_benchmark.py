import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from pathlib import Path
from collections import Counter

import pandas as pd

from data.sentiment_models import FinancialSentimentModel

BASE = Path(__file__).resolve().parent
INPUT = BASE / "benchmark_50_final.csv"
OUTPUT = BASE / "benchmark_predictions.csv"
METRICS = BASE / "benchmark_metrics.json"

LABELS = ["positive", "negative", "neutral"]

# Load and validate the benchmark
df = pd.read_csv(INPUT)
df["actual_label"] = df["actual_label"].astype(str).str.strip().str.lower()
df["title"] = df["title"].fillna("").astype(str).str.strip()

if len(df) != 50:
    raise ValueError(f"Expected 50 headlines, found {len(df)}")

if df["actual_label"].isna().any() or not set(df["actual_label"]).issubset(LABELS):
    raise ValueError("Invalid or missing actual labels.")

if (df["title"] == "").any():
    raise ValueError("Some headlines have empty titles.")

print("Loading FinBERT...")
finbert = FinancialSentimentModel(prefer_finbert=True)

print("Loading VADER...")
vader = FinancialSentimentModel(prefer_finbert=False)

print("\nStarting evaluation on", len(df), "headlines")

finbert_predictions = []
vader_predictions = []
finbert_times = []
vader_times = []
finbert_backends = []

for i, row in df.iterrows():
    headline = row["title"]

    # FinBERT
    result_f = finbert.score(headline)
    finbert_predictions.append(result_f["label"])
    finbert_times.append(result_f["inference_ms"])
    finbert_backends.append(result_f["backend"])

    # VADER
    result_v = vader.score(headline)
    vader_predictions.append(result_v["label"])
    vader_times.append(result_v["inference_ms"])

    print(f"{i + 1:02d}/50 | FinBERT: {result_f['label']:8s} "
          f"| VADER: {result_v['label']}")

df["finbert_prediction"] = finbert_predictions
df["vader_prediction"] = vader_predictions
df["finbert_inference_ms"] = finbert_times
df["vader_inference_ms"] = vader_times

# Metric calculation without requiring scikit-learn
def calculate_metrics(actual, predicted):
    matrix = {
        actual_label: {
            predicted_label: 0 for predicted_label in LABELS
        }
        for actual_label in LABELS
    }

    for a, p in zip(actual, predicted):
        matrix[a][p] += 1

    accuracy = sum(
        matrix[label][label] for label in LABELS
    ) / len(actual)

    report = {}
    f1_values = []
    weighted_f1 = 0

    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in LABELS if a != label)
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        support = sum(matrix[label].values())

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

        f1_values.append(f1)
        weighted_f1 += f1 * support

    report["accuracy"] = round(accuracy, 4)
    report["macro_f1"] = round(sum(f1_values) / len(LABELS), 4)
    report["weighted_f1"] = round(weighted_f1 / len(actual), 4)
    report["confusion_matrix"] = matrix

    return report

actual = df["actual_label"].tolist()

finbert_metrics = calculate_metrics(actual, finbert_predictions)
vader_metrics = calculate_metrics(actual, vader_predictions)

agreement = sum(
    f == v for f, v in zip(finbert_predictions, vader_predictions)
) / len(df)

results = {
    "dataset_size": len(df),
    "actual_label_distribution": dict(Counter(actual)),
    "finbert": finbert_metrics,
    "vader": vader_metrics,
    "model_agreement": round(agreement, 4),
    "finbert_backend_usage": dict(Counter(finbert_backends)),
    "average_finbert_inference_ms": round(
        sum(finbert_times) / len(finbert_times), 2
    ),
    "average_vader_inference_ms": round(
        sum(vader_times) / len(vader_times), 2
    )
}

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

with open(METRICS, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\n========== FINAL RESULTS ==========")
print("Dataset size:", len(df))
print("FinBERT backend usage:", results["finbert_backend_usage"])

for name, metrics in [
    ("FINBERT", finbert_metrics),
    ("VADER", vader_metrics)
]:
    print(f"\n{name}")
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Weighted F1:", metrics["weighted_f1"])

    for label in LABELS:
        print(label, metrics[label])

    print("Confusion matrix:", metrics["confusion_matrix"])

print("\nFinBERT/VADER agreement:",
      f"{agreement * 100:.2f}%")

print("Average FinBERT time:",
      results["average_finbert_inference_ms"], "ms")

print("Average VADER time:",
      results["average_vader_inference_ms"], "ms")

print("\nSaved predictions:", OUTPUT)
print("Saved metrics:", METRICS)

if set(finbert_backends) != {"finbert"}:
    print("\nWARNING: Some FinBERT predictions used a fallback backend.")
    print("Check backend usage before interpreting the comparison.")

print("\nEvaluation complete.")


from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parent

files = [
    ("indian_financial_headlines_reviewed.csv", "original_reviewed"),
    ("screened_headlines.csv", "ICICI_SBI"),
    ("next_screened_headlines.csv", "INFOSYS"),
    ("final_screened_headlines.csv", "WIPRO"),
    ("last_screened_headlines.csv", "TATA_MOTORS"),
    ("last_4_screened.csv", "RELIANCE"),
    ("final_3_screened.csv", "LT_ADANI"),
]

frames = []

for filename, batch in files:
    path = BASE / filename

    if not path.exists():
        print("MISSING:", filename)
        continue

    df = pd.read_csv(path)
    df["batch"] = batch

    for col in ["ticker", "title", "summary", "source", "date",
                "url", "actual_label", "review_status"]:
        if col not in df.columns:
            df[col] = ""

    frames.append(df)

if not frames:
    raise RuntimeError("No input files found.")

combined = pd.concat(frames, ignore_index=True)

combined["title_key"] = combined["title"].fillna("").apply(
    lambda x: re.sub(r"\W+", "", str(x).lower())
)

before = len(combined)
combined = combined.drop_duplicates(
    subset=["title_key"], keep="first"
).copy()
duplicates = before - len(combined)

combined.drop(columns=["title_key"], inplace=True)

columns = [
    "ticker", "title", "summary", "source", "date", "url",
    "actual_label", "review_status", "batch"
]

combined = combined[columns]
output = BASE / "benchmark_50_candidates.csv"
combined.to_csv(output, index=False, encoding="utf-8-sig")

print("\n=== BENCHMARK CONSOLIDATION ===")
print("Rows before deduplication:", before)
print("Exact duplicate titles removed:", duplicates)
print("Final unique headlines:", len(combined))
print("Missing labels:", combined["actual_label"].isna().sum()
      + (combined["actual_label"].astype(str).str.strip() == "").sum())
print("\nLabel distribution:")
print(combined["actual_label"].value_counts(dropna=False))
print("\nSaved:", output)

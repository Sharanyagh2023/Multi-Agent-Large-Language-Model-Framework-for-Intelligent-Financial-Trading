import pandas as pd
import re

benchmark_path = "wp3_work/benchmark_50_candidates.csv"
review_path = "wp3_work/21_headlines_to_review.csv"

benchmark = pd.read_csv(benchmark_path)
reviewed = pd.read_csv(review_path)

def normalize(title):
    return re.sub(r"\W+", "", str(title).lower())

reviewed["key"] = reviewed["title"].apply(normalize)
benchmark["key"] = benchmark["title"].apply(normalize)

label_map = dict(zip(reviewed["key"], reviewed["actual_label"]))

matched = benchmark["key"].isin(label_map)

benchmark.loc[matched, "actual_label"] = (
    benchmark.loc[matched, "key"].map(label_map)
)

benchmark.loc[matched, "review_status"] = "human_reviewed"

benchmark = benchmark.drop(columns=["key"])

benchmark.to_csv(
    "wp3_work/benchmark_50_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Matched labels:", matched.sum())
print("Total headlines:", len(benchmark))
print("Missing labels:", benchmark["actual_label"].isna().sum())
print("\nFinal distribution:")
print(benchmark["actual_label"].value_counts(dropna=False))

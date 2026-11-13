import csv
from pathlib import Path

input_file = Path("wp3_work/indian_financial_headlines.csv")
output_file = Path("wp3_work/indian_financial_headlines_review.csv")

# Original CSV row numbers (zero-based, excluding the header)
labels = {
    0: "positive",
    1: "positive",
    2: "neutral",
    3: "neutral",
    4: "neutral",
    5: "positive",
    6: "positive",
    8: "positive",
    12: "neutral",
    13: "neutral",
    15: "neutral",
    17: "positive",
    18: "negative",
    20: "positive",
    22: "neutral",
    26: "neutral",
    28: "positive",
    29: "neutral",
    30: "positive",
    31: "positive",
    32: "positive",
    33: "positive",
    34: "positive",
    35: "positive",
    38: "positive",
    42: "neutral",
    45: "positive",
    46: "negative",
    47: "neutral",
}

with input_file.open("r", encoding="utf-8-sig", newline="") as f:
    articles = list(csv.DictReader(f))

cleaned = []

for row_number, article in enumerate(articles):
    if row_number not in labels:
        continue

    article["actual_label"] = ""
    article["proposed_label"] = labels[row_number]
    article["review_status"] = "needs_human_confirmation"
    cleaned.append(article)

fieldnames = [
    "title",
    "summary",
    "source",
    "date",
    "actual_label",
    "proposed_label",
    "review_status",
]

with output_file.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned)

print("Original articles:", len(articles))
print("Selected articles:", len(cleaned))
print("Proposed labels:", {
    label: sum(1 for row in cleaned if row["proposed_label"] == label)
    for label in ["positive", "negative", "neutral"]
})
print("Saved:", output_file)

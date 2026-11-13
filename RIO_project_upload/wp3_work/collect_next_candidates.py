import re
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.news import NewsService

BASE = Path(__file__).resolve().parent

files = [
    BASE / "indian_financial_headlines_reviewed.csv",
    BASE / "additional_headlines_candidates.csv",
]

existing = set()
for file in files:
    if file.exists():
        d = pd.read_csv(file)
        if "title" in d:
            existing.update(
                re.sub(r"\W+", "", str(t).lower())
                for t in d["title"].dropna()
            )

service = NewsService(lookback_days=14)
tickers = ["INFY", "WIPRO", "TATAMOTORS", "RELIANCE", "LT", "ADANIENT"]

results = []
seen = set()

for ticker in tickers:
    print(f"Checking {ticker}...")
    try:
        articles = service.get_articles(
            ticker=ticker, as_of="2026-11-12", limit=15
        )
    except Exception as e:
        print("Skipped:", e)
        continue

    for a in articles:
        title = str(a.get("title", "")).strip()
        key = re.sub(r"\W+", "", title.lower())

        if not title or key in existing or key in seen:
            continue

        seen.add(key)
        results.append({
            "ticker": ticker,
            "title": title,
            "summary": a.get("summary", ""),
            "source": a.get("source", ""),
            "date": a.get("date", ""),
            "url": a.get("url", ""),
            "actual_label": "",
            "review_status": "needs_human_review",
        })

        if len(results) == 13:
            break

    print("New candidates:", len(results))
    if len(results) == 13:
        break

out = pd.DataFrame(results)
out.to_csv(BASE / "next_headlines_candidates.csv",
           index=False, encoding="utf-8-sig")

print("\nCollected:", len(out))
print("Saved:", BASE / "next_headlines_candidates.csv")

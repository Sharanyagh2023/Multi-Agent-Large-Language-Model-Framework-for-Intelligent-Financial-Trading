import sys
import re
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.news import NewsService

BASE = Path(__file__).resolve().parent
REVIEWED = BASE / "indian_financial_headlines_reviewed.csv"
OUTPUT = BASE / "additional_headlines_candidates.csv"

if not REVIEWED.exists():
    raise FileNotFoundError(f"Missing reviewed file: {REVIEWED}")

reviewed = pd.read_csv(REVIEWED)
existing_titles = {
    re.sub(r"\W+", "", str(x).lower())
    for x in reviewed["title"].dropna()
}

selection = {
    "ICICIBANK": 15,
    "SBIN": 15,
    "INFY": 15,
    "WIPRO": 15,
    "TATAMOTORS": 15,
    "LT": 15,
    "ADANIENT": 15,
}

service = NewsService(lookback_days=14)
candidates = []
seen = set()

for ticker, limit in selection.items():
    print(f"\nCollecting {ticker}...")
    try:
        articles = service.get_articles(
            ticker=ticker,
            as_of="2026-11-12",
            limit=limit
        )
    except Exception as exc:
        print(f"Could not collect {ticker}: {exc}")
        continue

    for article in articles:
        title = str(article.get("title", "")).strip()
        normalized = re.sub(r"\W+", "", title.lower())

        if not title or normalized in existing_titles or normalized in seen:
            continue

        seen.add(normalized)
        candidates.append({
            "ticker": ticker,
            "title": title,
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "url": article.get("url", ""),
            "actual_label": "",
            "review_status": "needs_human_review",
        })

        if len(candidates) >= 21:
            break

    print(f"Retrieved {len(articles)} articles; unique candidates: {len(candidates)}")

    if len(candidates) >= 21:
        break

df = pd.DataFrame(candidates)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n=== COLLECTION COMPLETE ===")
print("Candidates collected:", len(df))
print("Saved to:", OUTPUT)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
from pathlib import Path
from data.news import NewsService

service = NewsService()

selection = {
    "HDFCBANK": 20,
    "RELIANCE": 17,
    "TCS": 13,
}

articles = []
seen = set()

for ticker, limit in selection.items():
    results = service.get_articles(
        ticker=ticker,
        as_of="2026-11-12",
        limit=50
    )

    count = 0
    for article in results:
        key = article.get("url") or article.get("title", "").strip().lower()

        if not key or key in seen:
            continue

        seen.add(key)
        articles.append({
            "title": article.get("title", ""),
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "actual_label": ""
        })
        count += 1

        if count >= limit:
            break

output = Path("wp3_work/indian_financial_headlines.csv")
with output.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["title", "summary", "source", "date", "actual_label"]
    )
    writer.writeheader()
    writer.writerows(articles)

print("Articles exported:", len(articles))
print("Target articles: 50")
print("Saved to:", output)

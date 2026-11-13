import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.news import NewsService

service = NewsService(lookback_days=14)

selections = [
    ("LT", "Noida Metro Aqua Line Extension: L&T bags contract"),
    ("LT", "Karnataka Minister Eshwar Khandre asks officials to blacklist"),
    ("ADANIENT", "Kutch Copper inks MoU with Chile's Codelco"),
]

rows = []

for ticker, phrase in selections:
    articles = service.get_articles(
        ticker=ticker, as_of="2026-11-12", limit=20
    )

    match = next(
        (a for a in articles if phrase.lower() in a.get("title", "").lower()),
        None
    )

    if match:
        rows.append({
            "ticker": ticker,
            "title": match.get("title", ""),
            "summary": match.get("summary", ""),
            "source": match.get("source", ""),
            "date": match.get("date", ""),
            "url": match.get("url", ""),
            "actual_label": "",
            "review_status": "needs_human_review"
        })
    else:
        print("Not found:", phrase)

df = pd.DataFrame(rows)
df.to_csv(
    "wp3_work/final_3_screened.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved headlines:", len(df))
print(df[["ticker", "title", "source"]].to_string(index=False))


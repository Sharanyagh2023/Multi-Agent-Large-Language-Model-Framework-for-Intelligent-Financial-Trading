"""Live check for the "done when" list: python -m data.news.demo RELIANCE.NS INFY.NS ...

Needs internet. Prints company articles, then the macro block, per ticker.
"""
import sys

from data.news import NewsService


def main(tickers: list[str]) -> None:
    service = NewsService()
    for ticker in tickers:
        articles = service.get_articles(ticker, limit=10)
        print(f"\n== {ticker}: {len(articles)} articles")
        for a in articles:
            print(f"  {a['date']}  [{a['event_type']:<10}] {a['title']}  ({a['source']})")
    print("\n== Macro")
    for a in service.get_macro(limit=5):
        print(f"  {a['date']}  {a['title']}  ({a['source']})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])

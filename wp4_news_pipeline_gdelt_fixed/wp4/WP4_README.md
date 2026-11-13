# WP4 – News analyst and news pipeline

## Files (all inside WP4's ownership)
- `data/news/` – NewsService, sources, cleaning, classifier, SQLite archive, aliases, macro queries
- `agents/news_analyst.py` – NewsAnalyst (collect_data / build_prompt)
- `tests/test_news.py` – offline tests (pytest style, fake sources, no network)
- `SOURCES.md` – source evaluation and limits

No new dependencies (standard library only).

## Interface for WP3 (agreed contract)
`NewsService.get_articles(ticker, as_of, limit)` -> list of dicts:
`date` (YYYY-MM-DD), `title`, `summary`, `source`, `url`, plus extra `event_type`, `relevance`.
`NewsService.get_macro(as_of, limit)` returns the same shape for India-wide news.
Nothing published after `as_of` is returned.

## Check it live (needs internet)
    python -m data.news.demo RELIANCE.NS TCS.NS HDFCBANK.NS
    python -m pytest tests/test_news.py

## Config section to give WP5 (config/config.yaml)
    analysts:
      news:
        db_path: data/news/archive.sqlite
        lookback_days: 14
        refresh_hours: 6
        company_limit: 10
        macro_limit: 5

## Things to adjust when wiring into the base code
1. `agents/news_analyst.py`: confirm the `AnalystAgent` import path and the exact
   `collect_data` / `build_prompt` signatures (a stand-in class is used if the import fails).
2. `data/news/aliases.py`: edit the table to match the final test universe.
3. Add `NewsAnalyst.sources(data)` output to `AnalystReport.sources` once WP5 adds the field.

## Known limits
- Google News RSS has no deep history; the archive builds history from the first run.
- Historical `as_of` dates only work for dates already archived (or inside GDELT's window).
- Rule-based classifier is keyword-based: expect some "other" and some mislabels.
- Near-duplicate threshold (0.9) can merge two headlines that differ only by a number.

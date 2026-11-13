"""SQLite archive of every article we have seen.

Why: free sources have little history. Keeping each article with its
publication date means the archive grows into a point-in-time dataset, so
backtests over later months can ask "what was known on date X" honestly.
"""
import sqlite3
from datetime import date, datetime, timedelta

from data.news.models import Article

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    ticker TEXT NOT NULL,            -- 'MACRO' for market-wide items
    url TEXT NOT NULL,
    kind TEXT NOT NULL,
    published TEXT NOT NULL,         -- ISO date the article appeared
    fetched_at TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    relevance REAL NOT NULL,
    PRIMARY KEY (ticker, url)
);
CREATE INDEX IF NOT EXISTS idx_articles_pit ON articles (ticker, published);
CREATE TABLE IF NOT EXISTS fetch_log (key TEXT PRIMARY KEY, fetched_at TEXT NOT NULL);
"""


class ArticleStore:
    def __init__(self, path: str = "data/news/archive.sqlite"):
        self.path = path
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, articles: list[Article]) -> int:
        """Insert new articles; ones already stored are left untouched."""
        now = datetime.utcnow().isoformat(timespec="seconds")
        added = 0
        with self._conn() as conn:
            for a in articles:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (a.ticker or "MACRO", a.url, a.kind, a.published.isoformat(), now,
                     a.title, a.summary, a.source, a.event_type, a.relevance))
                added += cur.rowcount
        return added

    def query(self, ticker: str, as_of: date, lookback_days: int, limit: int) -> list[Article]:
        """Articles published in (as_of - lookback, as_of]. Never newer than as_of."""
        since = (as_of - timedelta(days=lookback_days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT published,title,summary,source,url,kind,ticker,event_type,relevance "
                "FROM articles WHERE ticker=? AND published<=? AND published>? "
                "ORDER BY published DESC, relevance DESC LIMIT ?",
                (ticker, as_of.isoformat(), since, limit)).fetchall()
        return [Article(date.fromisoformat(r[0]), *r[1:]) for r in rows]

    def last_fetch(self, key: str) -> datetime | None:
        with self._conn() as conn:
            row = conn.execute("SELECT fetched_at FROM fetch_log WHERE key=?", (key,)).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def mark_fetched(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?)",
                         (key, datetime.utcnow().isoformat(timespec="seconds")))

"""NewsService: the one object WP3 and the news analyst talk to.

Contract (agreed with WP3/WP5):
    get_articles(ticker, as_of, limit) -> [{date, title, summary, source, url, ...}]

No look-ahead: nothing published after `as_of` is ever returned.
"""
import logging
from datetime import date, datetime, timedelta

from data.news.aliases import names_for, normalise_ticker
from data.news.classifier import classify_all
from data.news.cleaning import deduplicate, filter_relevant
from data.news.macro import MACRO_QUERIES
from data.news.models import Article
from data.news.sources import GdeltDoc, GoogleNewsRSS
from data.news.sources.base import NewsSource
from data.news.store import ArticleStore

log = logging.getLogger(__name__)


def to_date(value) -> date:
    """Accept date, datetime or 'YYYY-MM-DD'; default to today."""
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


class NewsService:
    def __init__(self, db_path: str = "data/news/archive.sqlite",
                 sources: list[NewsSource] | None = None,
                 lookback_days: int = 14, refresh_hours: float = 6,
                 today: date | None = None):
        self.store = ArticleStore(db_path)
        self.sources = sources if sources is not None else [GoogleNewsRSS(), GdeltDoc()]
        self.lookback_days = lookback_days
        self.refresh_hours = refresh_hours
        self._today = today          # lets tests fix "today"

    def today(self) -> date:
        return self._today or date.today()

    # ---- public API -------------------------------------------------
    def get_articles(self, ticker: str, as_of=None, limit: int = 10) -> list[dict]:
        key, as_of = normalise_ticker(ticker), to_date(as_of)
        self._ensure_company(ticker, key, as_of)
        found = self.store.query(key, as_of, self.lookback_days, limit)
        return [a.to_dict() for a in found]

    def get_macro(self, as_of=None, limit: int = 5) -> list[dict]:
        as_of = to_date(as_of)
        self._ensure_macro(as_of)
        return [a.to_dict() for a in self.store.query("MACRO", as_of, self.lookback_days, limit)]

    # ---- fetching ---------------------------------------------------
    def _is_live(self, as_of: date) -> bool:
        return as_of >= self.today() - timedelta(days=1)

    def _stale(self, key: str) -> bool:
        last = self.store.last_fetch(key)
        return last is None or datetime.utcnow() - last > timedelta(hours=self.refresh_hours)

    def _sources_for(self, as_of: date) -> list[NewsSource]:
        """Live dates use every source; past dates only sources with history."""
        return self.sources if self._is_live(as_of) else [s for s in self.sources if s.has_history]

    def _ensure_company(self, ticker: str, key: str, as_of: date) -> None:
        if self._is_live(as_of):
            if not self._stale(key):
                return
        elif self.store.query(key, as_of, self.lookback_days, 1):
            return                   # past date already covered by the archive
        raw, ok = self._fetch(self._sources_for(as_of), names_for(ticker)[:3], as_of)
        articles = classify_all(deduplicate(filter_relevant(raw, ticker)))
        for a in articles:
            a.ticker, a.kind = key, "company"
        self.store.save(articles)
        if ok and self._is_live(as_of):
            self.store.mark_fetched(key)

    def _ensure_macro(self, as_of: date) -> None:
        if not self._is_live(as_of) or not self._stale("MACRO"):
            return
        collected, any_ok = [], False
        for query in MACRO_QUERIES:
            raw, ok = self._fetch(self.sources, [query], as_of)
            collected += raw
            any_ok = any_ok or ok
        for a in collected:
            a.ticker, a.kind, a.event_type = "MACRO", "macro", "macro"
            a.relevance = 1.0
        self.store.save(deduplicate(collected))
        if any_ok:
            self.store.mark_fetched("MACRO")

    def _fetch(self, sources: list[NewsSource], terms: list[str],
               as_of: date) -> tuple[list[Article], bool]:
        """Ask each source; one failing source must not break the rest."""
        start, raw, any_ok = as_of - timedelta(days=self.lookback_days), [], False
        for source in sources:
            try:
                raw += source.fetch(terms, start, as_of)
                any_ok = True
            except Exception as exc:                        # network, parse, rate limit
                log.warning("news source %s failed: %s", source.name, exc)
        return [a for a in raw if a.published <= as_of], any_ok

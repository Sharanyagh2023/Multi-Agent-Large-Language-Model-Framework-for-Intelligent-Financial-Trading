"""GDELT DOC 2.0 API source with conservative rate-limit handling.

GDELT can return HTTP 429, transient TLS/network errors, or an HTML/error
response instead of JSON. This source treats those conditions as a temporary
source failure so the NewsService can continue with Google News RSS.
"""
import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from urllib.parse import quote_plus

from data.news.models import Article
from data.news.sources.base import NewsSource, USER_AGENT


class GdeltDoc(NewsSource):
    name = "gdelt"
    has_history = True

    BASE = (
        "https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist"
        "&format=json&maxrecords={n}&sort=datedesc"
        "&startdatetime={s}&enddatetime={e}"
    )
    # GDELT is deliberately kept slow: its public endpoint rate-limits fairly
    # aggressively. A process-wide timestamp also covers the three macro queries.
    MIN_INTERVAL = 5.0
    RETRIES = 2
    TIMEOUT = 20
    _last_request = 0.0

    @classmethod
    def _wait_for_slot(cls) -> None:
        elapsed = time.monotonic() - cls._last_request
        if elapsed < cls.MIN_INTERVAL:
            time.sleep(cls.MIN_INTERVAL - elapsed)

    @classmethod
    def _get(cls, url: str) -> bytes:
        last_exc = None
        for attempt in range(cls.RETRIES + 1):
            cls._wait_for_slot()
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            })
            cls._last_request = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=cls.TIMEOUT) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code != 429 or attempt >= cls.RETRIES:
                    raise
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else cls.MIN_INTERVAL * (attempt + 1)
                except (TypeError, ValueError):
                    delay = cls.MIN_INTERVAL * (attempt + 1)
                time.sleep(min(max(delay, cls.MIN_INTERVAL), 30.0))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt >= cls.RETRIES:
                    raise
                time.sleep(cls.MIN_INTERVAL * (attempt + 1))
        raise last_exc or RuntimeError("GDELT request failed")

    def fetch(self, terms, start, end, limit=50):
        query = " OR ".join(f'"{t}"' for t in terms)
        q = f'({query}) sourcecountry:IN sourcelang:english'
        url = self.BASE.format(
            q=quote_plus(q), n=min(limit, 250),
            s=start.strftime("%Y%m%d000000"),
            e=end.strftime("%Y%m%d235959"),
        )
        return self.parse(self._get(url))

    @staticmethod
    def parse(raw: bytes) -> list[Article]:
        """Parse JSON safely; non-JSON GDELT error pages become a clear failure."""
        if not raw or not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        articles = []
        for row in data.get("articles", []):
            try:
                published = datetime.strptime(row["seendate"][:8], "%Y%m%d").date()
            except (KeyError, TypeError, ValueError):
                continue
            title = (row.get("title") or "").strip()
            if title and row.get("url"):
                articles.append(Article(
                    published, title, "", row.get("domain", "gdelt"), row["url"]
                ))
        return articles

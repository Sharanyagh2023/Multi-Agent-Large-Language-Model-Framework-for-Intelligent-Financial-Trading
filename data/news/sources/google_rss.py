"""Google News RSS: live, free, no key, but no deep history.

Limits to verify against Google's terms before heavy use: the feed is meant
for personal readers, returns about 100 items per query, and only reaches back
a short time. It is the right *live* source; the SQLite archive turns it into
history over time.
"""
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import quote_plus

from data.news.models import Article
from data.news.sources.base import NewsSource

_TAG = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", text or ""))).strip()


class GoogleNewsRSS(NewsSource):
    name = "google_news_rss"
    has_history = False

    BASE = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

    def fetch(self, terms, start, end, limit=50):
        # after:/before: operators narrow the window; Google treats them as hints.
        query = " OR ".join(f'"{t}"' for t in terms)
        q = f'{query} after:{start.isoformat()} before:{end.isoformat()}'
        xml_bytes = self.http_get(self.BASE.format(q=quote_plus(q)))
        return self.parse(xml_bytes)[:limit]

    @staticmethod
    def parse(xml_bytes: bytes) -> list[Article]:
        """Parse an RSS document. Separate from fetch() so it can be tested offline."""
        articles = []
        root = ET.fromstring(xml_bytes)
        for item in root.iter("item"):
            title = _clean(item.findtext("title"))
            link = (item.findtext("link") or "").strip()
            pub = item.findtext("pubDate")
            if not (title and link and pub):
                continue
            try:
                published = parsedate_to_datetime(pub).date()
            except (TypeError, ValueError):
                continue
            source_el = item.find("source")
            source = _clean(source_el.text) if source_el is not None else "Google News"
            # Google titles end with " - Publisher"; strip it, the publisher is in `source`.
            if source and title.endswith(" - " + source):
                title = title[: -len(source) - 3]
            articles.append(Article(published, title, _clean(item.findtext("description")),
                                    source, link))
        return articles

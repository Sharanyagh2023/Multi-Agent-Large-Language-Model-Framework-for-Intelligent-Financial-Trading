"""Common interface for news sources."""
import urllib.request
from abc import ABC, abstractmethod
from datetime import date

from data.news.models import Article

USER_AGENT = "Mozilla/5.0 (compatible; RIO-news-research/0.1)"


class NewsSource(ABC):
    """One place articles come from.

    Why an interface: the service can try several sources and keep going if
    one is down or blocks us, which the spec asks for ("always keep a fallback").
    """
    name = "base"
    has_history = False   # True only if the source can answer for past dates

    @abstractmethod
    def fetch(self, terms: list[str], start: date, end: date, limit: int = 50) -> list[Article]:
        """Return articles matching ANY of `terms` published in [start, end]."""

    @staticmethod
    def http_get(url: str, timeout: int = 15) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

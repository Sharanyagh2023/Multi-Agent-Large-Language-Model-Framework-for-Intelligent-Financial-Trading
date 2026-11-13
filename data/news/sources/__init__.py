"""News sources. Each source implements NewsSource.fetch()."""
from data.news.sources.base import NewsSource
from data.news.sources.google_rss import GoogleNewsRSS
from data.news.sources.gdelt import GdeltDoc

__all__ = ["NewsSource", "GoogleNewsRSS", "GdeltDoc"]

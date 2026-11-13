"""Plain data container shared by every part of the news pipeline."""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Article:
    """One news item.

    Why a dataclass: sources, cleaner, classifier and store all pass the same
    shape around, so a typo in a field name fails loudly instead of silently.
    """
    published: date
    title: str
    summary: str
    source: str
    url: str
    kind: str = "company"          # "company" or "macro"
    ticker: str = ""
    event_type: str = "other"
    relevance: float = 0.0

    def to_dict(self) -> dict:
        """The WP4 -> WP3 contract: date, title, summary, source, url.

        event_type and relevance are extra keys; consumers may ignore them.
        """
        return {
            "date": self.published.isoformat(),
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "event_type": self.event_type,
            "relevance": round(self.relevance, 3),
        }

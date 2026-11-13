"""Clean the feed: drop irrelevant matches, remove duplicates and near-duplicates."""
import re
from difflib import SequenceMatcher

from data.news.aliases import AMBIGUOUS, excluded_phrases, names_for
from data.news.models import Article

# Words that show an ambiguous alias really means the company.
_CONTEXT = re.compile(
    r"\b(shares?|stock|stocks|q[1-4]|fy\d{2}|results?|profit|revenue|earnings|dividend|"
    r"ltd|limited|nse|bse|sensex|nifty|brokerage|target price|board|ceo|order|deal)\b", re.I)


def _find(alias: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", text,
                     re.I) is not None


def score_relevance(article: Article, ticker: str) -> float:
    """0..1. Title mention is worth most; ambiguous aliases need context."""
    title, body = article.title, article.summary
    text = title + " " + body
    primary = names_for(ticker)[0]
    if not _find(primary, text) and any(_find(p, text) for p in excluded_phrases(ticker)):
        return 0.0        # e.g. Reliance Power is not Reliance Industries
    score = 0.0
    for alias in names_for(ticker):
        in_title, in_body = _find(alias, title), _find(alias, body)
        if not (in_title or in_body):
            continue
        if alias.upper() in AMBIGUOUS and not _CONTEXT.search(title + " " + body):
            continue
        score = max(score, 1.0 if in_title else 0.5)
    return score


def filter_relevant(articles: list[Article], ticker: str, min_score: float = 0.5) -> list[Article]:
    kept = []
    for a in articles:
        a.relevance = score_relevance(a, ticker)
        if a.relevance >= min_score:
            kept.append(a)
    return kept


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def deduplicate(articles: list[Article], threshold: float = 0.9) -> list[Article]:
    """Remove same-URL and near-identical headlines (syndicated copies).

    Keeps the earliest item in each cluster. O(n^2) is fine: a ticker has
    tens of articles, not thousands.
    """
    kept: list[Article] = []
    for a in sorted(articles, key=lambda x: (x.published, -x.relevance)):
        n = _norm(a.title)
        duplicate = any(
            a.url == k.url or SequenceMatcher(None, n, _norm(k.title)).ratio() >= threshold
            for k in kept)
        if not duplicate:
            kept.append(a)
    return kept

"""Rule-based event classifier. First version; an LLM classifier can replace it.

Order matters: the first matching rule wins, so put the most specific first.
"""
import re

from data.news.models import Article

RULES = [
    ("dividend", r"\b(dividend|bonus issue|stock split|record date|buyback|buy-back)\b"),
    ("earnings", r"\b(q[1-4]\b|quarterly|results?|net profit|net loss|revenue|ebitda|"
                 r"earnings|guidance|margins?)\b"),
    ("m&a", r"\b(acquir\w+|acquisition|merger|merge[sd]?|demerger|takeover|stake|"
            r"buys?|sells?|divest\w*|joint venture|open offer|ipo|qip)\b"),
    ("regulatory", r"\b(sebi|rbi|cci|nclt|tribunal|penalty|fine[sd]?|probe|"
                   r"investigation|notice|ban|licen[cs]e|court|tax demand|gst)\b"),
    ("management", r"\b(ceo|cfo|md\b|chairman|appoints?|appointed|resigns?|resignation|"
                   r"steps down|succession|promoter)\b"),
    ("macro", r"\b(rbi policy|repo rate|inflation|gdp|budget|fii|fpi|rupee|"
              r"crude|monsoon|us fed|tariff)\b"),
]
_COMPILED = [(name, re.compile(rx, re.I)) for name, rx in RULES]


def classify(article: Article) -> str:
    text = f"{article.title} {article.summary}"
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return "other"


def classify_all(articles: list[Article]) -> list[Article]:
    for a in articles:
        a.event_type = classify(a)
    return articles

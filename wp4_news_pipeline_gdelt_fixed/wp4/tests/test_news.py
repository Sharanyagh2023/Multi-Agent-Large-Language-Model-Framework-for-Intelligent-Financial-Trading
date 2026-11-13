"""Offline tests: no network. Fake sources stand in for Google/GDELT."""
import json
from datetime import date, timedelta

from agents.news_analyst import NewsAnalyst
from data.news.aliases import names_for, normalise_ticker
from data.news.classifier import classify
from data.news.cleaning import deduplicate, filter_relevant, score_relevance
from data.news.models import Article
from data.news.service import NewsService
from data.news.sources import GdeltDoc, GoogleNewsRSS
from data.news.sources.base import NewsSource

TODAY = date(2026, 11, 12)

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Reliance Q2 profit rises 9% - Mint</title><link>https://x/1</link>
<pubDate>Tue, 10 Nov 2026 08:00:00 GMT</pubDate><source url="https://mint">Mint</source>
<description>&lt;a href="x"&gt;Reliance Industries&lt;/a&gt; posts results</description></item>
<item><title>Bad item without date</title><link>https://x/2</link></item>
</channel></rss>"""


def art(title, d=TODAY, url=None, summary="", source="T"):
    return Article(d, title, summary, source, url or f"https://x/{abs(hash(title))}")


class FakeSource(NewsSource):
    name, has_history = "fake", True

    def __init__(self, articles=None, fail=False):
        self.articles, self.fail, self.calls = articles or [], fail, 0

    def fetch(self, terms, start, end, limit=50):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return list(self.articles)


def service(tmp_path, sources):
    return NewsService(str(tmp_path / "a.sqlite"), sources, today=TODAY)


def test_ticker_normalisation_and_aliases():
    assert normalise_ticker("reliance.ns") == "RELIANCE"
    assert "RIL" in names_for("RELIANCE.NS")
    assert names_for("UNKNOWN.NS") == ["UNKNOWN"]


def test_rss_parse_strips_publisher_and_html_and_skips_bad_items():
    items = GoogleNewsRSS.parse(RSS)
    assert len(items) == 1
    assert items[0].title == "Reliance Q2 profit rises 9%"
    assert items[0].source == "Mint" and items[0].published == date(2026, 11, 10)
    assert "<a" not in items[0].summary


def test_gdelt_parse():
    raw = json.dumps({"articles": [{"title": "Infosys wins deal", "url": "https://g/1",
                                    "seendate": "20261110T080000Z", "domain": "ex.com"}]}).encode()
    assert GdeltDoc.parse(raw)[0].published == date(2026, 11, 10)
    assert GdeltDoc.parse(b"") == []
    assert GdeltDoc.parse(b"<html>rate limited</html>") == []


def test_relevance_drops_unrelated_and_ambiguous_without_context():
    assert score_relevance(art("Reliance Industries shares jump"), "RELIANCE.NS") == 1.0
    assert score_relevance(art("Weather in Mumbai today"), "RELIANCE.NS") == 0.0
    assert score_relevance(art("Titan of industry speaks at summit"), "TITAN.NS") == 0.0
    assert score_relevance(art("Titan shares hit record high on Q2 results"), "TITAN.NS") == 1.0
    kept = filter_relevant([art("Reliance Industries news"), art("Cricket scores")], "RELIANCE")
    assert len(kept) == 1
    assert score_relevance(art("Reliance Power shares surge"), "RELIANCE.NS") == 0.0
    assert score_relevance(art("Reliance shares fall after Q2 results"), "RELIANCE.NS") == 1.0


def test_dedupe_same_url_and_near_duplicate_titles():
    a = art("Infosys wins $1 bn deal from Europe client", url="https://a")
    b = art("Infosys wins $1 bn deal from European client", url="https://b")
    c = art("Infosys appoints new CFO", url="https://c")
    d = art("Something else", url="https://a")
    out = deduplicate([a, b, c])
    assert len(out) == 2
    assert len(deduplicate([a, d])) == 1


def test_classifier():
    assert classify(art("Infosys Q2 net profit up")) == "earnings"
    assert classify(art("ITC declares dividend, record date set")) == "dividend"
    assert classify(art("SEBI imposes penalty on broker")) == "regulatory"
    assert classify(art("Tata Steel appoints new CEO")) == "management"
    assert classify(art("Company launches new store")) == "other"


def test_no_look_ahead(tmp_path):
    past = TODAY - timedelta(days=5)
    src = FakeSource([art("Reliance Industries Q2 results", past, "https://a"),
                      art("Reliance Industries buys stake", TODAY, "https://b")])
    svc = service(tmp_path, [src])
    svc.get_articles("RELIANCE.NS", TODAY)            # fills the archive
    got = svc.get_articles("RELIANCE.NS", past)       # point-in-time view
    assert [a["url"] for a in got] == ["https://a"]
    assert all(a["date"] <= past.isoformat() for a in got)


def test_contract_keys_and_limit(tmp_path):
    topics = ["Q2 profit", "retail expansion", "Jio tariff hike", "green energy plant",
              "new CFO named", "SEBI notice", "bond issue", "stake sale"]
    src = FakeSource([art(f"Reliance Industries {t}", TODAY, f"https://u/{i}")
                      for i, t in enumerate(topics)])
    got = service(tmp_path, [src]).get_articles("RELIANCE.NS", TODAY, limit=3)
    assert len(got) == 3
    assert {"date", "title", "summary", "source", "url"} <= set(got[0])


def test_one_source_failing_does_not_break_the_rest(tmp_path):
    ok = FakeSource([art("Infosys wins deal", TODAY, "https://ok")])
    got = service(tmp_path, [FakeSource(fail=True), ok]).get_articles("INFY.NS", TODAY)
    assert len(got) == 1


def test_all_sources_failing_returns_empty_and_retries_next_time(tmp_path):
    svc = service(tmp_path, [FakeSource(fail=True)])
    assert svc.get_articles("INFY.NS", TODAY) == []
    assert svc.store.last_fetch("INFY") is None       # failure must not block a retry


def test_refresh_is_throttled(tmp_path):
    src = FakeSource([art("Infosys wins deal", TODAY, "https://ok")])
    svc = service(tmp_path, [src])
    svc.get_articles("INFY.NS", TODAY)
    svc.get_articles("INFY.NS", TODAY)
    assert src.calls == 1


def test_macro_block_is_separate_and_flagged(tmp_path):
    src = FakeSource([art("RBI keeps repo rate unchanged", TODAY, "https://m")])
    macro = service(tmp_path, [src]).get_macro(TODAY)
    assert macro and macro[0]["event_type"] == "macro"


def test_analyst_no_company_news_returns_empty(tmp_path):
    analyst = NewsAnalyst(service(tmp_path, [FakeSource([])]))
    assert analyst.collect_data("INFY.NS", TODAY) == {}


def test_analyst_prompt_separates_blocks_and_numbers_sources(tmp_path):
    src = FakeSource([art("Infosys Q2 profit rises", TODAY, "https://c")])
    analyst = NewsAnalyst(service(tmp_path, [src]))
    data = analyst.collect_data("INFY.NS", TODAY)
    prompt = analyst.build_prompt("INFY.NS", data)
    assert "COMPANY NEWS" in prompt and "MACRO NEWS" in prompt and "[1]" in prompt
    assert "https://c" in analyst.sources(data)

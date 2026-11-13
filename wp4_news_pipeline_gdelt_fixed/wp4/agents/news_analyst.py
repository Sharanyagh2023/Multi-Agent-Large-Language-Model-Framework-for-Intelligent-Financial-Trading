
"""News analyst (WP4): turns the NewsService article list into a prompt.

Facts first, LLM second: code fetches, cleans and classifies the news; the
LLM only reads the numbered list and must cite it.
"""

from data.news import NewsService


# Import the real AnalystAgent from the integrated project.
# The fallback keeps this module importable when the base class is not
# available, such as in isolated WP4 testing.
try:
    from agents.base import AnalystAgent
except ImportError:

    class AnalystAgent:
        """Minimal fallback AnalystAgent."""

        def __init__(self, llm=None, market_data=None):
            self.llm = llm
            self.market_data = market_data

        def collect_data(self, ticker, as_of=None):
            raise NotImplementedError

        def build_prompt(self, ticker, data):
            raise NotImplementedError


class NewsAnalyst(AnalystAgent):
    """WP4 news analyst.

    Supports both:

        NewsAnalyst(service)

    for the original standalone WP4 tests, and:

        NewsAnalyst(
            llm,
            market_data,
            news_limit,
            news_service,
        )

    for the integrated project.
    """

    def __init__(
        self,
        llm=None,
        market_data=None,
        news_limit: int = 10,
        news_service: NewsService | None = None,
    ):
        # ------------------------------------------------------------
        # Backward compatibility with the original WP4 interface:
        #
        #     NewsAnalyst(service)
        #
        # In that case the first positional argument is actually the
        # NewsService instance, not the LLM.
        # ------------------------------------------------------------
        if isinstance(llm, NewsService) and news_service is None:
            news_service = llm
            llm = None

        # ------------------------------------------------------------
        # Integrated WP3/WP5 interface:
        #
        #     NewsAnalyst(llm, market_data, ...)
        #
        # Only call the real base constructor when the required
        # integrated dependencies are available.
        # ------------------------------------------------------------
        if llm is not None and market_data is not None:
            super().__init__(llm, market_data)

        self.news_limit = news_limit
        self.news_service = news_service or NewsService()

    def collect_data(self, ticker: str, as_of=None) -> dict:
        """Collect company and macro news.

        Company news is required for a stock-specific view.

        If there is no company news, return an empty dictionary so the
        analyst can report no view instead of generating a view from
        macro news alone.
        """

        company = self.news_service.get_articles(
            ticker,
            as_of,
            self.news_limit,
        )

        if not company:
            return {}

        macro = self.news_service.get_macro(
            as_of,
            self.news_limit,
        )

        return {
            "company_news": company,
            "macro_news": macro,
        }

    @staticmethod
    def sources(data: dict) -> list[str]:
        """Return URLs behind the claims.

        Used by AnalystReport.sources in the downstream WP5 workflow.
        """

        company_news = data.get("company_news", [])
        macro_news = data.get("macro_news", [])

        return [
            article["url"]
            for article in company_news + macro_news
            if article.get("url")
        ]

    def build_prompt(self, ticker: str, data: dict) -> str:
        """Build the numbered, citation-focused LLM prompt."""

        article_number = 0

        def block(items):
            nonlocal article_number

            lines = []

            for article in items:
                article_number += 1

                date = article.get("date", "")
                source = article.get("source", "")
                event_type = article.get("event_type", "")
                title = article.get("title", "")
                summary = article.get("summary", "")

                line = (
                    f"[{article_number}] "
                    f"{date} | "
                    f"{source} | "
                    f"{event_type} | "
                    f"{title}"
                )

                if summary:
                    line += f" - {summary}"

                lines.append(line)

            return "\n".join(lines) or "(none)"

        # Company and macro news are deliberately kept as separate
        # numbered blocks.
        company = block(
            data.get("company_news", [])
        )

        macro = block(
            data.get("macro_news", [])
        )

        return (
            f"You are a news analyst for the Indian stock {ticker}.\n"
            "Use ONLY the articles below. "
            "Do not use outside knowledge of prices or outcomes.\n\n"

            f"COMPANY NEWS (about {ticker}):\n"
            f"{company}\n\n"

            "MACRO NEWS (India-wide, context only, not about "
            f"{ticker} specifically):\n"
            f"{macro}\n\n"

            "Task: judge how the company news is likely to affect "
            "the stock over the next few weeks. "

            "Weigh event types (earnings, regulatory and management "
            "items usually matter more than 'other'). "

            "Use macro news only as backdrop and say so.\n"

            "Cite every claim with its number, e.g. [2]. "

            "If the evidence is thin or mixed, say that and lean neutral.\n"

            "Answer: signal (bullish/neutral/bearish), "
            "confidence 0-1, then 3-5 short bullets."
        )

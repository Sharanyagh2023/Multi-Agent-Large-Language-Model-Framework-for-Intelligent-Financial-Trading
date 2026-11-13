"""Ticker -> company name and aliases, so searches find the right articles.

Why: news says "Reliance Industries" or "RIL", never "RELIANCE.NS". Edit this
table when the index changes. Entries are (primary name, [other aliases]).
Aliases that are also common words (Titan, Trent, ITC) are marked in
AMBIGUOUS so the cleaner demands extra context before accepting a match.
"""

COMPANIES = {
    "RELIANCE": ("Reliance Industries", ["RIL", "Reliance", "Reliance Jio", "Jio Platforms"]),
    "TCS": ("Tata Consultancy Services", ["TCS"]),
    "HDFCBANK": ("HDFC Bank", ["HDFC Bank Ltd"]),
    "ICICIBANK": ("ICICI Bank", ["ICICI Bank Ltd"]),
    "INFY": ("Infosys", ["Infosys Ltd"]),
    "HINDUNILVR": ("Hindustan Unilever", ["HUL"]),
    "ITC": ("ITC Ltd", ["ITC", "ITC Limited", "ITC Hotels"]),
    "SBIN": ("State Bank of India", ["SBI"]),
    "BHARTIARTL": ("Bharti Airtel", ["Airtel"]),
    "KOTAKBANK": ("Kotak Mahindra Bank", ["Kotak Bank"]),
    "LT": ("Larsen & Toubro", ["L&T", "Larsen and Toubro"]),
    "AXISBANK": ("Axis Bank", []),
    "BAJFINANCE": ("Bajaj Finance", []),
    "ASIANPAINT": ("Asian Paints", []),
    "MARUTI": ("Maruti Suzuki", ["Maruti", "Maruti Suzuki India"]),
    "HCLTECH": ("HCL Technologies", ["HCLTech", "HCL Tech"]),
    "SUNPHARMA": ("Sun Pharmaceutical", ["Sun Pharma"]),
    "TITAN": ("Titan Company", ["Titan", "Titan Co", "Tanishq"]),
    "ULTRACEMCO": ("UltraTech Cement", ["UltraTech"]),
    "NESTLEIND": ("Nestle India", []),
    "WIPRO": ("Wipro", ["Wipro Ltd"]),
    "ONGC": ("Oil and Natural Gas Corporation", ["ONGC"]),
    "NTPC": ("NTPC", ["NTPC Ltd"]),
    "POWERGRID": ("Power Grid Corporation", ["Power Grid Corp", "PGCIL"]),
    "M&M": ("Mahindra & Mahindra", ["M&M", "Mahindra and Mahindra"]),
    "TATAMOTORS": ("Tata Motors", []),
    "TATASTEEL": ("Tata Steel", []),
    "JSWSTEEL": ("JSW Steel", []),
    "ADANIENT": ("Adani Enterprises", []),
    "ADANIPORTS": ("Adani Ports", ["Adani Ports and SEZ", "APSEZ"]),
    "COALINDIA": ("Coal India", ["CIL"]),
    "BAJAJFINSV": ("Bajaj Finserv", []),
    "TECHM": ("Tech Mahindra", []),
    "INDUSINDBK": ("IndusInd Bank", []),
    "GRASIM": ("Grasim Industries", []),
    "HINDALCO": ("Hindalco Industries", ["Hindalco"]),
    "CIPLA": ("Cipla", []),
    "DRREDDY": ("Dr Reddy's Laboratories", ["Dr Reddys", "Dr. Reddy's"]),
    "EICHERMOT": ("Eicher Motors", ["Royal Enfield"]),
    "HEROMOTOCO": ("Hero MotoCorp", []),
    "BPCL": ("Bharat Petroleum", ["BPCL"]),
    "BRITANNIA": ("Britannia Industries", ["Britannia"]),
    "APOLLOHOSP": ("Apollo Hospitals", []),
    "DIVISLAB": ("Divi's Laboratories", ["Divis Labs", "Divi's Labs"]),
    "SBILIFE": ("SBI Life Insurance", ["SBI Life"]),
    "HDFCLIFE": ("HDFC Life Insurance", ["HDFC Life"]),
    "TATACONSUM": ("Tata Consumer Products", ["Tata Consumer"]),
    "BAJAJ-AUTO": ("Bajaj Auto", []),
    "SHRIRAMFIN": ("Shriram Finance", []),
    "TRENT": ("Trent Ltd", ["Trent", "Trent Limited", "Westside", "Zudio"]),
}

# Aliases that are ordinary words or very short: require company context
# (words like "shares", "stock", "Ltd", "Q2") near the match.
AMBIGUOUS = {"ITC", "TITAN", "TRENT", "HUL", "CIL", "SBI", "TCS", "LT", "M&M", "RELIANCE", "MARUTI"}

# Different companies that share a name prefix. If one of these phrases appears and the
# primary name does not, the article is about someone else.
EXCLUDE = {
    "RELIANCE": ["Reliance Power", "Reliance Infra", "Reliance Capital", "Reliance Communications",
                 "Reliance Home Finance", "Reliance Naval", "Reliance Nippon"],
}


def normalise_ticker(ticker: str) -> str:
    """'reliance.ns' -> 'RELIANCE'. Handles .NS / .BO suffixes."""
    t = ticker.strip().upper()
    for suffix in (".NS", ".BO"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


def names_for(ticker: str) -> list[str]:
    """All names to search for. Falls back to the bare symbol if unknown."""
    key = normalise_ticker(ticker)
    if key in COMPANIES:
        primary, others = COMPANIES[key]
        return [primary] + others
    return [key]


def primary_name(ticker: str) -> str:
    return names_for(ticker)[0]


def excluded_phrases(ticker: str) -> list[str]:
    return EXCLUDE.get(normalise_ticker(ticker), [])

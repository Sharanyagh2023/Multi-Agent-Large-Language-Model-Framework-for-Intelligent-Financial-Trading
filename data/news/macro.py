"""India macro context, kept as a separate block from company news.

Headlines only. Policy dates are not hard-coded here on purpose: a wrong date
is worse than none. The RBI MPC calendar can be added to config later.
"""
MACRO_QUERIES = [
    "RBI monetary policy repo rate",
    "Union Budget India",
    "FII FPI flows Indian equities",
]

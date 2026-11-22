"""Optional real price history from Alpha Vantage (set ALPHA_VANTAGE_API_KEY)."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import List, Optional

AV_URL = "https://www.alphavantage.co/query"


def fetch_daily_closes(ticker: str, api_key: Optional[str] = None, timeout: float = 20.0) -> Optional[List[float]]:
    """Return ~100 most recent daily closes, oldest first, or None if unavailable."""
    api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key or not ticker:
        return None
    query = urllib.parse.urlencode(
        {"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "compact", "apikey": api_key}
    )
    try:
        with urllib.request.urlopen(f"{AV_URL}?{query}", timeout=timeout) as resp:
            data = json.load(resp)
    except (OSError, ValueError):
        return None
    series = data.get("Time Series (Daily)")
    if not series:
        return None  # rate limit / bad symbol: the API returns a "Note" or "Error Message" instead
    return [float(series[day]["4. close"]) for day in sorted(series)]

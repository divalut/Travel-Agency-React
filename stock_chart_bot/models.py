"""Pydantic schemas shared across the pipeline (also used as Claude structured-output formats)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    """One point read off the chart, left (oldest) to right (newest)."""

    label: str = Field(description="X-axis label or best guess at the date/time, e.g. '2026-10-03' or 'Oct 3'")
    close: float = Field(description="Approximate closing / line price at this point, read off the y-axis")


class ChartReading(BaseModel):
    """What Claude can read off a screenshot of a stock chart."""

    is_stock_chart: bool = Field(description="False if the image is not a price chart")
    ticker: Optional[str] = Field(description="Ticker symbol if visible, e.g. 'AAPL'; null if not shown")
    company_or_asset: Optional[str] = Field(description="Company / asset name if visible")
    timeframe: Optional[str] = Field(description="Visible range or interval, e.g. '6M daily', '1D 5-min'")
    chart_type: Literal["candlestick", "line", "bar", "area", "other"]
    current_price: Optional[float] = Field(description="Latest price shown on the chart")
    price_points: List[PricePoint] = Field(
        description="20-60 evenly spaced approximate closes, oldest first, covering the whole visible range"
    )
    visible_indicators: List[str] = Field(description="Indicators drawn on the chart (e.g. 'SMA 50', 'RSI 14 at ~72', 'Volume')")
    support_levels: List[float] = Field(description="Price levels acting as support")
    resistance_levels: List[float] = Field(description="Price levels acting as resistance")
    patterns: List[str] = Field(
        description="Technical patterns visible (e.g. 'ascending triangle', 'double top', 'bull flag', 'higher lows')"
    )
    volume_observation: Optional[str] = Field(description="What the volume bars show, if present")
    reading_notes: str = Field(description="Caveats about legibility / how confident the reading is")


class Signal(BaseModel):
    method: str = Field(description="Name of the method, e.g. 'RSI(14)', 'Trend: 50-day SMA', 'News sentiment'")
    reading: str = Field(description="What the method shows right now")
    direction: Literal["bullish", "bearish", "neutral"]
    weight: Literal["low", "medium", "high"] = Field(description="How much this signal should count")


class Verdict(BaseModel):
    direction: Literal["up", "down", "sideways"]
    probability_up: float = Field(description="Estimated probability (0-1) that price is higher at the end of the horizon")
    confidence: Literal["low", "medium", "high"]
    horizon: str
    summary: str = Field(description="2-4 sentence plain-English conclusion")
    signals: List[Signal]
    bull_case: str
    bear_case: str
    key_levels: str = Field(description="Levels that would confirm or invalidate the call")
    risks: List[str]

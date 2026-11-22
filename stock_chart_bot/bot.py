"""The Claude-powered pipeline: read the screenshot -> research -> verdict."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import anthropic

from .indicators import TechReport, analyze
from .market_data import fetch_daily_closes
from .models import ChartReading, Verdict

MODEL = "claude-opus-5-5"
# Server-side fallback: if a request is declined by a safety classifier, the API
# re-runs it on Anthropic's recommended fallback model inside the same call.
BETAS = ["server-side-fallback-2026-07-01"]
FALLBACKS = "default"
MAX_CONTINUATIONS = 5

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

VISION_PROMPT = """You are reading a screenshot of a stock/asset price chart.
Extract what is actually visible - do not guess at things you cannot see.
- Read the y-axis scale carefully and sample 20-60 evenly spaced approximate closing prices across the
  whole visible range, oldest (left) to newest (right). For candlesticks use the candle close.
- Record any ticker, timeframe, indicators, drawn levels, and volume you can see.
- Name technical patterns only if they are reasonably clear.
If the image is not a price chart, set is_stock_chart to false and leave lists empty."""

RESEARCH_SYSTEM = """You are a market research analyst. Use web search to gather evidence that bears on
whether this asset is likely to rise or fall over the stated horizon. Cover:
1. Recent news, earnings, guidance, analyst rating changes and upcoming catalysts (dates).
2. Broader market / sector conditions right now.
3. What established technical-analysis research says about the reliability of the specific patterns
   and indicator readings seen on this chart (e.g. historical success rates, typical failure modes).
Be concrete, cite dates and sources, and separate facts from opinion. Finish with a short bullet list of
the most decision-relevant findings."""

VERDICT_SYSTEM = """You are a disciplined market analyst producing a probabilistic call, not a promise.
Weigh every method: chart patterns, the computed indicators, real price data if present, and the research.
Rules:
- Indicators computed from prices read off a screenshot are approximate; weight them lower than real data.
- Do not be overconfident: short-horizon price moves are close to a coin flip, so probability_up should
  rarely be outside 0.30-0.70 and confidence should be 'high' only when independent methods strongly agree.
- If the ticker is unknown, rely on the chart alone and say so.
- Signals must list each method separately with its own direction and weight."""


class BotError(RuntimeError):
    pass


@dataclass
class Analysis:
    reading: ChartReading
    tech_from_chart: TechReport
    tech_from_market: Optional[TechReport]
    research: str
    sources: List[str]
    verdict: Verdict
    horizon: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reading": self.reading.model_dump(),
            "technicals_from_chart": self.tech_from_chart.to_dict(),
            "technicals_from_market_data": self.tech_from_market.to_dict() if self.tech_from_market else None,
            "research": self.research,
            "sources": self.sources,
            "verdict": self.verdict.model_dump(),
            "horizon": self.horizon,
            "warnings": self.warnings,
        }


def load_image_block(path: str | Path) -> dict:
    path = Path(path)
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise BotError(f"Unsupported image type '{path.suffix}'. Use PNG, JPEG, GIF or WebP.")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _check_stop(response) -> None:
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        category = getattr(detail, "category", None) if detail else None
        raise BotError(f"The model declined this request (category: {category}).")
    if response.stop_reason == "max_tokens":
        raise BotError("The response was cut off at max_tokens.")


class StockChartBot:
    def __init__(self, client: Optional[anthropic.Anthropic] = None, log: Callable[[str], None] = lambda _m: None):
        self.client = client or anthropic.Anthropic()
        self.log = log

    # Step 1 -----------------------------------------------------------------
    def read_chart(self, image_path: str | Path) -> ChartReading:
        response = self.client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            betas=BETAS,
            fallbacks=FALLBACKS,
            output_config={"effort": "high"},
            output_format=ChartReading,
            messages=[{"role": "user", "content": [load_image_block(image_path), {"type": "text", "text": VISION_PROMPT}]}],
        )
        _check_stop(response)
        if response.parsed_output is None:
            raise BotError("Could not parse the chart reading.")
        return response.parsed_output

    # Step 2 -----------------------------------------------------------------
    def research(self, reading: ChartReading, tech: TechReport, horizon: str) -> tuple[str, List[str]]:
        subject = reading.ticker or reading.company_or_asset or "the asset in this chart (ticker unknown)"
        brief = (
            f"Asset: {subject}\nHorizon: {horizon}\nChart timeframe: {reading.timeframe}\n"
            f"Latest price on chart: {reading.current_price}\nPatterns seen: {', '.join(reading.patterns) or 'none'}\n"
            f"Indicators on chart: {', '.join(reading.visible_indicators) or 'none'}\n"
            f"Support: {reading.support_levels} Resistance: {reading.resistance_levels}\n\n"
            f"Computed technicals:\n{tech.to_text()}"
        )
        messages: list = [{"role": "user", "content": brief}]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}]

        response = None
        for _ in range(MAX_CONTINUATIONS + 1):
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                betas=BETAS,
                fallbacks=FALLBACKS,
                output_config={"effort": "high"},
                system=RESEARCH_SYSTEM,
                tools=tools,
                messages=messages,
            )
            if response.stop_reason != "pause_turn":
                break
            # Server-side search loop paused: send the turn back unchanged and it resumes.
            self.log("  ...research still running, resuming")
            messages = messages[:1] + [{"role": "assistant", "content": response.content}]
        _check_stop(response)

        texts, sources = [], []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
                for c in getattr(block, "citations", None) or []:
                    url = getattr(c, "url", None)
                    if url and url not in sources:
                        sources.append(url)
            elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                for r in block.content:
                    url = getattr(r, "url", None)
                    if url and url not in sources:
                        sources.append(url)
        return "".join(texts).strip(), sources

    # Step 3 -----------------------------------------------------------------
    def verdict(
        self,
        reading: ChartReading,
        tech_chart: TechReport,
        tech_market: Optional[TechReport],
        research: str,
        horizon: str,
    ) -> Verdict:
        payload = {
            "horizon": horizon,
            "chart_reading": reading.model_dump(),
            "technicals_from_screenshot": tech_chart.to_dict(),
            "technicals_from_real_daily_data": tech_market.to_dict() if tech_market else None,
            "research": research or "(research skipped)",
        }
        response = self.client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            betas=BETAS,
            fallbacks=FALLBACKS,
            output_config={"effort": "high"},
            output_format=Verdict,
            system=VERDICT_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
        _check_stop(response)
        if response.parsed_output is None:
            raise BotError("Could not parse the verdict.")
        return response.parsed_output

    # Full pipeline ----------------------------------------------------------
    def analyze(self, image_path: str | Path, horizon: str = "1-4 weeks", do_research: bool = True) -> Analysis:
        warnings: List[str] = []

        self.log("1/4 Reading the chart screenshot...")
        reading = self.read_chart(image_path)
        if not reading.is_stock_chart:
            raise BotError("That image doesn't look like a price chart.")

        self.log("2/4 Running technical indicators...")
        tech_chart = analyze([p.close for p in reading.price_points], "prices read from screenshot (approximate)")
        tech_market = None
        if reading.ticker:
            closes = fetch_daily_closes(reading.ticker)
            if closes:
                tech_market = analyze(closes, f"Alpha Vantage daily closes for {reading.ticker}")
            else:
                warnings.append("No real market data (set ALPHA_VANTAGE_API_KEY to cross-check with actual prices).")
        else:
            warnings.append("No ticker visible on the chart, so research and real-data checks are limited.")

        research, sources = "", []
        if do_research:
            self.log("3/4 Researching news, catalysts and pattern reliability (web search)...")
            research, sources = self.research(reading, tech_market or tech_chart, horizon)
        else:
            self.log("3/4 Research skipped.")

        self.log("4/4 Weighing the evidence...")
        verdict = self.verdict(reading, tech_chart, tech_market, research, horizon)
        return Analysis(reading, tech_chart, tech_market, research, sources, verdict, horizon, warnings)

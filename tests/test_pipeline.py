"""End-to-end pipeline test through the real Anthropic SDK with a mocked HTTP transport."""

import base64
import json

import anthropic
import httpx2
import pytest

from stock_chart_bot import BotError, StockChartBot
from stock_chart_bot.cli import render

# 1x1 transparent PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

READING = {
    "is_stock_chart": True,
    "ticker": "ACME",
    "company_or_asset": "Acme Corp",
    "timeframe": "6M daily",
    "chart_type": "candlestick",
    "current_price": 139.5,
    "price_points": [{"label": f"d{i}", "close": 100 + i * 0.7 + (2 if i % 3 == 0 else -1)} for i in range(55)],
    "visible_indicators": ["SMA 50", "Volume"],
    "support_levels": [128.0],
    "resistance_levels": [142.0],
    "patterns": ["ascending triangle", "higher lows"],
    "volume_observation": "Rising into breakout attempts",
    "reading_notes": "Clear axis labels",
}

VERDICT = {
    "direction": "up",
    "probability_up": 0.6,
    "confidence": "medium",
    "horizon": "1-4 weeks",
    "summary": "Uptrend intact; testing resistance.",
    "signals": [
        {"method": "Trend", "reading": "higher lows", "direction": "bullish", "weight": "high"},
        {"method": "RSI(14)", "reading": "64", "direction": "bullish", "weight": "medium"},
    ],
    "bull_case": "Break of 142 targets 150.",
    "bear_case": "Rejection at 142 back to 128.",
    "key_levels": "142 breakout / 128 invalidation",
    "risks": ["Earnings next week"],
}


def _message(content, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


class FakeAPI:
    def __init__(self, research_pauses=0, vision_stop="end_turn"):
        self.requests = []
        self.research_pauses = research_pauses
        self.vision_stop = vision_stop

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "headers": dict(request.headers)})
        if body.get("tools"):  # research step
            if self.research_pauses:
                self.research_pauses -= 1
                msg = _message(
                    [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "ACME"}}],
                    stop_reason="pause_turn",
                )
            else:
                msg = _message(
                    [
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/acme-news",
                                    "title": "Acme beats",
                                    "encrypted_content": "x",
                                    "page_age": None,
                                }
                            ],
                        },
                        {"type": "text", "text": "Acme beat earnings estimates."},
                    ]
                )
        elif isinstance(body["messages"][0]["content"], list):  # vision step
            msg = _message([{"type": "text", "text": json.dumps(READING)}], stop_reason=self.vision_stop)
        else:  # verdict step
            msg = _message([{"type": "text", "text": json.dumps(VERDICT)}])
        return httpx2.Response(200, json=msg)


def _bot(fake):
    client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(fake)),
    )
    return StockChartBot(client=client)


@pytest.fixture
def chart(tmp_path):
    p = tmp_path / "chart.png"
    p.write_bytes(PNG)
    return p


def test_full_pipeline(chart, monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    fake = FakeAPI(research_pauses=1)
    a = _bot(fake).analyze(chart)

    assert a.reading.ticker == "ACME"
    assert a.tech_from_chart.lean == "bullish"
    assert a.verdict.direction == "up"
    assert "earnings" in a.research
    assert a.sources == ["https://example.com/acme-news"]

    # vision + 2 research (one pause_turn resume) + verdict
    assert len(fake.requests) == 4
    vision = fake.requests[0]["body"]
    assert vision["model"] == "claude-opus-5-5"
    assert vision["fallbacks"] == "default"
    assert vision["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert vision["output_config"]["format"]["type"] == "json_schema"
    assert "server-side-fallback-2026-07-01" in fake.requests[0]["headers"]["anthropic-beta"]

    resumed = fake.requests[2]["body"]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]
    assert resumed[1]["content"][0]["type"] == "server_tool_use"

    text = render(a)
    assert "▲ UP" in text and "Not financial advice" in text


def test_no_research_flag(chart):
    fake = FakeAPI()
    a = _bot(fake).analyze(chart, do_research=False)
    assert len(fake.requests) == 2 and a.research == ""


def test_refusal_raises(chart):
    with pytest.raises(BotError, match="declined"):
        _bot(FakeAPI(vision_stop="refusal")).read_chart(chart)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "chart.bmp"
    p.write_bytes(b"x")
    with pytest.raises(BotError, match="Unsupported"):
        _bot(FakeAPI()).read_chart(p)

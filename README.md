# Stock Chart Bot

Give it a screenshot of a stock chart and it tells you whether the price is more likely to go **up, down or sideways**, and why.

> **Not financial advice.** Price prediction is probabilistic. Short-term moves are close to a coin flip, and the bot is told to say so.

## How it works

| Step | What happens |
|---|---|
| 1. Read the chart | Claude (vision) reads the screenshot and pulls out the ticker, timeframe, about 20–60 sampled prices, drawn indicators, support and resistance levels, chart patterns and volume. It returns structured JSON. |
| 2. Technical analysis | Plain-Python indicators run on those prices: linear-regression trend, price vs SMA 20/50, the 20/50 cross, RSI(14), MACD(12,26,9), Bollinger %B and 10-bar rate of change. Each indicator casts a bullish, bearish or neutral vote. If `ALPHA_VANTAGE_API_KEY` is set and a ticker is visible, the same methods also run on real daily closes as a cross-check. |
| 3. Research | Claude uses web search to find recent news, earnings, analyst changes, upcoming catalysts, market and sector conditions, and what TA research says about how reliable the detected patterns are. |
| 4. Verdict | Claude weighs all the evidence and returns a direction, P(up), a confidence level, a breakdown signal by signal, bull and bear cases, key levels and risks. |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export ALPHA_VANTAGE_API_KEY=...      # optional: real price data for cross-checking
```

## Usage

```bash
python -m stock_chart_bot path/to/chart.png
python -m stock_chart_bot chart.png --horizon "3 months"
python -m stock_chart_bot chart.png --no-research    # chart and indicators only, faster and cheaper
python -m stock_chart_bot chart.png --json           # machine-readable output
```

From Python:

```python
from stock_chart_bot import StockChartBot

analysis = StockChartBot().analyze("chart.png", horizon="1-4 weeks")
print(analysis.verdict.direction, analysis.verdict.probability_up)
```

Supported images: PNG, JPEG, GIF, WebP. For the best readings, use a chart with visible axis labels and the ticker in frame.

## Notes

- Model: `claude-opus-5-5` with server-side refusal fallback (`fallbacks: "default"`) turned on.
- Prices read off a screenshot are approximate, so the bot gives indicators computed from them less weight than indicators computed from real data.
- Each run makes about 3 Claude API calls (more if the web search runs long), and they are billed to your API key.

## Tests

```bash
pip install pytest
python -m pytest
```

The tests use the real SDK with a mocked HTTP transport, so they don't need an API key.

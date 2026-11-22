"""Command-line entry point: python -m stock_chart_bot chart.png"""

from __future__ import annotations

import argparse
import json
import sys

import anthropic

from .bot import Analysis, BotError, StockChartBot

DISCLAIMER = "Not financial advice. Chart-reading and indicators are probabilistic; do your own research."
ARROWS = {"up": "▲ UP", "down": "▼ DOWN", "sideways": "► SIDEWAYS"}


def render(a: Analysis) -> str:
    r, v = a.reading, a.verdict
    out = []
    title = r.ticker or r.company_or_asset or "Unknown asset"
    out.append(f"\n=== {title}  ({r.timeframe or 'timeframe unknown'}, {r.chart_type}) ===")
    out.append(f"Last price on chart: {r.current_price}")
    if r.patterns:
        out.append(f"Patterns: {', '.join(r.patterns)}")
    if r.support_levels or r.resistance_levels:
        out.append(f"Support: {r.support_levels}   Resistance: {r.resistance_levels}")

    out.append(f"\n--- Technicals ---\n{a.tech_from_chart.to_text()}")
    if a.tech_from_market:
        out.append(a.tech_from_market.to_text())

    out.append(f"\n--- Verdict ({a.horizon}) ---")
    out.append(f"{ARROWS[v.direction]}   P(up) ≈ {v.probability_up:.0%}   confidence: {v.confidence}")
    out.append(v.summary)
    out.append("\nSignals:")
    for s in v.signals:
        out.append(f"  [{s.direction:^7}] ({s.weight}) {s.method}: {s.reading}")
    out.append(f"\nBull case: {v.bull_case}")
    out.append(f"Bear case: {v.bear_case}")
    out.append(f"Key levels: {v.key_levels}")
    if v.risks:
        out.append("Risks:\n" + "\n".join(f"  - {x}" for x in v.risks))
    if a.sources:
        out.append("\nSources:\n" + "\n".join(f"  - {u}" for u in a.sources[:10]))
    for w in a.warnings:
        out.append(f"\nNote: {w}")
    out.append(f"\n{DISCLAIMER}\n")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="stock_chart_bot", description="Analyze a stock chart screenshot with Claude.")
    p.add_argument("image", help="Path to a PNG/JPEG/GIF/WebP screenshot of a price chart")
    p.add_argument("--horizon", default="1-4 weeks", help="Forecast horizon (default: '1-4 weeks')")
    p.add_argument("--no-research", action="store_true", help="Skip the web-search research step")
    p.add_argument("--json", action="store_true", help="Print the full analysis as JSON")
    args = p.parse_args(argv)

    bot = StockChartBot(log=lambda m: print(m, file=sys.stderr))
    try:
        analysis = bot.analyze(args.image, horizon=args.horizon, do_research=not args.no_research)
    except (BotError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("Error: no valid Anthropic credentials. Set ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("Error: rate limited by the Claude API; try again shortly.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"Error: Claude API returned {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("Error: could not reach the Claude API.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({**analysis.to_dict(), "disclaimer": DISCLAIMER}, indent=2))
    else:
        print(render(analysis))
    return 0


if __name__ == "__main__":
    sys.exit(main())

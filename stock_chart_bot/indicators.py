"""Classic technical-analysis methods, implemented in plain Python.

These run on either the prices Claude read off the screenshot (approximate) or
real daily closes from Alpha Vantage (when an API key is configured).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence


def sma(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def ema_series(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Wilder's RSI."""
    if len(values) <= period:
        return None
    gains, losses = [], []
    for prev, cur in zip(values[:-1], values[1:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) for the latest point, or None."""
    if len(values) < slow + signal:
        return None
    fast_e = ema_series(values, fast)
    slow_e = ema_series(values, slow)
    line = [f - s for f, s in zip(fast_e, slow_e)]
    sig = ema_series(line[slow - 1 :], signal)
    return line[-1], sig[-1], line[-1] - sig[-1]


def linear_slope_pct(values: Sequence[float]) -> Optional[float]:
    """Least-squares slope, as % of mean price per bar."""
    n = len(values)
    if n < 3:
        return None
    xs = range(n)
    mx = (n - 1) / 2
    my = sum(values) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0 or my == 0:
        return None
    return (num / den) / my * 100


def bollinger_position(values: Sequence[float], period: int = 20, k: float = 2.0) -> Optional[float]:
    """Where the last price sits in the band: 0 = lower band, 1 = upper band."""
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    std = (sum((v - mean) ** 2 for v in window) / period) ** 0.5
    if std == 0:
        return 0.5
    lower, upper = mean - k * std, mean + k * std
    return (values[-1] - lower) / (upper - lower)


def rate_of_change(values: Sequence[float], period: int = 10) -> Optional[float]:
    if len(values) <= period or values[-period - 1] == 0:
        return None
    return (values[-1] / values[-period - 1] - 1) * 100


@dataclass
class TechSignal:
    method: str
    value: str
    direction: str  # bullish | bearish | neutral
    score: int  # +1 / 0 / -1


@dataclass
class TechReport:
    source: str
    n_points: int
    signals: List[TechSignal]

    @property
    def score(self) -> int:
        return sum(s.score for s in self.signals)

    @property
    def lean(self) -> str:
        if not self.signals:
            return "neutral"
        ratio = self.score / len(self.signals)
        if ratio >= 0.25:
            return "bullish"
        if ratio <= -0.25:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_points": self.n_points,
            "score": self.score,
            "lean": self.lean,
            "signals": [asdict(s) for s in self.signals],
        }

    def to_text(self) -> str:
        lines = [f"Source: {self.source} ({self.n_points} points). Net score {self.score:+d} -> {self.lean}."]
        for s in self.signals:
            lines.append(f"- {s.method}: {s.value} [{s.direction}]")
        return "\n".join(lines)


def _sig(method: str, value: str, score: int) -> TechSignal:
    direction = {1: "bullish", -1: "bearish"}.get(score, "neutral")
    return TechSignal(method, value, direction, score)


def analyze(closes: Sequence[float], source: str) -> TechReport:
    """Run every method that has enough data and turn each into a +1/0/-1 vote."""
    closes = [float(c) for c in closes if c is not None]
    signals: List[TechSignal] = []
    if len(closes) < 3:
        return TechReport(source, len(closes), signals)
    last = closes[-1]

    slope = linear_slope_pct(closes)
    if slope is not None:
        score = 1 if slope > 0.05 else -1 if slope < -0.05 else 0
        signals.append(_sig("Trend (linear regression)", f"{slope:+.2f}% per bar", score))

    for period in (20, 50):
        avg = sma(closes, period)
        if avg is not None:
            diff = (last / avg - 1) * 100
            score = 1 if diff > 1 else -1 if diff < -1 else 0
            signals.append(_sig(f"Price vs SMA {period}", f"{diff:+.1f}% vs {avg:.2f}", score))

    s20, s50 = sma(closes, 20), sma(closes, 50)
    if s20 is not None and s50 is not None:
        score = 1 if s20 > s50 else -1
        signals.append(_sig("SMA 20/50 cross", "golden (20 above 50)" if score > 0 else "death (20 below 50)", score))

    r = rsi(closes)
    if r is not None:
        # Overbought/oversold are mean-reversion signals; mid-range RSI follows momentum.
        if r >= 70:
            score, note = -1, "overbought"
        elif r <= 30:
            score, note = 1, "oversold"
        elif r >= 55:
            score, note = 1, "bullish momentum"
        elif r <= 45:
            score, note = -1, "bearish momentum"
        else:
            score, note = 0, "neutral"
        signals.append(_sig("RSI(14)", f"{r:.1f} ({note})", score))

    m = macd(closes)
    if m is not None:
        line, sig, hist = m
        score = 1 if hist > 0 else -1 if hist < 0 else 0
        signals.append(_sig("MACD(12,26,9)", f"line {line:.3f}, signal {sig:.3f}, hist {hist:+.3f}", score))

    bb = bollinger_position(closes)
    if bb is not None:
        score = -1 if bb > 1 else 1 if bb < 0 else 0
        signals.append(_sig("Bollinger %B(20,2)", f"{bb:.2f}", score))

    roc = rate_of_change(closes)
    if roc is not None:
        score = 1 if roc > 1 else -1 if roc < -1 else 0
        signals.append(_sig("Rate of change (10)", f"{roc:+.1f}%", score))

    return TechReport(source, len(closes), signals)

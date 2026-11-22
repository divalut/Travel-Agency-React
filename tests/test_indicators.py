import math

from stock_chart_bot import indicators as ind


def test_sma_and_short_input():
    assert ind.sma([1, 2, 3, 4], 2) == 3.5
    assert ind.sma([1, 2], 5) is None


def test_rsi_extremes():
    up = [float(i) for i in range(1, 40)]
    down = list(reversed(up))
    assert ind.rsi(up) == 100.0
    assert ind.rsi(down) < 1
    assert ind.rsi(up[:10]) is None


def test_rsi_known_value():
    # Flat-ish alternating series should sit near 50
    series = [100 + (1 if i % 2 else -1) for i in range(60)]
    assert 40 < ind.rsi(series) < 60


def test_macd_sign_follows_trend():
    up = [100 * 1.01**i for i in range(60)]
    line, _sig, _hist = ind.macd(up)
    assert line > 0
    assert ind.macd(up[:20]) is None


def test_slope_and_bollinger():
    up = [float(i) for i in range(10, 40)]
    assert ind.linear_slope_pct(up) > 0
    assert ind.bollinger_position(up) > 0.8
    assert math.isclose(ind.bollinger_position([5.0] * 25), 0.5)


def test_analyze_uptrend_is_bullish_downtrend_is_bearish():
    # Gentle uptrend with noise so RSI isn't pinned at 100 (which reads as overbought)
    up = [100 + i * 0.5 + (2 if i % 3 == 0 else -1) for i in range(80)]
    down = [200 - i * 0.5 + (2 if i % 3 == 0 else -1) for i in range(80)]
    assert ind.analyze(up, "t").lean == "bullish"
    assert ind.analyze(down, "t").lean == "bearish"


def test_analyze_handles_tiny_input():
    rep = ind.analyze([1.0, 2.0], "t")
    assert rep.signals == [] and rep.lean == "neutral"
    assert "Net score" in rep.to_text()

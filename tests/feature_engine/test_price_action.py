import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.price_action import compute_price_action


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def test_bullish_engulfing():
    df = pd.DataFrame([
        _bar(100, 100.5, 99, 99),      # 阴
        _bar(98.5, 101.5, 98.4, 101),  # 阳，实体吞没前一根
    ])
    out = compute_price_action(df)
    assert out["pattern_engulfing_bull"].iloc[-1] == 1
    assert out["pattern_engulfing_bear"].iloc[-1] == 0


def test_bearish_engulfing():
    df = pd.DataFrame([
        _bar(99, 101, 99, 101),
        _bar(101.5, 101.6, 98, 98.5),
    ])
    out = compute_price_action(df)
    assert out["pattern_engulfing_bear"].iloc[-1] == 1


def test_inside_bar():
    df = pd.DataFrame([
        _bar(100, 105, 95, 102),
        _bar(101, 104, 96, 103),  # 完全在前一根 high/low 内
    ])
    out = compute_price_action(df)
    assert out["pattern_inside_bar"].iloc[-1] == 1


def test_doji():
    df = pd.DataFrame([_bar(100, 100.5, 99.5, 100.001)])
    out = compute_price_action(df)
    assert out["pattern_doji"].iloc[-1] == 1


def test_structure_and_bos():
    # 构造 HH-HL 上升结构后跌破前低 → BOS 下
    closes = [100, 102, 101, 104, 103, 106, 105, 108, 100]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    opens = [c - 0.2 for c in closes]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    out = compute_price_action(df)
    assert out["structure_regime"].iloc[-1] in {"uptrend", "downtrend", "range"}
    # 最后一根低于前几根，至少能产出 bos_flag=1 或 choch_flag=1 之一
    assert out["bos_flag"].iloc[-1] == 1 or out["choch_flag"].iloc[-1] == 1

from __future__ import annotations
import numpy as np
import pandas as pd


def _engulfing(o1, c1, o2, c2, bullish: bool) -> int:
    body1_top = max(o1, c1); body1_bot = min(o1, c1)
    body2_top = max(o2, c2); body2_bot = min(o2, c2)
    if bullish:
        return int(c1 < o1 and c2 > o2 and body2_top >= body1_top and body2_bot <= body1_bot)
    else:
        return int(c1 > o1 and c2 < o2 and body2_top >= body1_top and body2_bot <= body1_bot)


def _pinbar(o, h, l, c) -> int:
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = h - l
    if rng == 0:
        return 0
    return int((upper > 2 * body and upper / rng > 0.6) or (lower > 2 * body and lower / rng > 0.6))


def _doji(o, h, l, c) -> int:
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return 0
    return int(body / rng < 0.1)


def _marubozu(o, h, l, c) -> int:
    body = abs(c - o)
    rng = h - l
    if rng == 0:
        return 0
    return int(body / rng > 0.9)


def _swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """3-bar fractal swing detection: bar i is a swing high/low if it is
    strictly the max/min of its immediate ``lookback``-sized neighborhood.

    Note: a swing at index i is only *confirmed* once bar i+lookback exists
    (we need to see the bars after it to know it was a local extreme). Callers
    must not treat swing_h[i]/swing_l[i] as "known" until index i+lookback.
    """
    n = len(highs)
    swing_h = np.zeros(n, dtype=bool)
    swing_l = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max() and (highs[i] > highs[i - 1]):
            swing_h[i] = True
        if lows[i] == window_l.min() and (lows[i] < lows[i - 1]):
            swing_l[i] = True
    return swing_h, swing_l


def compute_price_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    engulf_bull = np.zeros(n, dtype=int)
    engulf_bear = np.zeros(n, dtype=int)
    inside = np.zeros(n, dtype=int)
    pin = np.zeros(n, dtype=int)
    doji = np.zeros(n, dtype=int)
    maru = np.zeros(n, dtype=int)

    for i in range(n):
        if i >= 1:
            engulf_bull[i] = _engulfing(o[i-1], c[i-1], o[i], c[i], True)
            engulf_bear[i] = _engulfing(o[i-1], c[i-1], o[i], c[i], False)
            inside[i] = int(h[i] <= h[i-1] and l[i] >= l[i-1])
        pin[i] = _pinbar(o[i], h[i], l[i], c[i])
        doji[i] = _doji(o[i], h[i], l[i], c[i])
        maru[i] = _marubozu(o[i], h[i], l[i], c[i])

    out["pattern_engulfing_bull"] = engulf_bull
    out["pattern_engulfing_bear"] = engulf_bear
    out["pattern_pinbar"] = pin
    out["pattern_inside_bar"] = inside
    out["pattern_doji"] = doji
    out["pattern_marubozu"] = maru

    # Swing points: 3-bar fractal. A swing at index j is only "known" once
    # bar j+1 has printed (need the next bar to confirm j was a local
    # extreme), so at row i we may only use swings with j <= i - 1.
    swing_lookback = 1
    swing_h_mask, swing_l_mask = _swing_points(h, l, lookback=swing_lookback)
    trail_window = 20

    swing_high_last = np.full(n, np.nan)
    swing_low_last = np.full(n, np.nan)
    for i in range(n):
        earliest = max(0, i - trail_window)
        for j in range(i - 1, earliest - 1, -1):
            if swing_h_mask[j]:
                swing_high_last[i] = h[j]
                break
        for j in range(i - 1, earliest - 1, -1):
            if swing_l_mask[j]:
                swing_low_last[i] = l[j]
                break
    out["swing_high_last"] = swing_high_last
    out["swing_low_last"] = swing_low_last

    # Structure regime + BOS / CHoCH（简化实现，仅用 i 时刻已知信息）
    structure = np.array(["range"] * n, dtype=object)
    bos = np.zeros(n, dtype=int)
    choch = np.zeros(n, dtype=int)
    prev_regime = "range"
    highs_seen: list[float] = []
    lows_seen: list[float] = []
    for i in range(n):
        # swing confirmed exactly as of this bar (needs confirm_idx + 1 == i)
        confirm_idx = i - 1
        if confirm_idx >= 0:
            if swing_h_mask[confirm_idx]:
                highs_seen.append(h[confirm_idx])
            if swing_l_mask[confirm_idx]:
                lows_seen.append(l[confirm_idx])

        recent_h = highs_seen[-2:]
        recent_l = lows_seen[-2:]
        if len(recent_h) == 2 and len(recent_l) == 2:
            if recent_h[-1] > recent_h[-2] and recent_l[-1] > recent_l[-2]:
                regime = "uptrend"
            elif recent_h[-1] < recent_h[-2] and recent_l[-1] < recent_l[-2]:
                regime = "downtrend"
            else:
                regime = "range"
        else:
            regime = "range"
        structure[i] = regime

        prior_swing_low = swing_low_last[i]
        prior_swing_high = swing_high_last[i]
        if prev_regime == "uptrend" and not np.isnan(prior_swing_low) and l[i] < prior_swing_low:
            bos[i] = 1
        if prev_regime == "downtrend" and not np.isnan(prior_swing_high) and h[i] > prior_swing_high:
            bos[i] = 1
        if regime != prev_regime and prev_regime != "range" and regime != "range":
            choch[i] = 1
        prev_regime = regime

    out["structure_regime"] = structure
    out["bos_flag"] = bos
    out["choch_flag"] = choch
    return out

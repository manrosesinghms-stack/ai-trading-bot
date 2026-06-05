"""
Builds a rich statistical knowledge base from 5 years of historical forex data.
This knowledge is injected into every Claude AI prompt so the model has deep
pair-specific context rather than relying on generic training knowledge alone.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from data_manager import load_pair, PAIRS

KB_FILE = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")

PIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCAD": 0.0001, "USDCHF": 0.0001, "NZDUSD": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,   "AUDJPY": 0.01,
    "XAUUSD": 0.1,
}

DAY_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _pip(pair: str) -> float:
    return PIP.get(pair, 0.0001)


def _trend(series: pd.Series) -> str:
    if len(series) < 2:
        return "Unknown"
    pct = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
    if pct > 4:   return "Bullish"
    if pct < -4:  return "Bearish"
    return "Sideways"


def _cluster_levels(levels: list[float], cluster_pct: float = 0.003) -> list[float]:
    """Merge nearby price levels into representative cluster midpoints."""
    if not levels:
        return []
    sorted_lvls = sorted(levels)
    clusters: list[list[float]] = [[sorted_lvls[0]]]
    for lv in sorted_lvls[1:]:
        if lv <= clusters[-1][-1] * (1 + cluster_pct):
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    return [round(float(np.mean(c)), 5) for c in clusters]


def analyze_pair(pair: str, df: pd.DataFrame) -> dict:
    df = df.dropna().copy()
    pip = _pip(pair)

    df["Range"]   = (df["High"] - df["Low"]) / pip
    df["Returns"] = df["Close"].pct_change()
    df["DOW"]     = pd.to_datetime(df.index).dayofweek
    df["Month"]   = pd.to_datetime(df.index).month

    adr_all  = float(df["Range"].mean())
    adr_1y   = float(df["Range"].tail(252).mean())
    adr_3m   = float(df["Range"].tail(63).mean())

    # Day-of-week volatility
    dow_vol = (
        df.groupby("DOW")["Range"].mean()
        .rename(index=DAY_NAMES)
        .to_dict()
    )
    dow_vol = {k: round(v, 1) for k, v in dow_vol.items() if isinstance(k, str)}

    # Monthly volatility (avg range per month across all years)
    month_vol = (
        df.groupby("Month")["Range"].mean()
        .rename(index=MONTH_NAMES)
        .to_dict()
    )
    month_vol = {k: round(v, 1) for k, v in month_vol.items()}

    # Monthly directional bias (% months that closed higher)
    df["MonthYear"] = pd.to_datetime(df.index).to_period("M")
    monthly_close = df.groupby("MonthYear")["Close"].last()
    monthly_bullish_pct = float((monthly_close.pct_change() > 0).mean() * 100)

    # Trend analysis at multiple timeframes
    current = float(df["Close"].iloc[-1])

    # Key S/R: local swing highs and lows over 5 years
    highs = df["High"]
    lows  = df["Low"]
    swing_highs = highs[(highs > highs.shift(3)) & (highs > highs.shift(-3))].tail(200).tolist()
    swing_lows  = lows[(lows  < lows.shift(3))  & (lows  < lows.shift(-3))].tail(200).tolist()

    resistance_raw = [h for h in swing_highs if h > current * 0.995]
    support_raw    = [l for l in swing_lows   if l < current * 1.005]

    resistance = sorted(_cluster_levels(resistance_raw), key=lambda x: abs(x - current))[:6]
    support    = sorted(_cluster_levels(support_raw),    key=lambda x: abs(x - current))[:6]

    # Volatility statistics
    annual_vol   = float(df["Returns"].std() * 252 ** 0.5 * 100)
    recent_vol   = float(df["Returns"].tail(20).std() * 252 ** 0.5 * 100)

    # Extreme moves
    weekly_ret = df["Close"].pct_change(5) * 100
    best_week  = round(float(weekly_ret.max()), 2)
    worst_week = round(float(weekly_ret.min()), 2)

    # Consecutive trend runs (how many bars it typically trends before reversing)
    df["Dir"] = np.sign(df["Returns"])
    runs = df["Dir"].ne(df["Dir"].shift()).cumsum()
    avg_run = float(df.groupby(runs).size().mean())

    # Best and worst months historically
    best_month  = max(month_vol, key=month_vol.get) if month_vol else "?"
    worst_month = min(month_vol, key=month_vol.get) if month_vol else "?"

    return {
        "data_start":       str(df.index[0].date()),
        "data_end":         str(df.index[-1].date()),
        "total_bars":       len(df),
        "current_price":    round(current, 5),
        "adr_5y_pips":      round(adr_all, 1),
        "adr_1y_pips":      round(adr_1y,  1),
        "adr_3m_pips":      round(adr_3m,  1),
        "adr_by_day":       dow_vol,
        "monthly_vol_pips": month_vol,
        "monthly_bullish_pct": round(monthly_bullish_pct, 1),
        "best_volatility_month":  best_month,
        "worst_volatility_month": worst_month,
        "trend_5y":    _trend(df["Close"]),
        "trend_1y":    _trend(df["Close"].tail(252)),
        "trend_6m":    _trend(df["Close"].tail(126)),
        "trend_3m":    _trend(df["Close"].tail(63)),
        "annual_vol_pct": round(annual_vol,  1),
        "recent_vol_pct": round(recent_vol,  1),
        "resistance_levels": resistance,
        "support_levels":    support,
        "best_week_pct":     best_week,
        "worst_week_pct":    worst_week,
        "avg_trend_run_bars": round(avg_run, 1),
    }


def calculate_correlations() -> dict[str, dict[str, float]]:
    closes: dict[str, pd.Series] = {}
    for pair in PAIRS:
        df = load_pair(pair, "1d")
        if df is not None and len(df) > 100:
            closes[pair] = df["Close"].rename(pair)

    if len(closes) < 2:
        return {}

    combined = pd.concat(closes.values(), axis=1).dropna()
    corr = combined.pct_change().corr()

    return {
        pair: {
            other: round(float(val), 3)
            for other, val in corr[pair].items()
            if other != pair and not np.isnan(val)
        }
        for pair in corr.columns
    }


def build_knowledge_base(progress_callback=None) -> dict:
    kb: dict = {}
    pairs = list(PAIRS.keys())

    for i, pair in enumerate(pairs):
        if progress_callback:
            progress_callback(i / (len(pairs) + 1), f"Analysing {pair}...")
        df = load_pair(pair, "1d")
        if df is None or len(df) < 100:
            print(f"  Skipping {pair} — no data cached")
            continue
        kb[pair] = analyze_pair(pair, df)

    if progress_callback:
        progress_callback((len(pairs)) / (len(pairs) + 1), "Calculating correlations...")

    kb["correlations"] = calculate_correlations()

    with open(KB_FILE, "w") as f:
        json.dump(kb, f, indent=2, default=str)

    return kb


def load_knowledge_base() -> dict:
    if not os.path.exists(KB_FILE):
        return {}
    with open(KB_FILE) as f:
        return json.load(f)


def format_for_ai(symbol: str, kb: dict) -> str:
    """Return a concise but information-dense knowledge block for the AI prompt."""
    if not kb or symbol not in kb:
        return ""

    d = kb[symbol]
    corr = kb.get("correlations", {}).get(symbol, {})

    pos_corr = sorted(
        [(p, v) for p, v in corr.items() if v > 0.65], key=lambda x: -x[1]
    )
    neg_corr = sorted(
        [(p, v) for p, v in corr.items() if v < -0.65], key=lambda x: x[1]
    )

    best_day  = max(d["adr_by_day"], key=d["adr_by_day"].get) if d.get("adr_by_day") else "?"
    worst_day = min(d["adr_by_day"], key=d["adr_by_day"].get) if d.get("adr_by_day") else "?"

    res_str = " | ".join(str(r) for r in d.get("resistance_levels", [])[:4])
    sup_str = " | ".join(str(s) for s in d.get("support_levels",    [])[:4])

    lines = [
        f"=== {symbol} HISTORICAL KNOWLEDGE ({d.get('data_start','?')} → {d.get('data_end','?')}, {d.get('total_bars','?')} daily bars) ===",
        "",
        "VOLATILITY PROFILE:",
        f"  ADR (5yr avg): {d.get('adr_5y_pips','?')} pips | 1yr avg: {d.get('adr_1y_pips','?')} pips | 3m avg: {d.get('adr_3m_pips','?')} pips",
        f"  Annual volatility: {d.get('annual_vol_pct','?')}% | Recent (20d): {d.get('recent_vol_pct','?')}%",
        f"  Highest-range day of week: {best_day} ({d['adr_by_day'].get(best_day,'?')} pips) | Lowest: {worst_day} ({d['adr_by_day'].get(worst_day,'?')} pips)",
        f"  Best volatility month: {d.get('best_volatility_month','?')} | Lowest: {d.get('worst_volatility_month','?')}",
        f"  Avg trend run before reversal: {d.get('avg_trend_run_bars','?')} bars",
        f"  Record week: +{d.get('best_week_pct','?')}% | Worst week: {d.get('worst_week_pct','?')}%",
        "",
        "TREND CONTEXT:",
        f"  5-Year: {d.get('trend_5y','?')} | 1-Year: {d.get('trend_1y','?')} | 6-Month: {d.get('trend_6m','?')} | 3-Month: {d.get('trend_3m','?')}",
        f"  Historically closes higher {d.get('monthly_bullish_pct','?')}% of months",
        "",
        "KEY PRICE LEVELS (from 5yr swing-point clusters):",
        f"  Resistance: {res_str if res_str else 'None identified above current price'}",
        f"  Support:    {sup_str if sup_str else 'None identified below current price'}",
        "",
        "CORRELATIONS (>0.65 threshold):",
        f"  Positive: {', '.join(f'{p}({v:+.2f})' for p,v in pos_corr) or 'None'}",
        f"  Negative: {', '.join(f'{p}({v:+.2f})' for p,v in neg_corr) or 'None'}",
    ]

    return "\n".join(lines)

"""
Multi-Timeframe (MTF) Confluence Analyzer.
Top-down framework: Higher TF sets the trend → Middle TF confirms structure
→ Entry TF provides the signal.

Weighting: Higher TF 40% | Middle TF 35% | Entry TF 25%
Score range: -1.0 (full Sell) → +1.0 (full Buy)
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from data_manager import PAIRS as YF_PAIRS

# Timeframe hierarchy: entry_tf → [entry, middle, higher]
TF_HIERARCHY: dict[str, list[str]] = {
    "M5":  ["M5",  "M15", "H1"],
    "M15": ["M15", "H1",  "H4"],
    "M30": ["M30", "H1",  "H4"],
    "H1":  ["H1",  "H4",  "D1"],
    "H4":  ["H4",  "D1",  "W1"],
    "D1":  ["D1",  "W1",  "MN"],
}

TF_WEIGHTS   = [0.25, 0.35, 0.40]   # entry, middle, higher

# Yahoo Finance interval / period mapping
YF_INTERVAL: dict[str, str] = {
    "M5":  "5m",  "M15": "15m", "M30": "30m",
    "H1":  "1h",  "H4":  "4h",  "D1":  "1d",
    "W1":  "1wk", "MN":  "1mo",
}
YF_PERIOD: dict[str, str] = {
    "M5":  "5d",  "M15": "10d", "M30": "20d",
    "H1":  "60d", "H4":  "180d","D1":  "1y",
    "W1":  "3y",  "MN":  "5y",
}

SIGNAL_SCORE = {"Buy": 1.0, "Sell": -1.0, "Hold": 0.0}


def _fetch_tf(pair: str, tf: str) -> pd.DataFrame | None:
    """Fetch OHLCV data for a given pair and timeframe via yfinance."""
    ticker   = YF_PAIRS.get(pair, pair + "=X")
    interval = YF_INTERVAL.get(tf)
    period   = YF_PERIOD.get(tf)
    if not interval or not period:
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df if len(df) >= 50 else None
    except Exception:
        return None


def analyze_mtf(pair: str, entry_tf: str = "H1") -> dict:
    """
    Run strategy analysis across the 3-timeframe hierarchy.
    Returns confluence score, signal, and per-TF breakdown.
    """
    from strategies import calculate_all_indicators, run_all_strategies

    tfs    = TF_HIERARCHY.get(entry_tf, ["H1", "H4", "D1"])
    labels = ["Entry TF", "Middle TF", "Higher TF"]

    results: list[dict] = []
    for tf in tfs:
        df = _fetch_tf(pair, tf)
        if df is None:
            results.append({
                "tf": tf, "label": labels[len(results)],
                "signal": "Hold", "score": 0.0,
                "buy_count": 0, "sell_count": 0,
                "strength": 0.0, "error": "No data",
            })
            continue
        try:
            df = calculate_all_indicators(df)
            res = run_all_strategies(df)
            results.append({
                "tf":         tf,
                "label":      labels[len(results)],
                "signal":     res["overall"],
                "score":      SIGNAL_SCORE.get(res["overall"], 0.0),
                "strength":   res["overall_strength"],
                "buy_count":  res["buy_count"],
                "sell_count": res["sell_count"],
                "hold_count": res["hold_count"],
                "last_price": res["last_price"],
            })
        except Exception as e:
            results.append({
                "tf": tf, "label": labels[len(results)],
                "signal": "Hold", "score": 0.0,
                "strength": 0.0, "error": str(e),
            })

    # Weighted confluence score
    weighted = sum(
        r["score"] * r["strength"] * w
        for r, w in zip(results, TF_WEIGHTS)
    )
    raw_score = sum(
        r["score"] * w for r, w in zip(results, TF_WEIGHTS)
    )

    # Signal labels
    if raw_score >= 0.55:
        signal, confidence = "Strong Buy",  min(raw_score, 1.0)
    elif raw_score >= 0.25:
        signal, confidence = "Buy",          raw_score
    elif raw_score <= -0.55:
        signal, confidence = "Strong Sell",  min(abs(raw_score), 1.0)
    elif raw_score <= -0.25:
        signal, confidence = "Sell",         abs(raw_score)
    else:
        signal, confidence = "Neutral",      0.0

    # Alignment check
    signals = [r["signal"] for r in results]
    all_agree = len(set(s for s in signals if s != "Hold")) == 1
    higher_tf_trend = results[-1]["signal"] if results else "Hold"

    return {
        "pair":             pair,
        "entry_tf":         entry_tf,
        "timeframes":       results,
        "confluence_score": round(raw_score, 3),
        "weighted_score":   round(weighted, 3),
        "signal":           signal,
        "confidence":       round(confidence, 3),
        "all_agree":        all_agree,
        "higher_tf_trend":  higher_tf_trend,
        "recommendation":   _build_recommendation(signal, all_agree, higher_tf_trend, results),
    }


def _build_recommendation(signal: str, all_agree: bool, htf: str, results: list) -> str:
    if all_agree and "Buy" in signal:
        return "✅ Full confluence — all timeframes bullish. High-probability long setup."
    if all_agree and "Sell" in signal:
        return "✅ Full confluence — all timeframes bearish. High-probability short setup."

    if htf == "Buy" and "Sell" in signal:
        return "⚠️ Counter-trend trade — higher TF bullish but entry TF bearish. Wait for H/M TF confirmation."
    if htf == "Sell" and "Buy" in signal:
        return "⚠️ Counter-trend trade — higher TF bearish but entry TF bullish. Wait for H/M TF confirmation."

    entry_sig = results[0]["signal"] if results else "Hold"
    mid_sig   = results[1]["signal"] if len(results) > 1 else "Hold"

    if htf == entry_sig and htf != "Hold":
        return f"🟡 Partial confluence — entry & higher TF agree ({htf}). Middle TF lagging. Can enter with caution."
    if signal == "Neutral":
        return "🟡 Mixed signals across timeframes — no clear edge. Wait for alignment."
    return f"Signal: {signal}. Monitor for full cross-timeframe alignment before entering."


def format_mtf_for_ai(pair: str, entry_tf: str) -> str:
    """Return formatted MTF context block for AI prompt injection."""
    result = analyze_mtf(pair, entry_tf)

    lines = [f"MULTI-TIMEFRAME CONFLUENCE ({pair}):"]
    for r in result["timeframes"]:
        icon = "🟢" if r["signal"] == "Buy" else "🔴" if r["signal"] == "Sell" else "🟡"
        err  = f" [{r.get('error','')}]" if r.get("error") else ""
        lines.append(
            f"  {icon} {r['label']} ({r['tf']}): {r['signal']}"
            f" ({r.get('buy_count',0)}B/{r.get('sell_count',0)}S){err}"
        )
    lines.append(f"  Confluence Score:  {result['confluence_score']:+.3f}")
    lines.append(f"  MTF Signal:        {result['signal']} (confidence {result['confidence']:.2f})")
    lines.append(f"  Higher-TF Trend:   {result['higher_tf_trend']}")
    lines.append(f"  Recommendation:    {result['recommendation']}")
    return "\n".join(lines)

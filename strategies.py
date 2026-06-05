"""
Technical strategies — uses the `ta` library (pip install ta).
Replaced pandas_ta which has Python version compatibility issues.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import ta
from ta.trend import (EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator)
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange


# ─────────────────────── indicator calculation ───────────────────────────────

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["EMA20"]  = EMAIndicator(close, window=20,  fillna=False).ema_indicator()
    df["EMA50"]  = EMAIndicator(close, window=50,  fillna=False).ema_indicator()
    df["EMA200"] = EMAIndicator(close, window=200, fillna=False).ema_indicator()
    df["SMA20"]  = SMAIndicator(close, window=20,  fillna=False).sma_indicator()
    df["SMA50"]  = SMAIndicator(close, window=50,  fillna=False).sma_indicator()

    # ── MACD ───────────────────────────────────────────────────────────────────
    _macd = MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df["MACD"]        = _macd.macd()
    df["MACD_Signal"] = _macd.macd_signal()
    df["MACD_Hist"]   = _macd.macd_diff()

    # ── RSI ────────────────────────────────────────────────────────────────────
    df["RSI"]      = RSIIndicator(close, window=14, fillna=False).rsi()
    df["RSI_Fast"] = RSIIndicator(close, window=7,  fillna=False).rsi()

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    _bb = BollingerBands(close, window=20, window_dev=2, fillna=False)
    df["BB_Upper"]  = _bb.bollinger_hband()
    df["BB_Lower"]  = _bb.bollinger_lband()
    df["BB_Middle"] = _bb.bollinger_mavg()
    df["BB_Width"]  = _bb.bollinger_wband()     # (upper-lower)/middle
    df["BB_Percent"] = _bb.bollinger_pband()    # (close-lower)/(upper-lower)

    # ── ATR ────────────────────────────────────────────────────────────────────
    df["ATR"] = AverageTrueRange(high, low, close, window=14, fillna=False).average_true_range()

    # ── Stochastic ─────────────────────────────────────────────────────────────
    _stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3, fillna=False)
    df["Stoch_K"] = _stoch.stoch()
    df["Stoch_D"] = _stoch.stoch_signal()

    # ── ADX ────────────────────────────────────────────────────────────────────
    _adx = ADXIndicator(high, low, close, window=14, fillna=False)
    df["ADX"]      = _adx.adx()
    df["DI_Plus"]  = _adx.adx_pos()
    df["DI_Minus"] = _adx.adx_neg()

    # ── CCI ────────────────────────────────────────────────────────────────────
    df["CCI"] = CCIIndicator(high, low, close, window=20, fillna=False).cci()

    # ── Williams %R ────────────────────────────────────────────────────────────
    df["Williams_R"] = WilliamsRIndicator(high, low, close, lbp=14, fillna=False).williams_r()

    # ── Pivot points (manual) ──────────────────────────────────────────────────
    df["Pivot"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    df["R1"] = 2 * df["Pivot"] - df["Low"].shift(1)
    df["S1"] = 2 * df["Pivot"] - df["High"].shift(1)
    df["R2"] = df["Pivot"] + (df["High"].shift(1) - df["Low"].shift(1))
    df["S2"] = df["Pivot"] - (df["High"].shift(1) - df["Low"].shift(1))

    return df


# ─────────────────────── helpers ──────────────────────────────────────────────

def _result(signal: str, strength: float, reasons: list[str]) -> dict:
    return {
        "signal":   signal,
        "strength": round(min(max(float(strength), 0.0), 1.0), 3),
        "reason":   " | ".join(reasons),
    }


# ─────────────────────── individual strategies ────────────────────────────────

def rsi_strategy(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi = float(last["RSI"])
    reasons = [f"RSI {rsi:.1f}"]

    if rsi < 25:
        return _result("Buy",  (25 - rsi) / 25,  reasons + ["Extreme oversold"])
    if rsi < 30 and float(prev["RSI"]) >= 30:
        return _result("Buy",  0.65, reasons + ["Crossed below 30 (oversold)"])
    if rsi > 75:
        return _result("Sell", (rsi - 75) / 25,  reasons + ["Extreme overbought"])
    if rsi > 70 and float(prev["RSI"]) <= 70:
        return _result("Sell", 0.65, reasons + ["Crossed above 70 (overbought)"])
    reasons.append("Neutral zone")
    return _result("Hold", 0, reasons)


def macd_strategy(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    reasons = []

    macd_l = float(last["MACD"]);  sig_l = float(last["MACD_Signal"]); hist_l = float(last["MACD_Hist"])
    macd_p = float(prev["MACD"]);  sig_p = float(prev["MACD_Signal"])

    bullish_cross = macd_p <= sig_p and macd_l > sig_l
    bearish_cross = macd_p >= sig_p and macd_l < sig_l
    hist_str = abs(hist_l) / (abs(macd_l) + 1e-8)

    if bullish_cross:
        reasons.append("MACD bullish crossover")
        return _result("Buy",  min(hist_str, 0.85), reasons)
    if bearish_cross:
        reasons.append("MACD bearish crossover")
        return _result("Sell", min(hist_str, 0.85), reasons)

    above = macd_l > sig_l
    reasons.append("MACD above signal — bullish" if above else "MACD below signal — bearish")
    if hist_l > 0 and hist_l > float(prev["MACD_Hist"]):
        reasons.append("Momentum rising")
    elif hist_l < 0 and hist_l < float(prev["MACD_Hist"]):
        reasons.append("Momentum falling")
    return _result("Hold", 0, reasons)


def ema_crossover_strategy(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    reasons = []

    trend = "Bullish" if float(last["Close"]) > float(last["EMA200"]) else "Bearish"
    reasons.append(f"Price {'above' if trend=='Bullish' else 'below'} EMA200 ({trend} trend)")

    golden = float(prev["EMA20"]) <= float(prev["EMA50"]) and float(last["EMA20"]) > float(last["EMA50"])
    death  = float(prev["EMA20"]) >= float(prev["EMA50"]) and float(last["EMA20"]) < float(last["EMA50"])

    if golden:
        reasons.append("Golden cross: EMA20 crossed above EMA50")
        return _result("Buy",  0.80, reasons)
    if death:
        reasons.append("Death cross: EMA20 crossed below EMA50")
        return _result("Sell", 0.80, reasons)

    ema_bull = float(last["EMA20"]) > float(last["EMA50"])
    reasons.append("EMA20 above EMA50 (short-term bullish)" if ema_bull else "EMA20 below EMA50 (short-term bearish)")

    if trend == "Bullish" and ema_bull:
        return _result("Buy",  0.45, reasons)
    if trend == "Bearish" and not ema_bull:
        return _result("Sell", 0.45, reasons)
    return _result("Hold", 0, reasons)


def bollinger_band_strategy(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    bb_pct = float(last["BB_Percent"])
    bb_w   = float(last["BB_Width"])
    reasons = [f"BB%={bb_pct:.2f}, Width={bb_w:.3f}"]

    avg_w = df["BB_Width"].rolling(50).mean().iloc[-1]
    if bb_w < avg_w * 0.6:
        reasons.append("BB Squeeze — breakout imminent")

    touched_lower = float(last["Close"]) <= float(last["BB_Lower"]) and float(prev["Close"]) > float(prev["BB_Lower"])
    touched_upper = float(last["Close"]) >= float(last["BB_Upper"]) and float(prev["Close"]) < float(prev["BB_Upper"])

    if touched_lower:
        reasons.append("Price pierced lower band — reversal signal")
        return _result("Buy",  0.70, reasons)
    if touched_upper:
        reasons.append("Price pierced upper band — reversal signal")
        return _result("Sell", 0.70, reasons)
    if bb_pct < 0.1:
        return _result("Buy",  0.40, reasons + ["Near lower band"])
    if bb_pct > 0.9:
        return _result("Sell", 0.40, reasons + ["Near upper band"])
    return _result("Hold", 0, reasons)


def adx_trend_strategy(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    adx = float(last["ADX"]); dip = float(last["DI_Plus"]); dim = float(last["DI_Minus"])
    reasons = [f"ADX={adx:.1f}, +DI={dip:.1f}, -DI={dim:.1f}"]

    if adx > 25:
        reasons.append("Strong trend")
        if dip > dim:
            return _result("Buy",  min(adx / 50, 1.0), reasons + ["Bullish (+DI > -DI)"])
        return _result("Sell", min(adx / 50, 1.0), reasons + ["Bearish (-DI > +DI)"])
    if adx > 20:
        reasons.append("Developing trend")
    else:
        reasons.append("Ranging market — avoid trend strategies")
    return _result("Hold", 0, reasons)


def stochastic_strategy(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    k = float(last["Stoch_K"]); d = float(last["Stoch_D"])
    reasons = [f"K={k:.1f}, D={d:.1f}"]

    if k < 20 and d < 20:
        if k > d and float(prev["Stoch_K"]) <= float(prev["Stoch_D"]):
            return _result("Buy",  (20 - k) / 20, reasons + ["Bullish crossover in oversold zone"])
        return _result("Hold", 0, reasons + ["Oversold — waiting for crossover"])
    if k > 80 and d > 80:
        if k < d and float(prev["Stoch_K"]) >= float(prev["Stoch_D"]):
            return _result("Sell", (k - 80) / 20, reasons + ["Bearish crossover in overbought zone"])
        return _result("Hold", 0, reasons + ["Overbought — waiting for crossover"])
    return _result("Hold", 0, reasons + ["Neutral"])


def fvg_strategy(df: pd.DataFrame) -> dict:
    """Fair Value Gap — ICT Smart Money (verified 3-0 by adversarial research)."""
    if len(df) < 5:
        return _result("Hold", 0, ["Not enough data for FVG"])

    c1 = df.iloc[-3]; c3 = df.iloc[-1]
    bullish_fvg = float(c1["High"]) < float(c3["Low"])
    bearish_fvg = float(c1["Low"])  > float(c3["High"])

    if bullish_fvg:
        gap = float(c3["Low"]) - float(c1["High"])
        return _result("Buy",  0.72, [f"Bullish FVG formed: gap {c1['High']:.5f}–{c3['Low']:.5f} ({gap:.5f})"])
    if bearish_fvg:
        gap = float(c1["Low"]) - float(c3["High"])
        return _result("Sell", 0.72, [f"Bearish FVG formed: gap {c3['High']:.5f}–{c1['Low']:.5f} ({gap:.5f})"])

    # Check if price is inside a historical FVG zone (lookback 30 bars)
    current = float(df.iloc[-1]["Close"])
    for i in range(2, min(30, len(df) - 1)):
        ca = df.iloc[-i - 1]; cc = df.iloc[-i + 1]
        if float(ca["High"]) < float(cc["Low"]):
            bot, top = float(ca["High"]), float(cc["Low"])
            if bot <= current <= top:
                return _result("Buy",  0.52, [f"Price inside bullish FVG zone ({bot:.5f}–{top:.5f}, {i} bars old)"])
        if float(ca["Low"]) > float(cc["High"]):
            bot2, top2 = float(cc["High"]), float(ca["Low"])
            if bot2 <= current <= top2:
                return _result("Sell", 0.52, [f"Price inside bearish FVG zone ({bot2:.5f}–{top2:.5f}, {i} bars old)"])

    return _result("Hold", 0, ["No active Fair Value Gap"])


def price_action_strategy(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    body = abs(float(last["Close"]) - float(last["Open"]))
    full = float(last["High"]) - float(last["Low"])
    if full == 0:
        return _result("Hold", 0, ["Doji — no range"])

    upper_wick = float(last["High"]) - max(float(last["Close"]), float(last["Open"]))
    lower_wick  = min(float(last["Close"]), float(last["Open"])) - float(last["Low"])
    body_ratio  = body / full

    # Bullish engulfing
    if (float(prev["Close"]) < float(prev["Open"]) and float(last["Close"]) > float(last["Open"])
            and float(last["Close"]) > float(prev["Open"]) and float(last["Open"]) < float(prev["Close"])):
        return _result("Buy", 0.75, ["Bullish engulfing pattern"])

    # Bearish engulfing
    if (float(prev["Close"]) > float(prev["Open"]) and float(last["Close"]) < float(last["Open"])
            and float(last["Close"]) < float(prev["Open"]) and float(last["Open"]) > float(prev["Close"])):
        return _result("Sell", 0.75, ["Bearish engulfing pattern"])

    # Hammer
    if lower_wick > 2 * body and lower_wick > upper_wick * 2 and body_ratio < 0.35:
        return _result("Buy",  0.60, ["Bullish hammer / pin bar"])

    # Shooting star
    if upper_wick > 2 * body and upper_wick > lower_wick * 2 and body_ratio < 0.35:
        return _result("Sell", 0.60, ["Bearish shooting star / pin bar"])

    return _result("Hold", 0, [f"No clear pattern (body ratio {body_ratio:.2f})"])


# ─────────────────────── combined runner ──────────────────────────────────────

def run_all_strategies(df: pd.DataFrame) -> dict:
    strats = {
        "RSI":             rsi_strategy(df),
        "MACD":            macd_strategy(df),
        "EMA Crossover":   ema_crossover_strategy(df),
        "Bollinger Bands": bollinger_band_strategy(df),
        "ADX Trend":       adx_trend_strategy(df),
        "Stochastic":      stochastic_strategy(df),
        "Price Action":    price_action_strategy(df),
        "Fair Value Gap":  fvg_strategy(df),
    }

    buy_count  = sum(1 for s in strats.values() if s["signal"] == "Buy")
    sell_count = sum(1 for s in strats.values() if s["signal"] == "Sell")
    hold_count = sum(1 for s in strats.values() if s["signal"] == "Hold")
    buy_str    = sum(s["strength"] for s in strats.values() if s["signal"] == "Buy")
    sell_str   = sum(s["strength"] for s in strats.values() if s["signal"] == "Sell")

    # Confidence = average strength of agreeing signals, BOOSTED by consensus
    # breadth (more strategies agreeing → higher conviction). Capped at 1.0.
    def _conf(total_str: float, count: int) -> float:
        if count == 0:
            return 0.0
        avg = total_str / count
        return min(1.0, avg * (1 + 0.20 * (count - 1)))

    if buy_count > sell_count and buy_str > sell_str:
        overall, strength = "Buy",  _conf(buy_str,  buy_count)
    elif sell_count > buy_count and sell_str > buy_str:
        overall, strength = "Sell", _conf(sell_str, sell_count)
    else:
        overall, strength = "Hold", 0.0

    return {
        "individual":       strats,
        "overall":          overall,
        "overall_strength": round(strength, 3),
        "buy_count":        buy_count,
        "sell_count":       sell_count,
        "hold_count":       hold_count,
        "last_price":       float(df["Close"].iloc[-1]),
        "atr":              float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0,
    }


def get_support_resistance(df: pd.DataFrame, lookback: int = 50) -> dict:
    recent = df.tail(lookback)
    current = float(df["Close"].iloc[-1])
    return {
        "resistance": sorted([float(x) for x in recent["High"].nlargest(3).tolist()], reverse=True),
        "support":    sorted([float(x) for x in recent["Low"].nsmallest(3).tolist()]),
        "pivot":      (float(df["High"].iloc[-1]) + float(df["Low"].iloc[-1]) + float(df["Close"].iloc[-1])) / 3,
    }

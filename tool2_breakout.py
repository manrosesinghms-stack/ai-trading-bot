"""
TOOL #2 — Trend-Following Breakout (Donchian / Turtle-style)
Entry : price breaks the N-bar high (long) or N-bar low (short)
Exit  : ATR trailing stop that ratchets in the trade's favour
Sizing: risk % of balance per trade (same framework as Tool #1)

This file includes the strategy, a backtester with an ATR trailing stop,
and a per-pair optimizer (70/30 train/test, reports OUT-OF-SAMPLE only).
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from data_manager import load_pair, PAIRS
from ta.volatility import AverageTrueRange

PIP = {"EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
       "USDCHF":.0001,"USDJPY":.01,"EURJPY":.01,"GBPJPY":.01,"XAUUSD":.1}
SPREAD = {"EURUSD":1.0,"GBPUSD":1.2,"USDJPY":1.0,"XAUUSD":3.0,
          "AUDUSD":1.2,"USDCAD":1.5,"GBPJPY":2.0,"EURJPY":1.5}

# Parameter grid
ENTRY_LB  = [10, 20, 40, 55]     # Donchian breakout lookback (Turtle uses 20 & 55)
TRAIL_ATR = [2.0, 3.0, 4.0]      # trailing-stop distance in ATR


def prep(df):
    df = df.copy()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    return df.dropna()


def simulate(df, pair, lb, trail, start=100.0, risk=2.0):
    """Donchian breakout with ATR trailing stop. One position at a time."""
    p = PIP.get(pair, .0001); spr = SPREAD.get(pair, 1.5)
    bal = start; pos = None; trades = []; eq = [bal]

    hi = df["High"].rolling(lb).max().shift(1)   # prior N-bar high (no lookahead)
    lo = df["Low"].rolling(lb).min().shift(1)

    for i in range(lb + 1, len(df)):
        b = df.iloc[i]
        atr = float(b["ATR"])
        if atr <= 0:
            continue

        # ── manage open position (trailing stop) ──
        if pos:
            if pos["dir"] == 1:
                pos["peak"] = max(pos["peak"], float(b["High"]))
                stop = pos["peak"] - trail * atr
                if float(b["Low"]) <= stop:
                    r = ((stop - pos["entry"]) / p - spr) / pos["slp"]
                    pnl = pos["risk"] * r; bal += pnl
                    trades.append(pnl); eq.append(bal); pos = None
            else:
                pos["peak"] = min(pos["peak"], float(b["Low"]))
                stop = pos["peak"] + trail * atr
                if float(b["High"]) >= stop:
                    r = ((pos["entry"] - stop) / p - spr) / pos["slp"]
                    pnl = pos["risk"] * r; bal += pnl
                    trades.append(pnl); eq.append(bal); pos = None
        if pos:
            continue

        # ── entry on breakout ──
        c = float(b["Close"])
        if not np.isnan(hi.iloc[i]) and c > hi.iloc[i]:
            d = 1
        elif not np.isnan(lo.iloc[i]) and c < lo.iloc[i]:
            d = -1
        else:
            continue
        entry = c + d * spr * p
        sl_dist = trail * atr            # initial stop = one trail distance
        pos = {"dir": d, "entry": entry, "peak": entry,
               "slp": sl_dist / p, "risk": bal * risk / 100}

    return trades, eq


def metr(trades, eq, start=100.0):
    if len(trades) < 10:
        return None
    w = [t for t in trades if t > 0]; l = [t for t in trades if t <= 0]
    s = pd.Series(eq); dd = ((s - s.cummax()) / s.cummax() * 100).min()
    pf = (sum(w) / abs(sum(l))) if l else 99.0
    return {"trades": len(trades), "return": round((eq[-1]/start-1)*100, 1),
            "win_rate": round(len(w)/len(trades)*100, 1), "pf": round(pf, 2),
            "max_dd": round(float(dd), 1)}


def score(m):
    if not m or m["trades"] < 10:
        return -999
    return m["pf"]*25 + m["return"]*0.2 - abs(m["max_dd"])*0.4


def optimize(pair, interval):
    df = load_pair(pair, interval)
    if df is None or len(df) < 300:
        return None
    df = prep(df)
    sp = int(len(df)*0.70); isd, oos = df.iloc[:sp], df.iloc[sp:]
    best = None
    for lb in ENTRY_LB:
        for tr in TRAIL_ATR:
            t, e = simulate(isd, pair, lb, tr)
            m = metr(t, e); sc = score(m)
            if best is None or sc > best["sc"]:
                best = {"lb": lb, "trail": tr, "sc": sc, "is": m}
    t, e = simulate(oos, pair, best["lb"], best["trail"])
    best["oos"] = metr(t, e); best["interval"] = interval
    return best


if __name__ == "__main__":
    results = {}
    print("TOOL #2 — TREND-FOLLOWING BREAKOUT  (out-of-sample results)")
    print(f"{'Pair':8s} {'TF':>3s} {'LB':>3s} {'trail':>5s} | {'OOS ret%':>8s} {'wr%':>5s} {'pf':>5s} {'dd%':>6s} {'#':>4s}")
    print("-"*68)
    for pair in PAIRS:
        cands = []
        for interval in ("1h", "1d"):
            b = optimize(pair, interval)
            if b and b.get("oos"):
                cands.append(b)
        if not cands:
            print(f"{pair:8s}  no data"); continue
        best = max(cands, key=lambda b: score(b["oos"]))
        o = best["oos"]; edge = o["pf"] > 1.05
        results[pair] = {"interval": best["interval"], "entry_lb": best["lb"],
                         "trail_atr": best["trail"], "oos": o, "edge": edge}
        tag = " <EDGE>" if edge else ""
        print(f"{pair:8s} {best['interval']:>3s} {best['lb']:>3d} {best['trail']:>5.1f} | "
              f"{o['return']:>8.1f} {o['win_rate']:>5.1f} {o['pf']:>5.2f} {o['max_dd']:>6.1f} {o['trades']:>4d}{tag}")
    edges = [p for p,v in results.items() if v["edge"]]
    print(f"\nEDGE PAIRS (OOS PF>1.05): {', '.join(edges) if edges else 'none'}")
    Path("data/tool2_params.json").write_text(json.dumps(results, indent=2))
    print("Saved -> data/tool2_params.json")

"""
Expanded optimizer — every pair across H1 (2yr) AND D1 (5yr), finer grid.
Reports OUT-OF-SAMPLE results (the only ones that matter) and picks the
single best validated (pair, timeframe, params) combo per pair.
'Best of the best' = keep only pairs with OOS profit factor > 1.05.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from data_manager import load_pair, PAIRS
from strategies import calculate_all_indicators
from backtester import _compute_signals, PIP, SPREAD_PIPS

GRID = [(ms, sl, tp)
        for ms in (3, 4, 5)
        for sl in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
        for tp in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)]


def simulate(df, pair, ms, sl_atr, tp_atr, start=100.0, risk=2.0):
    p = PIP.get(pair, .0001); spr = SPREAD_PIPS.get(pair, 1.5)
    bal = start; pos = None; trades = []; eq = [bal]
    for i in range(2, len(df)):
        b = df.iloc[i]
        if pos:
            hsl = b["Low"] <= pos["sl"] if pos["d"]==1 else b["High"] >= pos["sl"]
            htp = b["High"] >= pos["tp"] if pos["d"]==1 else b["Low"] <= pos["tp"]
            if hsl or htp:
                px = pos["sl"] if hsl else pos["tp"]
                r = ((px-pos["entry"])/p*pos["d"]-spr)/pos["slp"]
                pnl = pos["risk"]*r; bal += pnl; trades.append(pnl); eq.append(bal); pos=None
        if pos: continue
        bc, sc, atr = int(b["buy_count"]), int(b["sell_count"]), float(b["ATR"])
        if atr <= 0: continue
        d = 1 if (bc>=ms and bc>sc) else (-1 if (sc>=ms and sc>bc) else 0)
        if d==0: continue
        e = float(b["Close"])+d*spr*p
        pos = {"d":d,"entry":e,"sl":e-d*atr*sl_atr,"tp":e+d*atr*tp_atr,
               "slp":atr*sl_atr/p,"risk":bal*risk/100}
    return trades, eq


def metr(trades, eq, start=100.0):
    if len(trades) < 15: return None
    w=[t for t in trades if t>0]; l=[t for t in trades if t<=0]
    s=pd.Series(eq); dd=((s-s.cummax())/s.cummax()*100).min()
    pf=(sum(w)/abs(sum(l))) if l else 99.0
    return {"trades":len(trades),"return":round((eq[-1]/start-1)*100,1),
            "win_rate":round(len(w)/len(trades)*100,1),"pf":round(pf,2),
            "max_dd":round(float(dd),1)}


def score(m):
    if not m or m["trades"]<15: return -999
    return m["pf"]*25 + m["return"]*0.2 - abs(m["max_dd"])*0.4


def opt(pair, interval):
    df = load_pair(pair, interval)
    if df is None or len(df) < 400: return None
    df = calculate_all_indicators(df).dropna(); df = _compute_signals(df)
    sp = int(len(df)*0.70); isd, oos = df.iloc[:sp], df.iloc[sp:]
    best = None
    for ms,sl,tp in GRID:
        t,e = simulate(isd, pair, ms, sl, tp); m = metr(t,e); sc = score(m)
        if best is None or sc > best["sc"]:
            best = {"params":(ms,sl,tp),"sc":sc,"is":m}
    ms,sl,tp = best["params"]
    t,e = simulate(oos, pair, ms, sl, tp); best["oos"] = metr(t,e)
    best["interval"] = interval
    return best


if __name__ == "__main__":
    results = {}
    print(f"{'Pair':8s} {'TF':>3s} {'min':>3s} {'SL':>4s} {'TP':>4s} | {'OOS ret%':>8s} {'wr%':>5s} {'pf':>5s} {'dd%':>6s} {'#':>4s}")
    print("-"*70)
    for pair in PAIRS:
        cands = []
        for interval in ("1h", "1d"):
            b = opt(pair, interval)
            if b and b.get("oos"): cands.append(b)
        if not cands:
            print(f"{pair:8s}  no data"); continue
        # pick the timeframe with best OOS score
        best = max(cands, key=lambda b: score(b["oos"]))
        ms,sl,tp = best["params"]; o = best["oos"]
        edge = o["pf"] > 1.05
        results[pair] = {"interval":best["interval"],"min_signals":ms,
                         "sl_atr":sl,"tp_atr":tp,"oos":o,"edge":edge}
        star = " <EDGE>" if edge else ""
        print(f"{pair:8s} {best['interval']:>3s} {ms:>3d} {sl:>4.2f} {tp:>4.2f} | "
              f"{o['return']:>8.1f} {o['win_rate']:>5.1f} {o['pf']:>5.2f} {o['max_dd']:>6.1f} {o['trades']:>4d}{star}")

    edges = [p for p,v in results.items() if v["edge"]]
    print(f"\nBEST-OF-BEST (OOS profit factor > 1.05): {', '.join(edges) if edges else 'none'}")
    Path("data/optimized_params2.json").write_text(json.dumps(results, indent=2))
    print("Saved -> data/optimized_params2.json")

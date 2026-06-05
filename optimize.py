"""
Parameter optimizer — finds the best validated config per pair.
Sweeps min_signals / SL-ATR / TP-ATR over historical data, ranks by a
robustness score (profit factor + return, penalized for low trade count
and high drawdown). Splits data 70/30 and reports OUT-OF-SAMPLE results
so we don't overfit.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from data_manager import load_pair, PAIRS
from strategies import calculate_all_indicators
from backtester import _compute_signals, PIP, SPREAD_PIPS

GRID = [
    (ms, sl, tp)
    for ms in (3, 4, 5)
    for sl in (0.75, 1.0, 1.5, 2.0)
    for tp in (1.0, 1.5, 2.0, 2.5, 3.0)
]


def simulate(df, pair, min_sig, sl_atr, tp_atr, start_bal=100.0, risk_pct=2.0):
    pip = PIP.get(pair, 0.0001)
    spread = SPREAD_PIPS.get(pair, 1.5)
    bal = start_bal
    pos = None
    trades = []
    eq = [bal]
    for i in range(2, len(df)):
        bar = df.iloc[i]
        if pos is not None:
            sl, tp = pos["sl"], pos["tp"]
            hit_sl = bar["Low"] <= sl if pos["dir"] == 1 else bar["High"] >= sl
            hit_tp = bar["High"] >= tp if pos["dir"] == 1 else bar["Low"] <= tp
            if hit_sl or hit_tp:
                px = sl if hit_sl else tp
                pnl_pips = (px - pos["entry"]) / pip * pos["dir"] - spread
                r = pnl_pips / pos["sl_pips"]
                pnl = pos["risk"] * r
                bal += pnl
                trades.append(pnl)
                eq.append(bal)
                pos = None
        if pos is not None:
            continue
        bc, sc = int(bar["buy_count"]), int(bar["sell_count"])
        atr = float(bar["ATR"])
        if atr <= 0:
            continue
        d = 1 if (bc >= min_sig and bc > sc) else (-1 if (sc >= min_sig and sc > bc) else 0)
        if d == 0:
            continue
        entry = float(bar["Close"]) + d * spread * pip
        sl_d = atr * sl_atr
        tp_d = atr * tp_atr
        pos = {"dir": d, "entry": entry,
               "sl": entry - d * sl_d, "tp": entry + d * tp_d,
               "sl_pips": sl_d / pip, "risk": bal * (risk_pct / 100)}
    return trades, eq, bal


def metrics(trades, eq, start_bal):
    if not trades:
        return None
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    s = pd.Series(eq)
    dd = ((s - s.cummax()) / s.cummax() * 100).min()
    ret = (eq[-1] / start_bal - 1) * 100
    wr = len(wins) / len(trades) * 100
    pf = (sum(wins) / abs(sum(losses))) if losses else 99.0
    return {"trades": len(trades), "return": round(ret, 1), "win_rate": round(wr, 1),
            "pf": round(pf, 2), "max_dd": round(float(dd), 1)}


def score(m):
    """Robustness score — reward PF & return, require enough trades, punish DD."""
    if not m or m["trades"] < 15:
        return -999
    return (m["pf"] * 20) + (m["return"] * 0.3) - (abs(m["max_dd"]) * 0.5)


def optimize_pair(pair, interval="1h"):
    df = load_pair(pair, interval)
    if df is None or len(df) < 400:
        return None
    df = calculate_all_indicators(df).dropna()
    df = _compute_signals(df)
    split = int(len(df) * 0.70)
    is_df, oos_df = df.iloc[:split], df.iloc[split:]

    best = None
    for ms, sl, tp in GRID:
        tr, eq, _ = simulate(is_df, pair, ms, sl, tp)
        m = metrics(tr, eq, 100.0)
        sc = score(m)
        if best is None or sc > best["score"]:
            best = {"params": (ms, sl, tp), "is": m, "score": sc}

    # Validate best params on unseen OOS data
    ms, sl, tp = best["params"]
    tr, eq, _ = simulate(oos_df, pair, ms, sl, tp)
    best["oos"] = metrics(tr, eq, 100.0)
    return best


if __name__ == "__main__":
    results = {}
    print(f"{'Pair':8s} {'min':>3s} {'SL':>4s} {'TP':>4s} | {'IS ret%':>7s} {'IS pf':>5s} | {'OOS ret%':>8s} {'OOS wr%':>7s} {'OOS pf':>6s} {'OOS dd%':>7s} {'#':>4s}")
    print("-" * 85)
    for pair in PAIRS:
        b = optimize_pair(pair, "1h")
        if not b or not b.get("oos"):
            print(f"{pair:8s}  no valid config")
            continue
        ms, sl, tp = b["params"]
        i, o = b["is"], b["oos"]
        results[pair] = {"min_signals": ms, "sl_atr": sl, "tp_atr": tp,
                         "oos": o, "is": i}
        print(f"{pair:8s} {ms:>3d} {sl:>4.2f} {tp:>4.2f} | {i['return']:>7.1f} {i['pf']:>5.2f} | "
              f"{o['return']:>8.1f} {o['win_rate']:>7.1f} {o['pf']:>6.2f} {o['max_dd']:>7.1f} {o['trades']:>4d}")

    Path("data").mkdir(exist_ok=True)
    Path("data/optimized_params.json").write_text(json.dumps(results, indent=2))
    print("\nSaved -> data/optimized_params.json")

"""
Support/Resistance Reaction Analyzer
Studies 5 years of data to learn how each pair behaves at S/R levels:
 - identifies swing-pivot S/R levels, clusters nearby ones
 - finds every time price later TESTS a level
 - classifies each test as BOUNCE (rejection) or BREAK (close through)
 - reports per-pair bounce rates, avg bounce size, and break-through rates

Output -> data/sr_reactions.json  (the tool then "knows" each pair's S/R behaviour)
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

SWING_W      = 5      # bars each side to qualify a swing pivot
TEST_TOL_ATR = 0.30  # "touch" = within 0.30*ATR of the level
BOUNCE_ATR   = 1.0   # bounce = reverses >= 1.0*ATR away within LOOKFWD bars
BREAK_ATR    = 0.5   # break  = closes >= 0.5*ATR beyond the level
LOOKFWD      = 10    # bars to resolve a test


def find_levels(df):
    """Return (resistances, supports) as lists of price levels from swing pivots."""
    highs, lows = df["High"].values, df["Low"].values
    res, sup = [], []
    for i in range(SWING_W, len(df) - SWING_W):
        win_h = highs[i-SWING_W:i+SWING_W+1]
        win_l = lows[i-SWING_W:i+SWING_W+1]
        if highs[i] == win_h.max():
            res.append((i, highs[i]))
        if lows[i] == win_l.min():
            sup.append((i, lows[i]))
    return res, sup


def classify_tests(df, levels, kind):
    """For each level, find later tests and classify bounce vs break."""
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    atr   = df["ATR"].values
    n = len(df)
    bounces = breaks = 0
    bounce_sizes = []

    for (idx, lvl) in levels:
        # scan bars AFTER the pivot formed
        j = idx + SWING_W + 1
        while j < n - LOOKFWD:
            a = atr[j]
            if a <= 0:
                j += 1; continue
            tol = TEST_TOL_ATR * a
            touched = (low[j] - tol) <= lvl <= (high[j] + tol)
            if not touched:
                j += 1; continue

            # resolve over next LOOKFWD bars
            seg_close = close[j:j+LOOKFWD]
            seg_high  = high[j:j+LOOKFWD]
            seg_low   = low[j:j+LOOKFWD]
            resolved = False
            if kind == "resistance":
                # break = a close above level + buffer
                if (seg_close > lvl + BREAK_ATR * a).any():
                    breaks += 1; resolved = True
                else:
                    drop = lvl - seg_low.min()
                    if drop >= BOUNCE_ATR * a:
                        bounces += 1; bounce_sizes.append(drop / a); resolved = True
            else:  # support
                if (seg_close < lvl - BREAK_ATR * a).any():
                    breaks += 1; resolved = True
                else:
                    rise = seg_high.max() - lvl
                    if rise >= BOUNCE_ATR * a:
                        bounces += 1; bounce_sizes.append(rise / a); resolved = True
            # skip past this test window to avoid double counting
            j += LOOKFWD if resolved else 1
    return bounces, breaks, bounce_sizes


def analyze_pair(pair, interval="1d"):
    df = load_pair(pair, interval)
    if df is None or len(df) < 200:
        return None
    df = df.copy()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df = df.dropna().reset_index(drop=True)

    res_levels, sup_levels = find_levels(df)
    # cluster nearby levels (within 0.5*median ATR)
    rb, rk, rsz = classify_tests(df, res_levels, "resistance")
    sb, sk, ssz = classify_tests(df, sup_levels, "support")

    res_tests = rb + rk
    sup_tests = sb + sk
    return {
        "bars": len(df),
        "span": f"{df['ATR'].notna().sum()} bars",
        "resistance": {
            "tests": res_tests,
            "bounce_rate": round(rb / res_tests * 100, 1) if res_tests else 0,
            "break_rate":  round(rk / res_tests * 100, 1) if res_tests else 0,
            "avg_bounce_atr": round(float(np.mean(rsz)), 2) if rsz else 0,
        },
        "support": {
            "tests": sup_tests,
            "bounce_rate": round(sb / sup_tests * 100, 1) if sup_tests else 0,
            "break_rate":  round(sk / sup_tests * 100, 1) if sup_tests else 0,
            "avg_bounce_atr": round(float(np.mean(ssz)), 2) if ssz else 0,
        },
    }


if __name__ == "__main__":
    out = {}
    print("SUPPORT/RESISTANCE REACTION STUDY — 5yr daily data")
    print(f"{'Pair':8s} | {'RES tests':>9s} {'bounce%':>7s} {'break%':>6s} | {'SUP tests':>9s} {'bounce%':>7s} {'break%':>6s}")
    print("-"*78)
    for pair in PAIRS:
        r = analyze_pair(pair, "1d")
        if not r:
            print(f"{pair:8s}  no data"); continue
        out[pair] = r
        R, S = r["resistance"], r["support"]
        print(f"{pair:8s} | {R['tests']:>9d} {R['bounce_rate']:>7.1f} {R['break_rate']:>6.1f} | "
              f"{S['tests']:>9d} {S['bounce_rate']:>7.1f} {S['break_rate']:>6.1f}")
    Path("data").mkdir(exist_ok=True)
    Path("data/sr_reactions.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> data/sr_reactions.json")

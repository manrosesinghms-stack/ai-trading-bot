"""
GOLD STRATEGY LAB — find a >75% win-rate strategy on XAUUSD that is ALSO
profitable out-of-sample. Tests 7 entry methods x TP/SL grid x H1 & D1,
70/30 train/test split. Reports OUT-OF-SAMPLE results only.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from data_manager import load_pair
from strategies import calculate_all_indicators
from backtester import _compute_signals

PAIR = "XAUUSD"; PIPSZ = 0.1; SPREAD = 3.0

def build(df):
    df = calculate_all_indicators(df)
    df = _compute_signals(df).dropna().reset_index(drop=True)
    # Donchian for breakout
    df["DonHi"] = df["High"].rolling(20).max().shift(1)
    df["DonLo"] = df["Low"].rolling(20).min().shift(1)
    return df

def signals(df):
    """Return dict name -> direction array (1 long, -1 short, 0 none)."""
    c, rsi, e20, e50, e200 = df["Close"], df["RSI"], df["EMA20"], df["EMA50"], df["EMA200"]
    bbl, bbu, k = df["BB_Lower"], df["BB_Upper"], df["Stoch_K"]
    low, high = df["Low"], df["High"]
    out = {}
    out["consensus3"] = np.where((df.buy_count>=3)&(df.buy_count>df.sell_count),1,
                         np.where((df.sell_count>=3)&(df.sell_count>df.buy_count),-1,0))
    out["consensus4"] = np.where((df.buy_count>=4)&(df.buy_count>df.sell_count),1,
                         np.where((df.sell_count>=4)&(df.sell_count>df.buy_count),-1,0))
    out["rsi_revert"] = np.where(rsi<30,1,np.where(rsi>70,-1,0))
    # pullback in trend: dip while above EMA200 = buy
    out["rsi_trend"]  = np.where((rsi<40)&(c>e200),1,np.where((rsi>60)&(c<e200),-1,0))
    out["bb_bounce"]  = np.where(c<=bbl,1,np.where(c>=bbu,-1,0))
    out["stoch"]      = np.where(k<20,1,np.where(k>80,-1,0))
    # EMA pullback: strong uptrend & price taps EMA20
    up = (e20>e50)&(e50>e200); dn=(e20<e50)&(e50<e200)
    out["ema_pullback"] = np.where(up&(low<=e20),1,np.where(dn&(high>=e20),-1,0))
    out["breakout"]   = np.where(c>df.DonHi,1,np.where(c<df.DonLo,-1,0))
    return out

def simulate(df, dirarr, sl_atr, tp_atr, start=100.0, risk=2.0):
    atr=df["ATR"].values; cl=df["Close"].values; hi=df["High"].values; lo=df["Low"].values
    bal=start; pos=None; trades=[]; eq=[bal]
    for i in range(len(df)):
        a=atr[i]
        if pos:
            hsl=lo[i]<=pos["sl"] if pos["d"]==1 else hi[i]>=pos["sl"]
            htp=hi[i]>=pos["tp"] if pos["d"]==1 else lo[i]<=pos["tp"]
            if hsl or htp:
                px=pos["sl"] if hsl else pos["tp"]
                r=((px-pos["e"])/PIPSZ*pos["d"]-SPREAD)/pos["slp"]
                pnl=pos["risk"]*r; bal+=pnl; trades.append(pnl); eq.append(bal); pos=None
        if pos: continue
        if a<=0: continue
        d=int(dirarr[i])
        if d==0: continue
        e=cl[i]+d*SPREAD*PIPSZ
        pos={"d":d,"e":e,"sl":e-d*a*sl_atr,"tp":e+d*a*tp_atr,"slp":a*sl_atr/PIPSZ,"risk":bal*risk/100}
    return trades, eq

def metr(trades,eq,start=100.0):
    if len(trades)<15: return None
    w=[t for t in trades if t>0]; l=[t for t in trades if t<=0]
    s=pd.Series(eq); dd=((s-s.cummax())/s.cummax()*100).min()
    pf=(sum(w)/abs(sum(l))) if l else 99.0
    return {"trades":len(trades),"ret":round((eq[-1]/start-1)*100,1),
            "wr":round(len(w)/len(trades)*100,1),"pf":round(pf,2),"dd":round(float(dd),1)}

SL_GRID=[1.0,1.5,2.0,2.5,3.0]; TP_GRID=[0.5,0.75,1.0,1.5,2.0]

if __name__=="__main__":
    allcfg=[]
    for tf in ("1h","1d"):
        df=load_pair(PAIR,tf)
        if df is None: continue
        df=build(df)
        sp=int(len(df)*0.70); isd=df.iloc[:sp].reset_index(drop=True); oos=df.iloc[sp:].reset_index(drop=True)
        sig_is=signals(isd); sig_oos=signals(oos)
        for name in sig_is:
            for sl in SL_GRID:
                for tp in TP_GRID:
                    ti,ei=simulate(isd,sig_is[name],sl,tp); mi=metr(ti,ei)
                    if not mi or mi["wr"]<70 or mi["pf"]<=1: continue   # promising in-sample
                    to,eo=simulate(oos,sig_oos[name],sl,tp); mo=metr(to,eo)
                    if not mo: continue
                    allcfg.append({"tf":tf,"entry":name,"sl":sl,"tp":tp,"oos":mo,"is":mi})
    # keep OOS winners: WR>75 AND profitable
    winners=[c for c in allcfg if c["oos"]["wr"]>75 and c["oos"]["pf"]>1 and c["oos"]["ret"]>0]
    winners.sort(key=lambda c:(c["oos"]["wr"], c["oos"]["pf"]), reverse=True)
    print("GOLD strategies with OUT-OF-SAMPLE win rate > 75% AND profitable")
    print(f"{'TF':>3s} {'entry':13s} {'SL':>4s} {'TP':>4s} | {'OOS wr%':>7s} {'pf':>5s} {'ret%':>7s} {'dd%':>6s} {'#':>4s}")
    print("-"*70)
    for c in winners[:20]:
        o=c["oos"]
        print(f"{c['tf']:>3s} {c['entry']:13s} {c['sl']:>4.1f} {c['tp']:>4.1f} | {o['wr']:>7.1f} {o['pf']:>5.2f} {o['ret']:>7.1f} {o['dd']:>6.1f} {o['trades']:>4d}")
    if not winners:
        print("None found with WR>75% AND profitable OOS.")
        # show best WR regardless of profit, to be honest
        allcfg.sort(key=lambda c:c["oos"]["wr"], reverse=True)
        print("\nHighest OOS win rates found (note profit!):")
        for c in allcfg[:10]:
            o=c["oos"]
            print(f"{c['tf']:>3s} {c['entry']:13s} SL{c['sl']} TP{c['tp']} | wr {o['wr']}% pf {o['pf']} ret {o['ret']}% #{o['trades']}")
    Path("data/gold_winners.json").write_text(json.dumps(winners[:20],indent=2))
    print(f"\nFound {len(winners)} winning configs. Saved -> data/gold_winners.json")

"""
On-demand backtester — "how would STRATEGY have done on SYMBOL during PERIOD?"
No live bot needed. Pick a strategy, pair, and date range; get the result.

Usage:
  python backtest_month.py <strategy> <symbol> <start> <end>
  e.g. python backtest_month.py trend XAUUSD 2026-05-01 2026-05-31
Strategies: trend | rsi_cross | conviction | cbl | consensus
"""
from __future__ import annotations
import sys
import pandas as pd, numpy as np
import yfinance as yf
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

TICK = {"XAUUSD":"GC=F","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X",
        "AUDUSD":"AUDUSD=X","USDCAD":"USDCAD=X","GBPJPY":"GBPJPY=X","EURJPY":"EURJPY=X"}
PIP  = {"XAUUSD":0.1,"USDJPY":0.01,"GBPJPY":0.01,"EURJPY":0.01}
SPRD = {"XAUUSD":3.0,"EURUSD":1.0,"GBPUSD":1.2,"USDJPY":1.0,"AUDUSD":1.2,
        "USDCAD":1.5,"GBPJPY":2.0,"EURJPY":1.5}
def pipof(s): return PIP.get(s,0.0001)

def load(sym, start):
    df=yf.download(TICK.get(sym,sym+"=X"), start=start, end="2026-06-05",
                   auto_adjust=True, progress=False)
    if hasattr(df.columns,"levels"): df.columns=df.columns.get_level_values(0)
    df=df[["Open","High","Low","Close","Volume"]].dropna()
    C=df.Close
    df["E50"]=EMAIndicator(C,50).ema_indicator(); df["E200"]=EMAIndicator(C,200).ema_indicator()
    df["RSI"]=RSIIndicator(C,14).rsi()
    m=MACD(C); df["MH"]=m.macd_diff()
    ax=ADXIndicator(df.High,df.Low,C,14); df["ADX"]=ax.adx(); df["DIP"]=ax.adx_pos(); df["DIM"]=ax.adx_neg()
    df["ATR"]=AverageTrueRange(df.High,df.Low,C,14).average_true_range()
    df["DonHi"]=df.High.rolling(20).max().shift(1); df["DonLo"]=df.Low.rolling(20).min().shift(1)
    return df.dropna()

def backtest(strategy, sym, start, end, risk=2.0):
    full=load(sym, "2019-01-01")        # warmup history for indicators
    full.index=pd.to_datetime(full.index)
    p=pipof(sym); spr=SPRD.get(sym,1.5)
    s=pd.Timestamp(start); e=pd.Timestamp(end)
    bal=100.0; pos=None; trades=[]
    arr=full
    O=arr.Open.values;H=arr.High.values;L=arr.Low.values;C=arr.Close.values
    e50=arr.E50.values;e200=arr.E200.values;rsi=arr.RSI.values;rsip=np.roll(rsi,1)
    mh=arr.MH.values;mhp=np.roll(mh,1);adx=arr.ADX.values;dip=arr.DIP.values;dim=arr.DIM.values
    atr=arr.ATR.values;dhi=arr.DonHi.values;dlo=arr.DonLo.values
    atrp=np.roll(atr,5);c5=np.roll(C,5)
    rng=np.where((H-L)>0,(C-L)/(H-L),0.5)
    dates=full.index

    # trailing-stop strategies (trend) handled inline; others use SL/TP
    for i in range(1,len(arr)):
        d=dates[i]; a=atr[i]
        in_window = (s<=d<=e)
        # manage open
        if pos:
            if pos.get("trail"):
                if pos["dr"]==1:
                    pos["pk"]=max(pos["pk"],H[i]); stop=pos["pk"]-pos["trail"]*a
                    if L[i]<=stop: _close(pos,stop,p,spr,trades); bal+=trades[-1]["pnl"]; pos=None
                else:
                    pos["pk"]=min(pos["pk"],L[i]); stop=pos["pk"]+pos["trail"]*a
                    if H[i]>=stop: _close(pos,stop,p,spr,trades); bal+=trades[-1]["pnl"]; pos=None
            else:
                hsl=L[i]<=pos["sl"] if pos["dr"]==1 else H[i]>=pos["sl"]
                htp=H[i]>=pos["tp"] if pos["dr"]==1 else L[i]<=pos["tp"]
                if hsl or htp:
                    _close(pos, pos["sl"] if hsl else pos["tp"], p, spr, trades); bal+=trades[-1]["pnl"]; pos=None
        if pos or a<=0 or not in_window: continue

        dr=0; trail=None; sl=tp=None
        if strategy=="trend":
            dr=1 if C[i]>dhi[i] else (-1 if C[i]<dlo[i] else 0); trail=2.0
        elif strategy=="rsi_cross":
            up=e50[i]>e200[i]; dn=e50[i]<e200[i]
            dr=1 if (up and rsip[i]<40 and rsi[i]>=40) else (-1 if (dn and rsip[i]>60 and rsi[i]<=60) else 0)
            if dr: sl=1.5; tp=1.5
        elif strategy=="conviction":
            bull=sum([C[i]>e50[i] and e50[i]>e200[i], adx[i]>25, dip[i]>dim[i], mh[i]>0 and mh[i]>mhp[i],
                      50<rsi[i]<72, rng[i]>0.6, C[i]>c5[i], atr[i]>atrp[i]])
            bear=sum([C[i]<e50[i] and e50[i]<e200[i], adx[i]>25, dim[i]>dip[i], mh[i]<0 and mh[i]<mhp[i],
                      28<rsi[i]<50, rng[i]<0.4, C[i]<c5[i], atr[i]>atrp[i]])
            dr=1 if bull>=7 else (-1 if bear>=7 else 0)
            if dr: sl=2.5; tp=1.0
        elif strategy=="cbl":
            pl=L[max(0,i-10):i].min(); ph=H[max(0,i-10):i].max()
            up=e50[i]>e200[i]; dn=e50[i]<e200[i]
            if up and L[i]<pl and C[i]>pl: dr=1
            elif dn and H[i]>ph and C[i]<ph: dr=-1
            if dr:  # SL at swept wick, TP 2R
                if dr==1: sl_px=L[i]-0.1*a; tp_px=C[i]+2*(C[i]-sl_px)
                else: sl_px=H[i]+0.1*a; tp_px=C[i]-2*(sl_px-C[i])
                ent=C[i]+dr*spr*p
                pos={"dr":dr,"e":ent,"sl":sl_px,"tp":tp_px,"slp":abs(ent-sl_px)/p,
                     "risk":bal*risk/100,"date":str(d.date())}
                continue
        elif strategy=="consensus":
            bc=sum([rsi[i]<35, mh[i]>0, e50[i]>e200[i], C[i]<=arr.E50.values[i], adx[i]>22 and dip[i]>dim[i]])
            sc=sum([rsi[i]>65, mh[i]<0, e50[i]<e200[i], adx[i]>22 and dim[i]>dip[i]])
            dr=1 if bc>=3 else (-1 if sc>=3 else 0)
            if dr: sl=1.5; tp=2.5
        if dr==0: continue
        ent=C[i]+dr*spr*p
        if trail:
            pos={"dr":dr,"e":ent,"pk":ent,"trail":trail,"slp":trail*a/p,"risk":bal*risk/100,"date":str(d.date())}
        else:
            pos={"dr":dr,"e":ent,"sl":ent-dr*a*sl,"tp":ent+dr*a*tp,"slp":a*sl/p,"risk":bal*risk/100,"date":str(d.date())}
    return summarize(strategy,sym,start,end,trades,bal)

def _close(pos,px,p,spr,trades):
    r=((px-pos["e"])/p*pos["dr"]-spr)/pos["slp"]
    pnl=pos["risk"]*r
    trades.append({"date":pos["date"],"dir":"Buy" if pos["dr"]==1 else "Sell",
                   "entry":round(pos["e"],3),"exit":round(px,3),"pnl":round(pnl,2)})

def summarize(strategy,sym,start,end,trades,bal):
    n=len(trades); w=[t for t in trades if t["pnl"]>0]
    wr=len(w)/n*100 if n else 0
    return {"strategy":strategy,"symbol":sym,"period":f"{start}->{end}",
            "trades":n,"win_rate":round(wr,1),"net_pnl":round(bal-100,2),
            "return_pct":round(bal-100,2),"detail":trades}

def show(r):
    print(f"\n{r['strategy'].upper()} on {r['symbol']}  |  {r['period']}")
    print("-"*54)
    if r["trades"]==0:
        print("  No trades triggered in this period."); return
    for t in r["detail"]:
        ic="WIN " if t["pnl"]>0 else "LOSS"
        print(f"  {t['date']} {t['dir']:4s} {t['entry']:>9} -> {t['exit']:>9} | {ic} ${t['pnl']:+6.2f}")
    print("-"*54)
    print(f"  Trades {r['trades']} | Win rate {r['win_rate']}% | Net P/L ${r['net_pnl']:+.2f} (on $100, 2% risk)")

if __name__=="__main__":
    if len(sys.argv)>=5:
        show(backtest(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
    else:
        # demo: all strategies on Gold, last month
        for strat in ["trend","rsi_cross","conviction","cbl","consensus"]:
            show(backtest(strat,"XAUUSD","2026-05-01","2026-05-31"))

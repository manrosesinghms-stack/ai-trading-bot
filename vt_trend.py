"""
Volatility-Targeted Trend-Follower (VTT)
The honest best-of-everything: the ONE robust signal (trend) + the ONE
academically-validated improvement (volatility targeting for position sizing).

Academic basis: Moskowitz/Ooi/Pedersen (TSMOM, 2012) + Hurst/Ooi/Pedersen (AQR),
with the Kim/Tse/Wald (2016) correction that vol-targeting is the durable edge.

Signal   : sign of trend (price vs N-day ago) — long in uptrend, short in downtrend
Sizing   : exposure = target_vol / realized_vol  (size DOWN when wild, UP when calm)
Rebalance: daily, compounded
"""
from __future__ import annotations
import numpy as np, pandas as pd

def vt_signal(close: pd.Series, lookback=100, target_vol=0.15, vol_window=20,
              max_lev=2.0, long_short=True):
    """Return daily target exposure series for a price series."""
    ret = close.pct_change()
    mom = close / close.shift(lookback) - 1.0
    sign = np.sign(mom)
    if not long_short:
        sign = sign.clip(lower=0)          # long-or-flat only
    realized = ret.rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / realized).clip(upper=max_lev).fillna(0)
    return (sign * scale).shift(1).fillna(0)   # shift(1): trade next bar, no lookahead

def backtest(close: pd.Series, **kw):
    ret = close.pct_change().fillna(0)
    exp = vt_signal(close, **kw)
    strat_ret = exp * ret
    eq = (1 + strat_ret).cumprod()
    yrs = len(eq) / 252
    cagr = eq.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else 0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    return {"total_pct": (eq.iloc[-1]-1)*100, "cagr_pct": cagr*100,
            "max_dd_pct": dd*100, "sharpe": sharpe, "eq": eq, "exposure": exp}

def current_call(close: pd.Series, lookback=100, target_vol=0.15, vol_window=20, max_lev=2.0):
    """What the model says to do RIGHT NOW for a price series."""
    exp = vt_signal(close, lookback, target_vol, vol_window, max_lev)
    e = float(exp.iloc[-1])
    mom = float(close.iloc[-1]/close.iloc[-lookback]-1)*100 if len(close)>lookback else 0
    direction = "LONG" if e > 0.05 else ("SHORT" if e < -0.05 else "FLAT/CASH")
    return {"direction": direction, "exposure": round(e,2),
            "trend_pct": round(mom,1), "leverage": round(abs(e),2)}

if __name__ == "__main__":
    import yfinance as yf
    def load(tkr, start="2004-01-01"):
        df = yf.download(tkr, start=start, end="2026-06-01", auto_adjust=True, progress=False)
        if hasattr(df.columns,'levels'): df.columns=df.columns.get_level_values(0)
        return df["Close"].dropna()
    print("VOLATILITY-TARGETED TREND-FOLLOWER — validation\n")
    for name,tkr in [("GOLD","GC=F"),("EURUSD","EURUSD=X"),("GBPUSD","GBPUSD=X"),("USDJPY","USDJPY=X")]:
        c = load(tkr)
        ret=c.pct_change().fillna(0); bh=(1+ret).cumprod()
        bh_cagr=(bh.iloc[-1]**(252/len(bh))-1)*100
        bh_dd=((bh-bh.cummax())/bh.cummax()).min()*100
        bh_sh=ret.mean()/ret.std()*np.sqrt(252)
        r=backtest(c)
        print(f"=== {name} (2004-2026) ===")
        print(f"  Buy & Hold      : CAGR {bh_cagr:>+5.1f}% | maxDD {bh_dd:>6.1f}% | Sharpe {bh_sh:.2f}")
        print(f"  VT Trend-Follow : CAGR {r['cagr_pct']:>+5.1f}% | maxDD {r['max_dd_pct']:>6.1f}% | Sharpe {r['sharpe']:.2f} | total {r['total_pct']:>+.0f}%")
        # per-era robustness
        for s,e in [('2004','2011'),('2011','2018'),('2018','2026')]:
            sub=c[(c.index>=s+'-01-01')&(c.index<e+'-01-01')]
            if len(sub)>250:
                rr=backtest(sub)
                print(f"    {s}-{e}: CAGR {rr['cagr_pct']:>+5.1f}% | maxDD {rr['max_dd_pct']:>6.1f}% | Sharpe {rr['sharpe']:.2f}")
        print()

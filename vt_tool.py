"""
Volatility-Targeted Trend Portfolio — Dashboard Tool
The honest, research-validated strategy: diversified trend-following with
volatility-targeted sizing across asset classes.

Run:  python -m streamlit run vt_tool.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from vt_trend import vt_signal, current_call

st.set_page_config(page_title="VT Trend Portfolio", page_icon="📈", layout="wide")
st.markdown("<style>[data-testid='stAppViewContainer']{background:#07071a}</style>", unsafe_allow_html=True)

UNIVERSE = {"Gold":"GC=F","Silver":"SI=F","Copper":"HG=F","Crude Oil":"CL=F","NatGas":"NG=F",
            "S&P 500":"ES=F","Nasdaq":"NQ=F","10yr Bond":"ZN=F","30yr Bond":"ZB=F",
            "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X","AUD/USD":"AUDUSD=X"}

st.title("📈 Volatility-Targeted Trend Portfolio")
st.caption("The research-validated strategy: diversified trend-following + volatility-targeted sizing. "
           "Smooth & robust, not a jackpot — honest by design.")

c1,c2,c3,c4 = st.columns(4)
lookback   = c1.slider("Trend lookback (days)", 50, 200, 100, 10)
target_vol = c2.slider("Target volatility %", 5, 30, 15) / 100
vol_window = c3.slider("Vol estimate window", 10, 60, 20, 5)
max_lev    = c4.slider("Max leverage", 1.0, 3.0, 2.0, 0.5)

@st.cache_data(ttl=1800)
def load_universe(_t=0):
    import yfinance as yf
    closes={}
    for name,tkr in UNIVERSE.items():
        try:
            d=yf.download(tkr,start="2010-01-01",end="2026-12-31",auto_adjust=True,progress=False)
            if hasattr(d.columns,"levels"): d.columns=d.columns.get_level_values(0)
            s=d["Close"].dropna()
            if len(s)>800: closes[name]=s
        except Exception: pass
    return closes

with st.spinner("Loading market data..."):
    closes = load_universe()

# ── Current signals ──
st.subheader("📍 Current Positions the Model Says to Hold")
rows=[]
for name,c in closes.items():
    call=current_call(c, lookback, target_vol, vol_window, max_lev)
    rows.append({"Market":name, "Trend %":call["trend_pct"], "Signal":call["direction"],
                 "Exposure":call["exposure"], "Size (x)":call["leverage"]})
dfc=pd.DataFrame(rows)
def color(v):
    if v=="LONG": return "background:#0d2b16;color:#00FF88;font-weight:700"
    if v=="SHORT": return "background:#2b0d0d;color:#FF4444;font-weight:700"
    return "color:#888"
st.dataframe(dfc.style.applymap(color, subset=["Signal"]), use_container_width=True, hide_index=True)
longs=sum(1 for r in rows if r["Signal"]=="LONG"); shorts=sum(1 for r in rows if r["Signal"]=="SHORT")
st.caption(f"Net stance: {longs} long / {shorts} short / {len(rows)-longs-shorts} flat across {len(rows)} markets")

# ── Portfolio backtest ──
st.subheader("📊 Portfolio Backtest (equal-risk blend)")
px=pd.DataFrame(closes).dropna()
S={}
for n in px.columns:
    ret=px[n].pct_change().fillna(0)
    S[n]=vt_signal(px[n],lookback,target_vol,vol_window,max_lev)*ret
port=pd.DataFrame(S).dropna().mean(axis=1)
eq=(1+port).cumprod()
yrs=len(eq)/252
cagr=(eq.iloc[-1]**(1/yrs)-1)*100
dd=((eq-eq.cummax())/eq.cummax()).min()*100
sh=port.mean()/port.std()*np.sqrt(252)

m1,m2,m3,m4=st.columns(4)
m1.metric("Total Return", f"{(eq.iloc[-1]-1)*100:+.0f}%")
m2.metric("CAGR", f"{cagr:+.1f}%")
m3.metric("Max Drawdown", f"{dd:.1f}%")
m4.metric("Sharpe", f"{sh:.2f}")

fig=go.Figure(go.Scatter(x=eq.index,y=(eq-1)*100,line=dict(color="#00D4FF",width=2),
                         fill="tozeroy",fillcolor="rgba(0,212,255,0.06)"))
fig.update_layout(height=320,template="plotly_dark",paper_bgcolor="#07071a",plot_bgcolor="#07071a",
                  margin=dict(t=10,b=20,l=0,r=0),yaxis_title="Cumulative return %")
st.plotly_chart(fig,use_container_width=True)

# ── Live $500 paper account (forward from start date) ──
st.subheader("💵 Live Paper Account — $500, tracking forward")
import json
from pathlib import Path
from datetime import datetime, timezone
cfgp = json.loads(Path("data/vt_paper.json").read_text()) if Path("data/vt_paper.json").exists() else {"start_date":"2026-06-05","start_balance":500.0}
start_date = pd.Timestamp(cfgp["start_date"]); start_bal = cfgp["start_balance"]

# portfolio returns from start_date forward (recomputed live from data = self-correcting)
fwd = port[port.index >= start_date]
if len(fwd) < 2:
    st.success(f"▶ Account started at **${start_bal:.2f}** on **{cfgp['start_date']}**. "
               f"Come back in a few days — forward results will appear here as new market data arrives.")
    cur_bal = start_bal
else:
    eqf = (1+fwd).cumprod(); cur_bal = start_bal*eqf.iloc[-1]
    days = (fwd.index[-1]-start_date).days
    ddf = ((eqf-eqf.cummax())/eqf.cummax()).min()*100
    p1,p2,p3,p4 = st.columns(4)
    chg = cur_bal-start_bal
    p1.metric("Starting", f"${start_bal:.2f}")
    p2.metric("Current Value", f"${cur_bal:.2f}", delta=f"{chg:+.2f}")
    p3.metric("Return", f"{(cur_bal/start_bal-1)*100:+.2f}%")
    p4.metric("Days running", days)
    figp=go.Figure(go.Scatter(x=eqf.index,y=start_bal*eqf,line=dict(color="#00FF88",width=2),
                              fill="tozeroy",fillcolor="rgba(0,255,136,0.06)"))
    figp.add_hline(y=start_bal,line=dict(color="#555",dash="dash"))
    figp.update_layout(height=260,template="plotly_dark",paper_bgcolor="#07071a",plot_bgcolor="#07071a",
                       margin=dict(t=10,b=20,l=0,r=0),yaxis_title="Account value $")
    st.plotly_chart(figp,use_container_width=True)

st.info("**How to use this honestly:** this is a *position/allocation* model. The $500 account above tracks "
        "the strategy's REAL forward performance from your start date — the true out-of-sample test. "
        "Rebalance ~weekly to the positions shown. Expect steady, low-drawdown growth — modest, not riches. "
        "Come back over weeks/months; that forward curve is the only result that truly matters.")

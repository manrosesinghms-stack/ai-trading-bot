"""
Dashboard — displays the headless engine's results (data/bot_state.json).
The actual trading runs on GitHub Actions cron every 15 min, so this page
just shows live progress. Works even when your PC is off.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Trading Bot Dashboard", page_icon="🤖",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#07071a}
.stat{background:#0d0d22;border:1px solid #1a1a3a;border-radius:12px;padding:14px;text-align:center}
.lbl{color:#555;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase}
.val{font-size:1.5rem;font-weight:800}
@keyframes p{0%{opacity:1}50%{opacity:.4}100%{opacity:1}}
.live{display:inline-block;width:10px;height:10px;border-radius:50%;background:#00FF88;animation:p 1.2s infinite;margin-right:6px}
</style>""", unsafe_allow_html=True)

DATA = Path(__file__).parent / "data"
STATE_FILE  = DATA / "bot_state.json"
CONFIG_FILE = DATA / "bot_config.json"

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="ref")
except ImportError:
    pass

RAW_BASE = "https://raw.githubusercontent.com/manrosesinghms-stack/ai-trading-bot/main/data"

def load(p, d):
    # Prefer the live file from GitHub (updated by the cron every 15 min),
    # fall back to the local copy bundled at deploy time.
    name = p.name
    try:
        import requests
        r = requests.get(f"{RAW_BASE}/{name}?t={int(datetime.now(timezone.utc).timestamp())}", timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: pass
    return d

state = load(STATE_FILE, {"balance":100,"start_balance":100,"positions":[],
             "trades":[],"scan_count":0,"started_at":None,"last_scan":None})
cfg = load(CONFIG_FILE, {"running":True,"symbols":["XAUUSD","USDCAD","EURJPY","AUDUSD"]})

def fmt(p, s):
    d = 3 if "JPY" in s else (2 if "XAU" in s else 5)
    return f"{p:.{d}f}"

PIP = {"EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
       "USDCHF":.0001,"USDJPY":.01,"EURJPY":.01,"GBPJPY":.01,"XAUUSD":.1}

@st.cache_data(ttl=55)
def live_price(sym, _t=0):
    try:
        import yfinance as yf
        from data_manager import PAIRS
        df = yf.download(PAIRS.get(sym, sym+"=X"), period="1d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty: return None
        if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

# Recompute LIVE unrealized P&L for each open position (not the 15-min snapshot)
_tick = int(datetime.now(timezone.utc).timestamp() // 55)
for _p in state.get("positions", []):
    px = live_price(_p["symbol"], _tick)
    if px:
        pp = PIP.get(_p["symbol"], .0001)
        slp = abs(_p["entry"] - _p["sl"]) / pp
        d = 1 if _p["direction"] == "Buy" else -1
        pnl_pips = (px - _p["entry"]) / pp * d - 1.5
        _p["current_price"] = px
        _p["pnl"] = round(_p["risk_usd"] * (pnl_pips / max(slp, 1)), 4)

# ── Header ──
running = cfg.get("running", True)
last = state.get("last_scan", "never")
# staleness check
stale = ""
if last and last != "never":
    try:
        lt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        mins = (datetime.now(timezone.utc) - lt).total_seconds()/60
        stale = f"{int(mins)} min ago" + ("  ⚠️ engine may be paused" if mins > 40 else "  ✅")
    except Exception:
        stale = last

c1, c2 = st.columns([3,2])
with c1:
    dot = '<span class="live"></span>' if running else "⏸ "
    st.markdown(f"## 🤖 Trading Bot {dot}<span style='color:#00FF88'>{'LIVE 24/7' if running else 'PAUSED'}</span>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='text-align:right;color:#666;padding-top:14px'>Last scan: <b style='color:#aaa'>{stale}</b><br>Scan #{state['scan_count']} · runs on GitHub every 15 min</div>", unsafe_allow_html=True)

# ── Controls ──
b1, b2, b3 = st.columns([1,1,4])
with b1:
    if st.button("⏸ Pause" if running else "▶ Resume", use_container_width=True):
        cfg["running"] = not running
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        st.rerun()
with b2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# ── Stats ──
bal = state["balance"]; sb = state["start_balance"]
ret = (bal/sb-1)*100
open_pnl = sum(p.get("pnl",0) for p in state["positions"])
eq = bal + open_pnl
trades = state["trades"]
wins = [t for t in trades if t.get("pnl",0) > 0]
wr = len(wins)/len(trades)*100 if trades else 0
closed_pnl = sum(t.get("pnl",0) for t in trades)
best = max((t.get("pnl",0) for t in trades), default=0)
worst = min((t.get("pnl",0) for t in trades), default=0)
bals = [sb] + [t.get("balance_after",sb) for t in reversed(trades)]
maxb = max(bals); mdd = (maxb - min(bals))/maxb*100 if maxb else 0

def col(v): return "#00FF88" if v>=0 else "#FF4444"
stats = [("Starting",f"${sb:.2f}","#eee"),("Balance",f"${bal:.2f}",col(ret)),
         ("Equity",f"${eq:.2f}",col(open_pnl)),("Return",f"{ret:+.2f}%",col(ret)),
         ("Closed P/L",f"${closed_pnl:+.2f}",col(closed_pnl)),("Open P/L",f"${open_pnl:+.2f}",col(open_pnl)),
         ("Win Rate",f"{wr:.0f}%","#00FF88" if wr>=50 else "#FFAA00"),("Trades",str(len(trades)),"#eee"),
         ("Best",f"${best:+.2f}","#00FF88"),("Worst",f"${worst:+.2f}","#FF4444"),
         ("Max DD",f"{mdd:.1f}%","#FFAA00"),("Open",str(len(state['positions'])),"#eee")]
cols = st.columns(len(stats))
for c,(l,v,cl) in zip(cols,stats):
    c.markdown(f'<div class="stat"><div class="lbl">{l}</div><div class="val" style="color:{cl}">{v}</div></div>', unsafe_allow_html=True)

st.markdown("")
left, right = st.columns([3,2], gap="medium")

# ── Chart ──
with left:
    sym = st.selectbox("Chart", cfg.get("symbols",["XAUUSD"]), index=0)
    tvmap = {"XAUUSD":"TVC:GOLD","USDCAD":"FX:USDCAD","EURJPY":"FX:EURJPY",
             "AUDUSD":"FX:AUDUSD","EURUSD":"FX:EURUSD","GBPUSD":"FX:GBPUSD",
             "USDJPY":"FX:USDJPY","GBPJPY":"FX:GBPJPY"}
    tv = tvmap.get(sym, f"FX:{sym}")
    components.html(f"""
<div style="height:420px;border-radius:12px;overflow:hidden">
<div id="tv" style="height:420px"></div>
<script src="https://s3.tradingview.com/tv.js"></script>
<script>new TradingView.widget({{"container_id":"tv","width":"100%","height":420,
"symbol":"{tv}","interval":"60","theme":"dark","style":"1","locale":"en",
"hide_top_toolbar":false,"allow_symbol_change":true,
"studies":["RSI@tv-basicstudies","MACD@tv-basicstudies"]}});</script>
</div>""", height=430)

# ── Positions + history ──
with right:
    st.markdown("#### 📊 Open Positions")
    if not state["positions"]:
        st.caption("No open positions.")
    for p in state["positions"]:
        pnl = p.get("pnl",0); c = "#00FF88" if pnl>=0 else "#FF4444"
        st.markdown(f"""<div style="background:#0d0d22;border:1px solid #1a1a3a;border-left:3px solid {c};
border-radius:8px;padding:9px 12px;margin:3px 0;font-size:.82rem">
<div style="display:flex;justify-content:space-between">
<b>{p['symbol']}</b><span style="color:{'#00FF88' if p['direction']=='Buy' else '#FF4444'}">{'▲' if p['direction']=='Buy' else '▼'} {p['direction']}</span>
<b style="color:{c}">${pnl:+.2f}</b></div>
<div style="color:#555;font-size:.72rem">Entry {fmt(p['entry'],p['symbol'])} · Now {fmt(p.get('current_price',p['entry']),p['symbol'])} · SL {fmt(p['sl'],p['symbol'])} · TP {fmt(p['tp'],p['symbol'])}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("#### 📜 Recent Trades")
    if not trades:
        st.caption("No closed trades yet.")
    for t in trades[:15]:
        pnl = t.get("pnl",0); c = "#00FF88" if pnl>=0 else "#FF4444"
        ic = "✅" if pnl>=0 else "❌"
        st.markdown(f"""<div style="background:#0d0d22;border:1px solid #1a1a3a;border-left:3px solid {c};
border-radius:8px;padding:8px 12px;margin:3px 0;font-size:.8rem">
<div style="display:flex;justify-content:space-between">
<b>{t['symbol']} {t['direction']}</b><span>{t.get('reason','')}</span>
<b style="color:{c}">{ic} ${pnl:+.2f}</b></div>
<div style="color:#444;font-size:.7rem">{fmt(t['entry'],t['symbol'])} → {fmt(t.get('exit',0),t['symbol'])} · bal ${t.get('balance_after',0):.2f} · {t.get('closed_at','')[:16]}</div>
</div>""", unsafe_allow_html=True)

# ── Equity curve ──
if trades:
    st.markdown("#### 📈 Equity Curve")
    labels = ["Start"] + [f"#{t['id']}" for t in reversed(trades)]
    fig = go.Figure(go.Scatter(x=labels, y=bals, mode="lines+markers",
        line=dict(color="#00D4FF",width=2), fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)"))
    fig.add_hline(y=sb, line=dict(color="#555",dash="dash",width=1))
    fig.update_layout(height=240, template="plotly_dark", paper_bgcolor="#07071a",
        plot_bgcolor="#07071a", margin=dict(t=10,b=30,l=0,r=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Engine runs on GitHub Actions every 15 min · dashboard auto-refreshes 60s · {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

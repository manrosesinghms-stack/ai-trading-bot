"""
AI Trading Terminal — Paper & Live Trading
Auto-refreshes every 60s, detects live signals, places paper/live trades.
Run:  python -m streamlit run quick_trade.py
"""
from __future__ import annotations
import os, time, json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#06060f; }
  [data-testid="stSidebar"] { background:#0a0a18; }
  section[data-testid="stMain"] > div { padding-top:.5rem; }

  /* Top bar */
  .topbar { display:flex; align-items:center; justify-content:space-between;
            background:#0d0d1f; border:1px solid #1a1a3a; border-radius:10px;
            padding:10px 18px; margin-bottom:10px; }
  .topbar-price { font-size:2rem; font-weight:800; color:#fff; }
  .topbar-pair  { font-size:.85rem; color:#666; }
  .topbar-chg-up  { color:#00FF88; font-size:1rem; font-weight:700; }
  .topbar-chg-dn  { color:#FF4444; font-size:1rem; font-weight:700; }

  /* Signal badge */
  .sig-live-buy  { display:inline-flex; align-items:center; gap:8px;
                   background:#0a2a16; border:2px solid #00FF88;
                   border-radius:10px; padding:8px 18px;
                   font-size:1.3rem; font-weight:800; color:#00FF88; }
  .sig-live-sell { display:inline-flex; align-items:center; gap:8px;
                   background:#2a0a0a; border:2px solid #FF4444;
                   border-radius:10px; padding:8px 18px;
                   font-size:1.3rem; font-weight:800; color:#FF4444; }
  .sig-live-hold { display:inline-flex; align-items:center; gap:8px;
                   background:#1a1a0a; border:2px solid #FFAA00;
                   border-radius:10px; padding:8px 18px;
                   font-size:1.3rem; font-weight:800; color:#FFAA00; }

  /* Pulse dot */
  @keyframes pulse { 0%{opacity:1} 50%{opacity:.3} 100%{opacity:1} }
  .dot-live { width:10px; height:10px; border-radius:50%;
              display:inline-block; animation:pulse 1.2s infinite; }
  .dot-buy  { background:#00FF88; }
  .dot-sell { background:#FF4444; }
  .dot-hold { background:#FFAA00; animation:none; }

  /* Confidence bar */
  .conf-wrap { background:#1a1a2a; border-radius:6px; height:10px; width:100%; margin:6px 0; }

  /* Strategy rows */
  .strat-row { display:flex; justify-content:space-between; align-items:center;
               padding:5px 0; border-bottom:1px solid #111128; font-size:.85rem; }
  .strat-name { color:#999; }
  .s-buy  { color:#00FF88; font-weight:700; }
  .s-sell { color:#FF4444; font-weight:700; }
  .s-hold { color:#555; }

  /* Cards */
  .card { background:#0d0d1f; border:1px solid #1a1a3a; border-radius:12px; padding:14px 16px; margin-bottom:8px; }
  .card-title { color:#555; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:8px; }

  /* Trade param rows */
  .tp-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #111128; font-size:.88rem; }
  .tp-label { color:#555; }
  .tp-val   { color:#eee; font-weight:600; }
  .tp-green { color:#00FF88; font-weight:700; }
  .tp-red   { color:#FF4444; font-weight:700; }

  /* Position rows */
  .pos-row { display:flex; justify-content:space-between; align-items:center;
             background:#0d0d1f; border:1px solid #1a1a3a; border-radius:8px;
             padding:10px 14px; margin:4px 0; font-size:.88rem; }
  .pos-profit { color:#00FF88; font-weight:700; }
  .pos-loss   { color:#FF4444; font-weight:700; }

  /* Refresh bar */
  .refresh-bar { background:#0d0d1f; border:1px solid #1a1a3a; border-radius:8px;
                 padding:6px 14px; font-size:.8rem; color:#555;
                 display:flex; justify-content:space-between; margin-bottom:8px; }

  /* Buttons */
  div[data-testid="stButton"] > button { border-radius:8px; font-weight:700; }

  /* Paper balance bar */
  .paper-bar { background:#0d0d1f; border:1px solid #1a1a3a; border-radius:10px;
               padding:8px 18px; display:flex; gap:30px; margin-bottom:10px;
               font-size:.88rem; }
  .pb-item   { display:flex; flex-direction:column; align-items:center; }
  .pb-label  { color:#555; font-size:.7rem; }
  .pb-val    { color:#eee; font-weight:700; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("mt5_ok",         False),
    ("mt5_creds",      {}),
    ("last_signal",    None),
    ("last_price",     None),
    ("paper_balance",  10000.0),
    ("paper_positions", []),
    ("paper_trades",   []),
    ("auto_scan",      False),
    ("refresh_count",  0),
    ("last_refresh",   "Never"),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Constants ──────────────────────────────────────────────────────────────────
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","XAUUSD","AUDUSD","USDCAD","GBPJPY","EURJPY"]
TFS     = ["M15","M30","H1","H4","D1"]
PIP_MAP = {"EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
           "USDCHF":.0001,"NZDUSD":.0001,"USDJPY":.01,"EURJPY":.01,
           "GBPJPY":.01,"XAUUSD":.1}

def pip(sym): return PIP_MAP.get(sym, .0001)

def fmt(p, sym):
    d = 3 if "JPY" in sym else (2 if "XAU" in sym else 5)
    return f"{p:.{d}f}"

def sig_cls(s):
    return "s-buy" if s=="Buy" else ("s-sell" if s=="Sell" else "s-hold")

def sig_icon(s):
    return "▲ Buy" if s=="Buy" else ("▼ Sell" if s=="Sell" else "— Hold")

# ── Execute helper ─────────────────────────────────────────────────────────────
def place_paper_trade(sym, direction, entry, sl_price, tp_price, lots, risk_usd):
    st.session_state.paper_positions.append({
        "id":        len(st.session_state.paper_trades) + len(st.session_state.paper_positions),
        "symbol":    sym,
        "direction": direction,
        "entry":     entry,
        "sl":        sl_price,
        "tp":        tp_price,
        "lots":      lots,
        "risk_usd":  risk_usd,
        "opened":    datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "pnl":       0.0,
    })

def place_live_trade(sym, direction, lots, sl_price, tp_price, comment="AI Trade"):
    try:
        from mt5_connector import place_order
        ticket, msg = place_order(sym, direction, lots, sl_price, tp_price, comment)
        return ticket, msg
    except Exception as e:
        return None, str(e)

def close_paper_position(idx, current_price):
    pos  = st.session_state.paper_positions[idx]
    p    = pip(pos["symbol"])
    pnl_pips = (current_price - pos["entry"]) / p
    if pos["direction"] == "Sell":
        pnl_pips = -pnl_pips
    pnl_pips -= 1.5   # spread
    r_mult = pnl_pips / ((abs(pos["entry"] - pos["sl"])) / p)
    pnl    = pos["risk_usd"] * r_mult
    st.session_state.paper_balance += pnl
    pos["exit"]   = current_price
    pos["pnl"]    = round(pnl, 2)
    pos["closed"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.session_state.paper_trades.append(pos)
    st.session_state.paper_positions.pop(idx)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 📊 Chart")
    symbol    = st.selectbox("Symbol",    SYMBOLS, key="sym")
    timeframe = st.selectbox("Timeframe", TFS, index=2, key="tf")

    st.markdown("### 🤖 Signal")
    min_conf  = st.slider("Min confidence %", 50, 95, 70, 5, key="mc")
    auto_scan = st.toggle("Auto-scan (place paper trade on signal)", value=st.session_state.auto_scan, key="as_toggle")
    st.session_state.auto_scan = auto_scan
    refresh_secs = st.select_slider("Auto-refresh", [30, 60, 120, 300], value=60, key="ref_int")

    st.markdown("### 🔌 MT5 Connection")
    with st.expander("Connect Live Account"):
        login_in   = st.text_input("Login",    placeholder="Account #")
        pass_in    = st.text_input("Password", type="password")
        server_in  = st.text_input("Server",   placeholder="ICMarkets-Demo01")
        api_key_in = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        if st.button("🔌 Connect", use_container_width=True):
            if api_key_in:
                os.environ["ANTHROPIC_API_KEY"] = api_key_in
                st.session_state.mt5_creds["api_key"] = api_key_in
            try:
                from mt5_connector import connect
                ok, msg = connect(int(login_in) if login_in.strip().isdigit() else None, pass_in, server_in)
                if ok:
                    st.session_state.mt5_ok = True
                    st.success(msg)
                else:
                    st.error(msg)
            except Exception as e:
                st.error(str(e))

    if st.session_state.mt5_ok:
        st.success("✅ MT5 Connected")
        if st.button("⛔ Disconnect", use_container_width=True):
            try:
                from mt5_connector import disconnect
                disconnect()
            except Exception: pass
            st.session_state.mt5_ok = False
            st.rerun()

    st.markdown("### 📺 Chart Source")
    use_tv_chart = st.toggle("Use TradingView chart", value=False, key="use_tv")

    st.divider()
    if st.button("🔄 Reset Paper Account", use_container_width=True):
        st.session_state.paper_balance   = 10000.0
        st.session_state.paper_positions = []
        st.session_state.paper_trades    = []
        st.rerun()
    if st.button("🔃 Refresh Now", use_container_width=True):
        st.session_state.refresh_count += 1
        st.cache_data.clear()
        st.rerun()

# ── Auto refresh ────────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_secs * 1000, key="auto_ref")
except ImportError:
    pass

# ── Header ─────────────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns([3, 2, 2])
with h1:
    mode = "⚡ MT5 Live" if st.session_state.mt5_ok else "📝 Paper Mode"
    st.markdown(f"## 📈 AI Trading Terminal &nbsp; <span style='font-size:.85rem; color:#555'>{mode}</span>", unsafe_allow_html=True)
with h2:
    st.markdown(f"<div style='padding-top:8px; color:#666; font-size:.85rem'>Symbol: <b style='color:#eee'>{symbol}</b> &nbsp; TF: <b style='color:#eee'>{timeframe}</b> &nbsp; Min Conf: <b style='color:#eee'>{min_conf}%</b></div>", unsafe_allow_html=True)
with h3:
    auto_label = "🤖 Auto-scan ON" if auto_scan else "⏸ Auto-scan OFF"
    auto_color = "#00FF88" if auto_scan else "#555"
    st.markdown(f"<div style='padding-top:8px; text-align:right; font-size:.85rem; color:{auto_color}'>{auto_label} &nbsp;|&nbsp; 🔄 {refresh_secs}s refresh</div>", unsafe_allow_html=True)

# ── Paper balance bar ──────────────────────────────────────────────────────────
open_pnl  = sum(p.get("pnl", 0) for p in st.session_state.paper_positions)
total_trades = len(st.session_state.paper_trades)
wins = [t for t in st.session_state.paper_trades if t.get("pnl", 0) > 0]
win_rate = f"{len(wins)/total_trades*100:.0f}%" if total_trades else "—"
paper_equity = st.session_state.paper_balance + open_pnl

pb_color  = "#00FF88" if open_pnl >= 0 else "#FF4444"
bal_color = "#00FF88" if paper_equity >= 10000 else "#FF4444"

st.markdown(f"""
<div class="paper-bar">
  <div class="pb-item"><span class="pb-label">PAPER BALANCE</span>
    <span class="pb-val" style="color:{bal_color}">${paper_equity:,.2f}</span></div>
  <div class="pb-item"><span class="pb-label">OPEN P/L</span>
    <span class="pb-val" style="color:{pb_color}">${open_pnl:+,.2f}</span></div>
  <div class="pb-item"><span class="pb-label">OPEN POSITIONS</span>
    <span class="pb-val">{len(st.session_state.paper_positions)}</span></div>
  <div class="pb-item"><span class="pb-label">TOTAL TRADES</span>
    <span class="pb-val">{total_trades}</span></div>
  <div class="pb-item"><span class="pb-label">WIN RATE</span>
    <span class="pb-val">{win_rate}</span></div>
  <div class="pb-item"><span class="pb-label">LAST REFRESH</span>
    <span class="pb-val">{st.session_state.last_refresh}</span></div>
</div>
""", unsafe_allow_html=True)

# ── Fetch data & run analysis ──────────────────────────────────────────────────
@st.cache_data(ttl=refresh_secs)
def get_analysis(symbol, timeframe, _tick=0):
    """Cache-busted every refresh_secs seconds by passing tick counter."""
    import yfinance as yf
    from data_manager import PAIRS as YF_PAIRS
    from strategies import calculate_all_indicators, run_all_strategies

    ticker  = YF_PAIRS.get(symbol, symbol + "=X")
    per_map = {"M15":"5d","M30":"10d","H1":"30d","H4":"90d","D1":"2y"}
    iv_map  = {"M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}

    df = yf.download(ticker, period=per_map.get(timeframe,"30d"),
                     interval=iv_map.get(timeframe,"1h"),
                     auto_adjust=True, progress=False)
    if df.empty:
        return None, None, None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    if len(df) < 30:
        return None, None, None, None

    df   = calculate_all_indicators(df)
    res  = run_all_strategies(df)
    price = float(df["Close"].iloc[-1])
    atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else price * 0.001
    return df, res, price, atr

# Use refresh_count as cache-buster
tick = st.session_state.refresh_count
with st.spinner("⚡ Fetching live data..."):
    df, res, current_price, atr_val = get_analysis(symbol, timeframe, tick)

if df is None:
    st.error("Could not fetch data. Check your internet connection and try again.")
    st.stop()

st.session_state.last_refresh = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
st.session_state.last_price   = current_price

# ── Calculate signal & trade params ───────────────────────────────────────────
action     = res["overall"]
confidence = res["overall_strength"]
conf_pct   = int(confidence * 100)
is_live    = action != "Hold" and conf_pct >= min_conf

# SL / TP (ATR-based)
sl_p = round(atr_val * 1.5 / pip(symbol))
tp_p = round(atr_val * 2.5 / pip(symbol))

if action == "Buy":
    sl_price = round(current_price - sl_p * pip(symbol), 5)
    tp_price = round(current_price + tp_p * pip(symbol), 5)
elif action == "Sell":
    sl_price = round(current_price + sl_p * pip(symbol), 5)
    tp_price = round(current_price - tp_p * pip(symbol), 5)
else:
    sl_price = tp_price = 0.0

lots = max(0.01, min(round(st.session_state.paper_balance * 0.01 / max(sl_p * 10, 1), 2), 10.0))
risk_usd   = round(lots * sl_p * 10, 2)
reward_usd = round(lots * tp_p * 10, 2)
rr         = round(tp_p / max(sl_p, 1), 2)

# Auto-scan: place paper trade automatically if signal is live
if auto_scan and is_live:
    # Don't re-enter same direction if already in position
    already_in = any(
        p["symbol"] == symbol and p["direction"] == action
        for p in st.session_state.paper_positions
    )
    if not already_in:
        place_paper_trade(symbol, action, current_price, sl_price, tp_price, lots, risk_usd)
        st.toast(f"🤖 Auto-scan placed {action} on {symbol} @ {fmt(current_price, symbol)}", icon="✅")

# Update open position P&L
for pos in st.session_state.paper_positions:
    p_size = pip(pos["symbol"])
    pnl_pips = (current_price - pos["entry"]) / p_size
    if pos["direction"] == "Sell":
        pnl_pips = -pnl_pips
    r_mult = pnl_pips / max(abs(pos["entry"] - pos["sl"]) / p_size, 1)
    pos["pnl"] = round(pos["risk_usd"] * r_mult, 2)

# ── Main 3-column layout ───────────────────────────────────────────────────────
col_chart, col_signals, col_trade = st.columns([2.2, 1.3, 1.5], gap="small")

# ══════════════════ CHART ═══════════════════════════════════════════════════════
with col_chart:
    # Price header
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    chg        = current_price - prev_close
    chg_pct    = chg / prev_close * 100
    chg_css    = "topbar-chg-up" if chg >= 0 else "topbar-chg-dn"
    chg_icon   = "▲" if chg >= 0 else "▼"

    st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-pair">{symbol} · {timeframe}</div>
    <div class="topbar-price">{fmt(current_price, symbol)}</div>
  </div>
  <div>
    <span class="{chg_css}">{chg_icon} {fmt(abs(chg), symbol)} ({chg_pct:+.2f}%)</span>
    <div style="font-size:.75rem; color:#444; margin-top:4px;">ATR {fmt(atr_val, symbol)}</div>
  </div>
  <div style="font-size:.75rem; color:#444; text-align:right;">
    {'✅ MT5 Live' if st.session_state.mt5_ok else '📝 Paper Mode'}<br>
    🔄 Refreshes every {refresh_secs}s
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Chart: TradingView or built-in ──────────────────────────────────────────
    TV_SYMBOL_MAP = {
        "EURUSD":"FX:EURUSD","GBPUSD":"FX:GBPUSD","USDJPY":"FX:USDJPY",
        "AUDUSD":"FX:AUDUSD","USDCAD":"FX:USDCAD","USDCHF":"FX:USDCHF",
        "GBPJPY":"FX:GBPJPY","EURJPY":"FX:EURJPY","XAUUSD":"TVC:GOLD",
    }
    TV_TF_MAP = {"M15":"15","M30":"30","H1":"60","H4":"240","D1":"D"}

    if use_tv_chart:
        tv_sym = TV_SYMBOL_MAP.get(symbol, f"FX:{symbol}")
        tv_tf  = TV_TF_MAP.get(timeframe, "60")
        tv_html = f"""
<div class="tradingview-widget-container" style="height:440px">
  <div id="tv_chart" style="height:440px"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
  new TradingView.widget({{
    "container_id": "tv_chart",
    "width":        "100%",
    "height":       440,
    "symbol":       "{tv_sym}",
    "interval":     "{tv_tf}",
    "timezone":     "Etc/UTC",
    "theme":        "dark",
    "style":        "1",
    "locale":       "en",
    "hide_top_toolbar": false,
    "allow_symbol_change": true,
    "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],
    "show_popup_button": false
  }});
  </script>
</div>"""
        import streamlit.components.v1 as components
        components.html(tv_html, height=450, scrolling=False)
    else:
        chart_df = df.tail(80)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"], name="Price",
            increasing_line_color="#00FF88", decreasing_line_color="#FF4444",
            increasing_fillcolor="#00FF88", decreasing_fillcolor="#FF4444",
        ), row=1, col=1)
        for col_name, clr, w in [("EMA20","#FFD700",1.2),("EMA50","#FF8C00",1.8),("EMA200","#FF4444",2)]:
            if col_name in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col_name],
                                         line=dict(color=clr,width=w), showlegend=False), row=1, col=1)
        if "BB_Upper" in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"],
                                     line=dict(color="rgba(120,80,255,0.5)",width=1,dash="dot"),
                                     showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"],
                                     line=dict(color="rgba(120,80,255,0.5)",width=1,dash="dot"),
                                     fill="tonexty", fillcolor="rgba(120,80,255,0.04)",
                                     showlegend=False), row=1, col=1)
        if action != "Hold" and sl_price and tp_price:
            fig.add_hline(y=sl_price, line=dict(color="#FF4444",width=1.5,dash="dash"), row=1, col=1)
            fig.add_hline(y=tp_price, line=dict(color="#00FF88",width=1.5,dash="dash"), row=1, col=1)
        if "MACD" in chart_df.columns:
            colors = ["#00FF88" if v >= 0 else "#FF4444" for v in chart_df["MACD_Hist"]]
            fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["MACD_Hist"],
                                 marker_color=colors, opacity=0.7, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"],
                                     line=dict(color="#00D4FF",width=1.2), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_Signal"],
                                     line=dict(color="#FF8C00",width=1.2), showlegend=False), row=2, col=1)
        fig.update_layout(height=420, template="plotly_dark", paper_bgcolor="#06060f",
                          plot_bgcolor="#06060f", xaxis_rangeslider_visible=False,
                          margin=dict(t=10,b=10,l=0,r=0))
        fig.update_xaxes(gridcolor="#111128"); fig.update_yaxes(gridcolor="#111128")
        st.plotly_chart(fig, use_container_width=True)

    # ── Webhook signals panel ─────────────────────────────────────────────────
    from pathlib import Path
    sig_file = Path(__file__).parent / "data" / "webhook_signals.json"
    webhook_url_file = Path(__file__).parent / "data" / "webhook_url.txt"

    with st.expander("📡 TradingView Webhook Signals", expanded=False):
        # Show URL
        if webhook_url_file.exists():
            url = webhook_url_file.read_text().strip()
            st.success(f"**Webhook URL:** `{url}`")
            st.caption("Paste this URL in TradingView → Alert → Webhook URL")
        else:
            st.info("**Start the webhook server** to get your TradingView URL:\n\n"
                    "Open a new Command Prompt and run:\n```\nstart_webhook.bat\n```")

        # Show recent signals
        if sig_file.exists():
            try:
                import json as _json
                sigs = _json.loads(sig_file.read_text())
                if sigs:
                    st.markdown(f"**Last {min(5,len(sigs))} signals received:**")
                    for s in sigs[:5]:
                        res = s.get("result", {})
                        ok  = res.get("ok", False)
                        icon = "✅" if ok else "❌"
                        st.markdown(
                            f"{icon} `{s['time']}` &nbsp; **{s['symbol']}** &nbsp; "
                            f"{'🟢' if s['action']=='BUY' else '🔴'} **{s['action']}** &nbsp; "
                            f"{s.get('lots','')} lots"
                        )
                else:
                    st.caption("No signals received yet.")
            except Exception:
                st.caption("Waiting for signals...")
        else:
            st.caption("No signals yet. Start the webhook server and create a TradingView alert.")

# ══════════════════ SIGNALS ══════════════════════════════════════════════════════
with col_signals:
    # Live signal badge
    badge_css = "sig-live-buy" if action == "Buy" else ("sig-live-sell" if action == "Sell" else "sig-live-hold")
    dot_cls   = "dot-buy" if action == "Buy" else ("dot-sell" if action == "Sell" else "dot-hold")
    label     = f"{'🟢 LIVE BUY' if action=='Buy' else ('🔴 LIVE SELL' if action=='Sell' else '🟡 HOLD')}"
    live_badge = "● SIGNAL LIVE" if is_live else "● Monitoring..."

    st.markdown(f"""
<div style="margin-bottom:10px">
  <div class="{badge_css}">
    <span class="dot-live {dot_cls}"></span>
    {label}
  </div>
  <div style="font-size:.75rem; color:#555; margin-top:4px;">{live_badge}</div>
</div>
""", unsafe_allow_html=True)

    # Confidence bar
    conf_color = "#00FF88" if conf_pct >= 70 else ("#FFAA00" if conf_pct >= 50 else "#FF4444")
    st.markdown(f"""
<div style="margin-bottom:10px">
  <div style="font-size:.75rem; color:#666; margin-bottom:3px;">AI Confidence</div>
  <div class="conf-wrap">
    <div style="background:{conf_color}; height:10px; border-radius:6px; width:{conf_pct}%"></div>
  </div>
  <div style="font-size:.9rem; color:{conf_color}; font-weight:800;">{conf_pct}%
    &nbsp;<span style="color:#444; font-weight:400; font-size:.75rem;">
    {res['buy_count']}B / {res['sell_count']}S / {res['hold_count']}H</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Strategy rows
    strat_html = '<div class="card">'
    strat_html += '<div class="card-title">8 Strategies</div>'
    for name, d in res["individual"].items():
        css  = sig_cls(d["signal"])
        icon = sig_icon(d["signal"])
        short_name = name.replace(" Crossover","").replace(" Trend","").replace(" Action","")
        strat_html += f'<div class="strat-row"><span class="strat-name">{short_name}</span><span class="{css}">{icon}</span></div>'
    strat_html += "</div>"
    st.markdown(strat_html, unsafe_allow_html=True)

    # Sessions
    from sessions import get_session_status, get_session_quality
    quality, q_desc = get_session_quality()
    q_color_map = {"Excellent":"#00FF88","Good":"#00FF88","Fair":"#FFAA00","Poor":"#FF4444"}
    q_color = q_color_map.get(quality, "#888")

    sess_html = '<div class="card">'
    sess_html += '<div class="card-title">Sessions (UTC)</div>'
    for s in get_session_status():
        dot = f'<span style="color:{"#00FF88" if s["active"] else "#333"}">{"●" if s["active"] else "○"}</span>'
        sess_html += f'<div style="font-size:.78rem; padding:2px 0; color:{"#ccc" if s["active"] else "#444"}">{dot} {s["session"]}</div>'
    sess_html += f'<div style="margin-top:6px; font-size:.75rem;">Quality: <b style="color:{q_color}">{quality}</b></div>'
    sess_html += "</div>"
    st.markdown(sess_html, unsafe_allow_html=True)

    # News warning
    try:
        from news_calendar import get_upcoming_events, should_block_trade
        blocked, reason = should_block_trade(symbol, 30)
        events = get_upcoming_events(symbol, 120)
        if blocked:
            st.error(f"🚨 {reason[:60]}...")
        elif events:
            st.warning(f"⚠️ {events[0]['currency']} {events[0]['title'][:30]} in {events[0]['mins_away']}min")
    except Exception:
        pass

# ══════════════════ TRADE PANEL ══════════════════════════════════════════════════
with col_trade:
    st.markdown('<div class="card-title">Trade Setup</div>', unsafe_allow_html=True)

    # Manual overrides
    with st.expander("⚙️ Adjust Parameters", expanded=False):
        sl_p   = st.number_input("SL pips", 5, 500, max(sl_p, 5), key="sl_ov")
        tp_p   = st.number_input("TP pips", 5, 1000, max(tp_p, 5), key="tp_ov")
        lots   = st.number_input("Lot size", 0.01, 10.0, lots, 0.01, format="%.2f", key="lots_ov")
        if action == "Buy":
            sl_price = round(current_price - sl_p * pip(symbol), 5)
            tp_price = round(current_price + tp_p * pip(symbol), 5)
        elif action == "Sell":
            sl_price = round(current_price + sl_p * pip(symbol), 5)
            tp_price = round(current_price - tp_p * pip(symbol), 5)
        risk_usd   = round(lots * sl_p * 10, 2)
        reward_usd = round(lots * tp_p * 10, 2)
        rr         = round(tp_p / max(sl_p, 1), 2)

    # Trade parameters card
    if action != "Hold":
        sl_dir = "below" if action == "Buy" else "above"
        tp_dir = "above" if action == "Buy" else "below"
        st.markdown(f"""
<div class="card">
  <div class="tp-row"><span class="tp-label">Direction</span>
    <span style="color:{'#00FF88' if action=='Buy' else '#FF4444'}; font-weight:800">{action.upper()}</span></div>
  <div class="tp-row"><span class="tp-label">Entry</span>
    <span class="tp-val">{fmt(current_price, symbol)}</span></div>
  <div class="tp-row"><span class="tp-label">Stop Loss ({sl_p}p)</span>
    <span class="tp-red">{fmt(sl_price, symbol)}</span></div>
  <div class="tp-row"><span class="tp-label">Take Profit ({tp_p}p)</span>
    <span class="tp-green">{fmt(tp_price, symbol)}</span></div>
  <div class="tp-row"><span class="tp-label">R : R</span>
    <span class="tp-val">1 : {rr}</span></div>
  <div class="tp-row"><span class="tp-label">Lots</span>
    <span class="tp-val">{lots}</span></div>
  <div class="tp-row" style="border:none">
    <span class="tp-label">Risk → Reward</span>
    <span class="tp-val">${risk_usd} → <span class="tp-green">${reward_usd}</span></span></div>
</div>
""", unsafe_allow_html=True)

        # ── Paper trade buttons ──
        st.markdown("**📝 Paper Trade**")
        pb1, pb2 = st.columns(2)
        with pb1:
            if st.button("🟢 BUY Paper", use_container_width=True,
                         type="primary" if action=="Buy" else "secondary",
                         key="paper_buy"):
                place_paper_trade(symbol, "Buy", current_price, sl_price, tp_price, lots, risk_usd)
                st.toast(f"📝 Paper BUY {symbol} @ {fmt(current_price,symbol)}", icon="✅")
                st.rerun()
        with pb2:
            if st.button("🔴 SELL Paper", use_container_width=True,
                         type="primary" if action=="Sell" else "secondary",
                         key="paper_sell"):
                place_paper_trade(symbol, "Sell", current_price, sl_price, tp_price, lots, risk_usd)
                st.toast(f"📝 Paper SELL {symbol} @ {fmt(current_price,symbol)}", icon="✅")
                st.rerun()

        # ── Live MT5 buttons ──
        if st.session_state.mt5_ok:
            st.markdown("**⚡ Live Trade (MT5)**")
            lb1, lb2 = st.columns(2)
            with lb1:
                if st.button("🚀 BUY Live", use_container_width=True, type="primary", key="live_buy"):
                    ticket, msg = place_live_trade(symbol, "Buy", lots, sl_price, tp_price)
                    if ticket:
                        st.success(f"✅ BUY #{ticket}")
                    else:
                        st.error(f"❌ {msg}")
            with lb2:
                if st.button("📉 SELL Live", use_container_width=True, type="primary", key="live_sell"):
                    ticket, msg = place_live_trade(symbol, "Sell", lots, sl_price, tp_price)
                    if ticket:
                        st.success(f"✅ SELL #{ticket}")
                    else:
                        st.error(f"❌ {msg}")
    else:
        st.markdown("""
<div class="card" style="text-align:center; padding:20px">
  <div style="font-size:2rem">🟡</div>
  <div style="color:#FFAA00; font-weight:700; font-size:1.1rem; margin:8px 0">HOLD</div>
  <div style="color:#555; font-size:.85rem">No clear signal.<br>Waiting for confluence...</div>
</div>
""", unsafe_allow_html=True)



# ══════════════════ OPEN POSITIONS ═══════════════════════════════════════════════
st.divider()
st.markdown("### 📋 Open Positions")

# Live MT5 positions
if st.session_state.mt5_ok:
    try:
        from mt5_connector import get_open_positions, close_position
        live_pos = get_open_positions()
        if live_pos:
            for p in live_pos:
                icon = "🟢" if p["profit"] >= 0 else "🔴"
                c1,c2,c3,c4,c5 = st.columns([1.5,1,1,1,0.6])
                c1.markdown(f"**{p['symbol']}** {p['type']}")
                c2.markdown(f"{p['volume']} lots")
                c3.markdown(f"Entry: `{p['open_price']:.5f}`")
                pnl_color = "green" if p["profit"]>=0 else "red"
                c4.markdown(f"<span style='color:{'#00FF88' if p['profit']>=0 else '#FF4444'}; font-weight:700'>${p['profit']:+.2f}</span>", unsafe_allow_html=True)
                with c5:
                    if st.button("✕", key=f"close_live_{p['ticket']}"):
                        ok, msg = close_position(p["ticket"])
                        st.toast(f"Closed #{p['ticket']}" if ok else f"Error: {msg}")
                        st.rerun()
        else:
            st.info("No live positions.")
    except Exception as e:
        st.error(f"MT5 positions error: {e}")

# Paper positions
if st.session_state.paper_positions:
    st.markdown("**📝 Paper Positions**")
    for i, pos in enumerate(st.session_state.paper_positions):
        pnl     = pos.get("pnl", 0)
        pnl_col = "#00FF88" if pnl >= 0 else "#FF4444"
        pnl_icon= "▲" if pnl >= 0 else "▼"

        c1,c2,c3,c4,c5,c6 = st.columns([1.2,0.8,1,1,1,0.6])
        c1.markdown(f"**{pos['symbol']}** {pos['direction']}")
        c2.markdown(f"{pos['lots']}L")
        c3.markdown(f"@ `{fmt(pos['entry'],pos['symbol'])}`")
        c4.markdown(f"SL `{fmt(pos['sl'],pos['symbol'])}`")
        c5.markdown(f"<span style='color:{pnl_col}; font-weight:700'>{pnl_icon} ${abs(pnl):.2f}</span>",
                    unsafe_allow_html=True)
        with c6:
            if st.button("✕", key=f"close_paper_{i}_{pos['id']}"):
                close_paper_position(i, current_price)
                st.toast(f"Paper {pos['direction']} closed: ${pos.get('pnl',0):+.2f}")
                st.rerun()
elif not st.session_state.mt5_ok:
    st.info("No open paper positions. Click **BUY Paper** or **SELL Paper** above to open one.")

# ══════════════════ TRADE HISTORY ════════════════════════════════════════════════
if st.session_state.paper_trades:
    st.divider()
    with st.expander(f"📜 Paper Trade History ({len(st.session_state.paper_trades)} trades)", expanded=False):
        df_hist = pd.DataFrame(st.session_state.paper_trades)
        total_pnl = df_hist["pnl"].sum()
        st.metric("Total Paper P/L", f"${total_pnl:+.2f}")
        cols_show = ["symbol","direction","entry","exit","pnl","opened","closed"]
        cols_show = [c for c in cols_show if c in df_hist.columns]
        st.dataframe(df_hist[cols_show], use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; color:#333; font-size:.75rem; margin-top:10px; padding:8px">
  AI Trading Terminal · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
  {'MT5 Live' if st.session_state.mt5_ok else 'Paper Mode'} ·
  Auto-refresh every {refresh_secs}s
</div>
""", unsafe_allow_html=True)

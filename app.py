"""
AI Trading Bot — MetaTrader 5 + Claude AI
Run:  streamlit run app.py
"""
from __future__ import annotations
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d1a; }
  .main-header { font-size:2.2rem; font-weight:700; color:#00D4FF; margin-bottom:0; }
  .sub-header  { color:#888; margin-top:0; }
  .sig-buy  { color:#00FF88; font-weight:bold; font-size:1.1rem; }
  .sig-sell { color:#FF4444; font-weight:bold; font-size:1.1rem; }
  .sig-hold { color:#FFAA00; font-weight:bold; font-size:1.1rem; }
  div[data-testid="stMetricValue"] { font-size:1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("mt5_connected", False),
    ("auto_trading", False),
    ("trade_log", []),
    ("last_analysis", {}),
    ("chat_history", []),
    ("mt5_creds", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ MT5 Connection")

    with st.expander("Credentials", expanded=not st.session_state.mt5_connected):
        mt5_login    = st.text_input("MT5 Login (account #)", placeholder="12345678")
        mt5_password = st.text_input("MT5 Password", type="password")
        mt5_server   = st.text_input("MT5 Server", placeholder="ICMarkets-Demo01")
        anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")

    col_btn, col_status = st.columns([2, 1])
    with col_btn:
        connect_btn = st.button("🔌 Connect", use_container_width=True)
    with col_status:
        if st.session_state.mt5_connected:
            st.success("Live")
        else:
            st.error("Off")

    if connect_btn:
        st.session_state.mt5_creds = {
            "login": mt5_login, "password": mt5_password,
            "server": mt5_server, "api_key": anthropic_key,
        }
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        try:
            from mt5_connector import connect
            login_int = int(mt5_login) if mt5_login.strip().isdigit() else None
            ok, msg = connect(login_int, mt5_password, mt5_server)
            if ok:
                st.session_state.mt5_connected = True
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.divider()
    st.markdown("## 📊 Settings")

    all_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                   "EURJPY", "GBPJPY", "XAUUSD", "USDCHF", "NZDUSD"]
    watch_list = st.multiselect("Watch List", all_symbols, default=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    active_symbol = st.selectbox("Active Symbol", watch_list or ["EURUSD"])
    timeframe = st.select_slider("Timeframe", ["M1","M5","M15","M30","H1","H4","D1"], value="H1")

    st.divider()
    st.markdown("## ⚠️ Risk")
    risk_pct     = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
    max_pos      = st.slider("Max Positions", 1, 10, 5)
    max_dd       = st.slider("Max Drawdown (%)", 5.0, 25.0, 10.0, 1.0)

    st.divider()
    col_at, col_toggle = st.columns([3, 1])
    col_at.markdown("### 🤖 Auto Trading")
    auto_on = col_toggle.toggle("", value=st.session_state.auto_trading, key="at_toggle")
    st.session_state.auto_trading = auto_on
    if auto_on:
        ai_thresh = st.slider("Min AI Confidence", 0.50, 0.95, 0.72, 0.01)
        st.warning("Bot will place trades automatically!")
    else:
        ai_thresh = 0.72
        st.info("Manual mode — review signals first")

    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 AI Trading Bot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">MetaTrader 5 · Forex · Claude AI Signals</p>', unsafe_allow_html=True)

tab_dash, tab_chart, tab_trade, tab_pos, tab_hist, tab_data, tab_bt, tab_paper, tab_chat = st.tabs([
    "📊 Dashboard", "📈 Charts", "🎯 Trade Setup",
    "📋 Positions", "📜 History",
    "📥 Data & KB", "🧪 Backtest", "📝 Paper Trade", "🤖 AI Chat",
])

# ═══════════════════════════ DASHBOARD ════════════════════════════════════════
with tab_dash:
    if not st.session_state.mt5_connected:
        st.info("Connect to MetaTrader 5 via the sidebar to see live data.")
        st.markdown("### What this bot does")
        st.markdown("""
- **7 technical strategies** run in parallel (RSI, MACD, EMA crossover, Bollinger Bands, ADX, Stochastic, Price Action)
- **Claude AI** analyses all signals together and gives a final Buy / Sell / Hold decision with confidence score
- **Auto-trading mode** places orders on MT5 automatically when confidence exceeds your threshold
- **Risk management** calculates correct lot size, SL and TP for every trade
- **Session awareness** — know which currency pairs to trade and when
        """)
    else:
        from mt5_connector import get_account_info, get_open_positions
        from sessions import get_session_status, get_session_quality

        acc  = get_account_info()
        pnl  = acc["profit"]
        pos  = get_open_positions()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Balance",     f"{acc['currency']} {acc['balance']:,.2f}")
        c2.metric("Equity",      f"{acc['currency']} {acc['equity']:,.2f}",
                  delta=f"{acc['equity']-acc['balance']:+,.2f}")
        c3.metric("Free Margin", f"{acc['currency']} {acc['free_margin']:,.2f}")
        c4.metric("Open P/L",    f"{pnl:+,.2f}",
                  delta=f"{'▲' if pnl >= 0 else '▼'} {abs(pnl):.2f}")
        c5.metric("Positions",   f"{len(pos)} / {max_pos}")

        st.divider()
        st.markdown("### 🌍 Trading Sessions")
        quality, q_desc = get_session_quality()
        badge = {"Excellent": "success", "Good": "success", "Fair": "warning", "Poor": "error"}[quality]
        getattr(st, badge)(f"**{quality}:** {q_desc}")

        cols = st.columns(4)
        for col, sess in zip(cols, get_session_status()):
            with col:
                if sess["active"]:
                    st.success(f"**{sess['session']}**\n\n✅ {sess['state']}\n\n*{', '.join(sess['best_pairs'][:2])}*")
                else:
                    st.error(f"**{sess['session']}**\n\n⏸ {sess['state']}")

        st.divider()
        st.markdown("### 🎯 Live Watchlist Signals")

        try:
            from mt5_connector import get_ohlcv, get_current_price
            from strategies import calculate_all_indicators, run_all_strategies

            rows = []
            with st.spinner("Scanning symbols..."):
                for sym in (watch_list or ["EURUSD"]):
                    df = get_ohlcv(sym, timeframe, 300)
                    if df is not None and len(df) > 60:
                        df   = calculate_all_indicators(df)
                        res  = run_all_strategies(df)
                        tick = get_current_price(sym)
                        rows.append({
                            "Symbol":  sym,
                            "Price":   f"{tick['bid']:.5f}" if tick else "—",
                            "Spread":  f"{tick['spread']:.1f}" if tick else "—",
                            "Signal":  res["overall"],
                            "Buy ✓":   res["buy_count"],
                            "Sell ✓":  res["sell_count"],
                            "RSI":     f"{df['RSI'].iloc[-1]:.1f}" if "RSI" in df else "—",
                            "Strength": f"{res['overall_strength']:.2f}",
                        })

            if rows:
                df_w = pd.DataFrame(rows)

                def _color(val):
                    if val == "Buy":  return "background:#0d2b0d; color:#00FF88"
                    if val == "Sell": return "background:#2b0d0d; color:#FF4444"
                    return "background:#2b2b0d; color:#FFAA00"

                st.dataframe(
                    df_w.style.applymap(_color, subset=["Signal"]),
                    use_container_width=True, hide_index=True,
                )
        except Exception as e:
            st.error(f"Watchlist scan failed: {e}")

# ═══════════════════════════ CHARTS ═══════════════════════════════════════════
with tab_chart:
    if not st.session_state.mt5_connected:
        st.warning("Connect to MT5 to view live charts.")
    else:
        from mt5_connector import get_ohlcv, get_current_price
        from strategies import calculate_all_indicators, run_all_strategies, get_support_resistance

        cc1, cc2, cc3 = st.columns([2, 1, 1])
        chart_sym = cc1.selectbox("Symbol", watch_list or ["EURUSD"], key="c_sym")
        chart_tf  = cc2.selectbox("Timeframe", ["M5","M15","M30","H1","H4","D1"], index=3, key="c_tf")
        num_bars  = cc3.number_input("Bars", 100, 1000, 300, 50)

        show_ema = st.checkbox("Show EMAs", True)
        show_bb  = st.checkbox("Show Bollinger Bands", True)
        show_sr  = st.checkbox("Show S/R Levels", True)

        try:
            df  = get_ohlcv(chart_sym, chart_tf, num_bars)
            if df is None or len(df) < 60:
                st.error("Not enough data — check symbol and timeframe.")
            else:
                df   = calculate_all_indicators(df)
                res  = run_all_strategies(df)
                sr   = get_support_resistance(df)
                tick = get_current_price(chart_sym)

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.60, 0.20, 0.20],
                    subplot_titles=(f"{chart_sym} {chart_tf}", "RSI (14)", "MACD"),
                )

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="Price",
                    increasing_line_color="#00FF88", decreasing_line_color="#FF4444",
                    increasing_fillcolor="#00FF88", decreasing_fillcolor="#FF4444",
                ), row=1, col=1)

                if show_ema:
                    for col_name, color, width in [("EMA20","#FFD700",1.2), ("EMA50","#FF8C00",1.5), ("EMA200","#FF4444",2)]:
                        if col_name in df:
                            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=col_name,
                                                     line=dict(color=color, width=width)), row=1, col=1)

                if show_bb and "BB_Upper" in df:
                    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                                             line=dict(color="rgba(100,100,255,0.5)", width=1, dash="dash")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                                             line=dict(color="rgba(100,100,255,0.5)", width=1, dash="dash"),
                                             fill="tonexty", fillcolor="rgba(100,100,255,0.04)"), row=1, col=1)

                if show_sr:
                    for lv in sr["resistance"][:2]:
                        fig.add_hline(y=lv, line=dict(color="#FF4444", width=1, dash="dot"), row=1, col=1)
                    for lv in sr["support"][:2]:
                        fig.add_hline(y=lv, line=dict(color="#00FF88", width=1, dash="dot"), row=1, col=1)

                if "RSI" in df:
                    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                             line=dict(color="#FFD700", width=1.5)), row=2, col=1)
                    for lvl, clr in [(70, "#FF4444"), (30, "#00FF88"), (50, "#555")]:
                        fig.add_hline(y=lvl, line=dict(color=clr, width=1, dash="dash"), row=2, col=1)

                if "MACD" in df:
                    colors = ["#00FF88" if v >= 0 else "#FF4444" for v in df["MACD_Hist"]]
                    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Hist",
                                         marker_color=colors, opacity=0.6), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD",
                                             line=dict(color="#00D4FF", width=1.5)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                                             line=dict(color="#FF8C00", width=1.5)), row=3, col=1)

                fig.update_layout(
                    height=720, template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=40, b=20, l=10, r=10),
                    legend=dict(orientation="h", y=1.03),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                col_l, col_r = st.columns([1, 2])
                with col_l:
                    overall = res["overall"]
                    icon = "🟢" if overall == "Buy" else "🔴" if overall == "Sell" else "🟡"
                    css  = f"sig-{overall.lower()}"
                    st.markdown(f"**Overall:** <span class='{css}'>{icon} {overall}</span>",
                                unsafe_allow_html=True)
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Buy",  res["buy_count"])
                    b2.metric("Sell", res["sell_count"])
                    b3.metric("Hold", res["hold_count"])
                    if tick:
                        st.metric("Bid / Ask", f"{tick['bid']:.5f} / {tick['ask']:.5f}",
                                  delta=f"Spread {tick['spread']:.1f} pips")

                with col_r:
                    st.markdown("**Strategy breakdown:**")
                    for name, data in res["individual"].items():
                        ic = "🟢" if data["signal"] == "Buy" else "🔴" if data["signal"] == "Sell" else "🟡"
                        st.markdown(f"{ic} **{name}** ({data['signal']}, {data['strength']:.2f}): _{data['reason']}_")

                if st.button("🤖 AI Commentary", type="primary"):
                    with st.spinner("Claude is analysing..."):
                        try:
                            from ai_analyzer import get_market_commentary
                            st.markdown("---")
                            st.markdown(get_market_commentary(chart_sym, res, chart_tf))
                        except Exception as e:
                            st.error(f"AI error: {e}")
        except Exception as e:
            st.error(f"Chart failed: {e}")
            import traceback; st.code(traceback.format_exc())

# ═══════════════════════════ TRADE SETUP ══════════════════════════════════════
with tab_trade:
    st.markdown("### 🎯 AI Trade Setup")
    if not st.session_state.mt5_connected:
        st.warning("Connect to MT5 first.")
    else:
        left, right = st.columns([1, 2])

        with left:
            t_sym = st.selectbox("Symbol",    watch_list or ["EURUSD"], key="t_sym")
            t_tf  = st.selectbox("Timeframe", ["M15","M30","H1","H4","D1"], index=2, key="t_tf")
            c_sl  = st.number_input("Custom SL (pips)", 10, 500, 50)
            c_tp  = st.number_input("Custom TP (pips)", 10, 1000, 100)

            run_analysis = st.button("🔍 Run AI Analysis", type="primary", use_container_width=True)

        if run_analysis:
            with st.spinner("Running 7 strategies + Claude AI analysis..."):
                try:
                    from mt5_connector import get_ohlcv, get_account_info, get_open_positions, get_current_price
                    from strategies import calculate_all_indicators, run_all_strategies
                    from ai_analyzer import analyze_trade_opportunity
                    from risk_manager import assess_risk

                    df   = get_ohlcv(t_sym, t_tf, 400)
                    df   = calculate_all_indicators(df)
                    res  = run_all_strategies(df)
                    acc  = get_account_info()
                    pos  = get_open_positions()
                    risk = assess_risk(acc, pos, max_pos, max_dd)
                    ai   = analyze_trade_opportunity(t_sym, t_tf, res, acc, {"open_positions_count": len(pos)})

                    st.session_state.last_analysis = {
                        "symbol": t_sym, "timeframe": t_tf, "results": res,
                        "ai": ai, "account": acc, "risk": risk,
                        "custom_sl": c_sl, "custom_tp": c_tp,
                    }
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback; st.code(traceback.format_exc())

        with right:
            an = st.session_state.last_analysis
            if not an:
                st.info("Click **Run AI Analysis** to get a trade signal.")
            else:
                ai   = an["ai"]
                risk = an["risk"]
                action     = ai.get("action", "Hold")
                confidence = ai.get("confidence", 0)

                if action == "Buy":
                    st.success(f"## 🟢 BUY — {confidence*100:.0f}% confidence")
                elif action == "Sell":
                    st.error(f"## 🔴 SELL — {confidence*100:.0f}% confidence")
                else:
                    st.warning(f"## 🟡 HOLD — {confidence*100:.0f}% confidence")

                st.markdown(f"**AI Reasoning:** {ai.get('reasoning','N/A')}")

                if ai.get("key_factors"):
                    st.markdown("**Key factors:** " + " · ".join(ai["key_factors"]))

                for w in ai.get("warnings", []):
                    st.warning(f"⚠️ {w}")

                st.divider()
                d1, d2, d3 = st.columns(3)
                sl_p = ai.get("stop_loss_pips",  an["custom_sl"])
                tp_p = ai.get("take_profit_pips", an["custom_tp"])
                rr   = ai.get("risk_reward", round(tp_p / max(sl_p, 1), 2))
                d1.metric("SL (pips)", sl_p)
                d2.metric("TP (pips)", tp_p)
                d3.metric("R:R",       f"1:{rr:.1f}")
                d1.metric("Session",   ai.get("session_quality","—").title())
                d2.metric("Risk Level", risk["risk_level"])
                d3.metric("Drawdown",  f"{risk['current_drawdown']:.1f}%")

                if not risk["can_trade"]:
                    st.error("🛑 TRADING BLOCKED — " + " | ".join(risk["warnings"]))
                elif risk["warnings"]:
                    for w in risk["warnings"]:
                        st.warning(f"⚠️ {w}")

                if action != "Hold" and risk["can_trade"]:
                    st.divider()
                    from mt5_connector import get_current_price, place_order
                    from risk_manager import calculate_position_size, calculate_sl_tp

                    tick   = get_current_price(an["symbol"])
                    entry  = tick["ask"] if action == "Buy" else tick["bid"]
                    sl_pr, tp_pr = calculate_sl_tp(entry, action, sl_p, tp_p, an["symbol"])

                    # Use MT5 pip value for accurate sizing
                    try:
                        from mt5_connector import get_pip_value
                        pv = get_pip_value(an["symbol"])
                    except Exception:
                        pv = 10.0

                    lots = calculate_position_size(an["account"]["balance"], risk_pct, sl_p, pv)

                    e1, e2 = st.columns(2)
                    e1.markdown(f"**Entry:** `{entry:.5f}`")
                    e1.markdown(f"**Stop Loss:** `{sl_pr:.5f}`")
                    e1.markdown(f"**Take Profit:** `{tp_pr:.5f}`")
                    e1.markdown(f"**Lot Size:** `{lots}`")
                    risk_amt = an["account"]["balance"] * (risk_pct / 100)
                    e2.markdown(f"**Risk amount:** `${risk_amt:.2f}`")
                    e2.markdown(f"**Potential profit:** `${risk_amt * rr:.2f}`")

                    if st.button(f"🚀 EXECUTE {action.upper()}", type="primary", use_container_width=True):
                        ticket, msg = place_order(an["symbol"], action, lots, sl_pr, tp_pr, "AI Trade")
                        if ticket:
                            st.success(f"✅ Trade placed — Ticket #{ticket}")
                            st.session_state.trade_log.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "symbol": an["symbol"], "action": action,
                                "lots": lots, "entry": entry, "sl": sl_pr,
                                "tp": tp_pr, "ticket": ticket, "auto": False,
                            })
                        else:
                            st.error(f"❌ {msg}")

# ═══════════════════════════ AUTO SCAN ════════════════════════════════════════
if st.session_state.mt5_connected and st.session_state.auto_trading:
    with st.sidebar:
        st.divider()
        st.markdown("### 🤖 Auto Scan")
        if st.button("▶ Run Auto Scan Now", use_container_width=True):
            with st.spinner("Scanning..."):
                try:
                    from mt5_connector import get_ohlcv, get_account_info, get_open_positions, get_current_price, place_order
                    from strategies import calculate_all_indicators, run_all_strategies
                    from ai_analyzer import analyze_trade_opportunity
                    from risk_manager import calculate_position_size, calculate_sl_tp, assess_risk, get_pip_value
                    from sessions import get_session_quality

                    quality, _ = get_session_quality()
                    if quality == "Poor":
                        st.warning("Poor session — scan skipped")
                    else:
                        acc  = get_account_info()
                        pos  = get_open_positions()
                        risk = assess_risk(acc, pos, max_pos, max_dd)
                        if not risk["can_trade"]:
                            st.warning(f"Trading blocked: {risk['warnings'][0]}")
                        else:
                            found = []
                            for sym in (watch_list or [])[:4]:
                                df = get_ohlcv(sym, timeframe, 300)
                                if df is None or len(df) < 60:
                                    continue
                                df  = calculate_all_indicators(df)
                                res = run_all_strategies(df)
                                if res["overall"] == "Hold" or res["overall_strength"] < 0.45:
                                    continue
                                ai = analyze_trade_opportunity(sym, timeframe, res, acc, {"open_positions_count": len(pos)})
                                if ai.get("action") == "Hold" or ai.get("confidence", 0) < ai_thresh:
                                    continue
                                action = ai["action"]
                                sl_p   = ai.get("stop_loss_pips", 50)
                                tp_p   = ai.get("take_profit_pips", 100)
                                tick   = get_current_price(sym)
                                entry  = tick["ask"] if action == "Buy" else tick["bid"]
                                sl_pr, tp_pr = calculate_sl_tp(entry, action, sl_p, tp_p, sym)
                                try:
                                    pv = get_pip_value(sym)
                                except Exception:
                                    pv = 10.0
                                lots   = calculate_position_size(acc["balance"], risk_pct, sl_p, pv)
                                ticket, msg = place_order(sym, action, lots, sl_pr, tp_pr, "AutoAI")
                                if ticket:
                                    found.append(f"✅ {sym} {action} #{ticket}")
                                    st.session_state.trade_log.append({
                                        "time": datetime.now().strftime("%H:%M:%S"),
                                        "symbol": sym, "action": action,
                                        "lots": lots, "ticket": ticket, "auto": True,
                                    })
                                else:
                                    found.append(f"❌ {sym} {action}: {msg}")
                            if found:
                                for f in found:
                                    st.write(f)
                            else:
                                st.info(f"No signals above {ai_thresh*100:.0f}% confidence")
                except Exception as e:
                    st.error(f"Auto scan error: {e}")

# ═══════════════════════════ POSITIONS ════════════════════════════════════════
with tab_pos:
    st.markdown("### 📋 Open Positions")
    if not st.session_state.mt5_connected:
        st.warning("Connect to MT5 first.")
    else:
        from mt5_connector import get_open_positions, close_position

        pos = get_open_positions()
        if not pos:
            st.info("No open positions.")
        else:
            total_pnl = sum(p["profit"] for p in pos)
            p1, p2, p3 = st.columns(3)
            p1.metric("Open Positions", len(pos))
            p2.metric("Total P/L",      f"${total_pnl:+,.2f}")
            p3.metric("Avg P/L",        f"${total_pnl/len(pos):+,.2f}")

            for p in pos:
                icon = "🟢" if p["profit"] >= 0 else "🔴"
                with st.expander(f"{icon} {p['symbol']} {p['type']} | {p['volume']} lots | P/L: ${p['profit']:+.2f}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Open Price",    f"{p['open_price']:.5f}")
                    c2.metric("Current Price", f"{p['current_price']:.5f}")
                    c3.metric("Profit",        f"${p['profit']:+.2f}")
                    c1.metric("Stop Loss",     f"{p['sl']:.5f}" if p["sl"] else "None")
                    c2.metric("Take Profit",   f"{p['tp']:.5f}" if p["tp"] else "None")
                    c3.metric("Swap",          f"${p['swap']:.2f}")
                    st.caption(f"Ticket: {p['ticket']} | Opened: {p['open_time']}")

                    if st.button(f"❌ Close #{p['ticket']}", key=f"cl_{p['ticket']}"):
                        ok, msg = close_position(p["ticket"])
                        if ok:
                            st.success("Position closed!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(msg)

# ═══════════════════════════ HISTORY ══════════════════════════════════════════
with tab_hist:
    st.markdown("### 📜 Trade History & Performance")
    if not st.session_state.mt5_connected:
        st.warning("Connect to MT5 first.")
    else:
        from mt5_connector import get_trade_history

        days    = st.slider("History (days)", 1, 90, 30)
        history = get_trade_history(days)

        if not history:
            st.info("No closed trades in this period.")
        else:
            df_h = pd.DataFrame(history)
            wins = df_h[df_h["profit"] > 0]
            losses = df_h[df_h["profit"] <= 0]
            total_p = df_h["profit"].sum()
            win_rate = len(wins) / len(df_h) * 100

            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("Total P/L",   f"${total_p:+,.2f}")
            h2.metric("Win Rate",    f"{win_rate:.1f}%")
            h3.metric("Trades",      len(df_h))
            h4.metric("Avg Win",     f"${wins['profit'].mean():.2f}"   if len(wins) else "—")
            h5.metric("Avg Loss",    f"${losses['profit'].mean():.2f}" if len(losses) else "—")

            df_sorted = df_h.sort_values("time")
            df_sorted["Equity Curve"] = df_sorted["profit"].cumsum()

            fig_eq = go.Figure(go.Scatter(
                x=df_sorted["time"], y=df_sorted["Equity Curve"],
                fill="tozeroy", line=dict(color="#00D4FF"),
                fillcolor="rgba(0,212,255,0.1)", name="Cumulative P/L",
            ))
            fig_eq.update_layout(title="Equity Curve", template="plotly_dark", height=280, margin=dict(t=40))
            st.plotly_chart(fig_eq, use_container_width=True)

            ch1, ch2 = st.columns(2)
            with ch1:
                by_sym = df_h.groupby("symbol")["profit"].sum().reset_index()
                fig_sym = px.bar(by_sym, x="symbol", y="profit", title="P/L by Symbol",
                                  template="plotly_dark",
                                  color="profit", color_continuous_scale=["#FF4444","#888","#00FF88"])
                st.plotly_chart(fig_sym, use_container_width=True)
            with ch2:
                fig_pie = px.pie(
                    pd.DataFrame({"Result": ["Wins","Losses"], "Count": [len(wins), len(losses)]}),
                    values="Count", names="Result", title="Win / Loss",
                    template="plotly_dark",
                    color_discrete_map={"Wins": "#00FF88", "Losses": "#FF4444"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(
                df_h[["time","symbol","type","volume","price","profit"]],
                use_container_width=True, hide_index=True,
            )

# ═══════════════════════════ DATA & KNOWLEDGE BASE ═══════════════════════════
with tab_data:
    st.markdown("### 📥 Data Download & Knowledge Base")
    st.markdown("""
Download **5 years of daily** + **2 years of hourly** OHLCV data for all 8 major pairs from Yahoo Finance (free, no API key).
The **Knowledge Base** then analyses this data and gives Claude AI deep pair-specific context:
- Average Daily Range per pair
- Day-of-week and monthly volatility profiles
- 5-year trend analysis
- Key support/resistance levels from swing-point clusters
- Cross-pair correlation matrix
    """)

    from data_manager import is_data_available, get_download_status, PAIRS as DMPAIRS

    # Status table
    status = get_download_status()
    rows_s = []
    for pair in DMPAIRS:
        d_key = f"{pair}_1d"
        h_key = f"{pair}_1h"
        rows_s.append({
            "Pair": pair,
            "Daily (5yr)":  "✅ Ready" if status.get(d_key,{}).get("exists") else "❌ Not downloaded",
            "Hourly (2yr)": "✅ Ready" if status.get(h_key,{}).get("exists") else "❌ Not downloaded",
            "Daily size":   f"{status.get(d_key,{}).get('size_kb',0)} KB" if status.get(d_key,{}).get("exists") else "—",
        })
    st.dataframe(pd.DataFrame(rows_s), use_container_width=True, hide_index=True)

    from knowledge_base import KB_FILE
    kb_exists = os.path.exists(KB_FILE)

    col_dl, col_kb = st.columns(2)
    with col_dl:
        force = st.checkbox("Force re-download (overwrite cache)")
        if st.button("⬇️ Download All Data", type="primary", use_container_width=True):
            from data_manager import download_all
            prog = st.progress(0, "Starting download...")
            def _dl_cb(pct, msg):
                prog.progress(min(pct, 1.0), msg)
            try:
                download_all(force_refresh=force, progress_callback=_dl_cb)
                prog.progress(1.0, "Download complete!")
                st.success("✅ All data downloaded and cached to data/ folder.")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")

    with col_kb:
        kb_label = "✅ KB exists — rebuild?" if kb_exists else "❌ KB not built yet"
        st.info(kb_label)
        if st.button("🧠 Build Knowledge Base", type="primary", use_container_width=True):
            if not is_data_available():
                st.error("Download data first!")
            else:
                from knowledge_base import build_knowledge_base
                prog2 = st.progress(0, "Analysing data...")
                def _kb_cb(pct, msg):
                    prog2.progress(min(pct, 1.0), msg)
                try:
                    kb = build_knowledge_base(progress_callback=_kb_cb)
                    prog2.progress(1.0, "Done!")
                    st.success(f"✅ Knowledge base built for {len([k for k in kb if k != 'correlations'])} pairs.")
                    # Reload into ai_analyzer
                    import ai_analyzer
                    from knowledge_base import load_knowledge_base
                    ai_analyzer._KB = load_knowledge_base()
                    st.rerun()
                except Exception as e:
                    st.error(f"Knowledge base build failed: {e}")

    if kb_exists:
        st.divider()
        st.markdown("### Knowledge Base Preview")
        from knowledge_base import load_knowledge_base, format_for_ai
        kb = load_knowledge_base()
        preview_pair = st.selectbox("View pair", [p for p in kb if p != "correlations"])
        if preview_pair:
            st.code(format_for_ai(preview_pair, kb), language="text")

        # ── COT Signals Section ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🏦 COT Institutional Positioning (CFTC)")
        st.caption("Managed Money net position from weekly CFTC Disaggregated Futures report. Verified signal source (research vote 3-0 for library, 2-1 for signal).")

        from cot_feed import load_cot_signals
        cot_sigs = load_cot_signals()
        cot_built = bool(cot_sigs and any("direction" in v for v in cot_sigs.values() if isinstance(v, dict)))

        cot_c1, cot_c2 = st.columns(2)
        with cot_c1:
            st.info("✅ COT data built" if cot_built else "❌ COT not built yet")
            force_cot = st.checkbox("Force re-download COT data")
            if st.button("📊 Build COT Signals", type="primary" if not cot_built else "secondary",
                          use_container_width=True):
                from cot_feed import build_cot_signals
                prog_cot = st.progress(0, "Downloading CFTC data...")
                def _cot_cb(pct, msg): prog_cot.progress(min(pct,1.0), msg)
                try:
                    if force_cot:
                        import glob, os
                        for f in glob.glob(os.path.join("data","cot","disagg_*.csv")):
                            os.remove(f)
                    result_cot = build_cot_signals(progress_callback=_cot_cb)
                    if "error" not in result_cot:
                        st.success(f"✅ COT signals built for {len([k for k in result_cot if k!='_updated'])} pairs")
                    else:
                        st.error(result_cot["error"])
                except Exception as e:
                    st.error(f"COT build failed: {e}")
                st.rerun()

        with cot_c2:
            if cot_built:
                from data_manager import PAIRS as DMPAIRS
                cot_rows = []
                for p in DMPAIRS:
                    sig = cot_sigs.get(p, {})
                    if "direction" in sig:
                        cot_rows.append({
                            "Pair":       p,
                            "Bias":       sig["direction"],
                            "Net MM":     f"{sig['net_mm_current']:+,}",
                            "Z-Score":    sig["z_score"],
                            "As Of":      sig.get("as_of_date","?"),
                            "Warning":    sig.get("extreme_warning",""),
                        })
                if cot_rows:
                    df_cot = pd.DataFrame(cot_rows)
                    def _cot_color(val):
                        if "Bullish" in str(val): return "color:#00FF88; font-weight:700"
                        if "Bearish" in str(val): return "color:#FF4444; font-weight:700"
                        return ""
                    st.dataframe(
                        df_cot.style.applymap(_cot_color, subset=["Bias"]),
                        use_container_width=True, hide_index=True,
                    )

        # Correlation heatmap
        if "correlations" in kb:
            st.markdown("### Correlation Matrix")
            corr_pairs = list(kb["correlations"].keys())
            corr_matrix = pd.DataFrame(
                [[kb["correlations"].get(r, {}).get(c, 1.0 if r==c else 0.0) for c in corr_pairs]
                 for r in corr_pairs],
                index=corr_pairs, columns=corr_pairs,
            )
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f",
                color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                title="5-Year Return Correlation (Daily)", template="plotly_dark",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════ BACKTESTER ═══════════════════════════════════════
with tab_bt:
    st.markdown("### 🧪 Strategy Backtester — 5 Years of Real Data")

    if not is_data_available():
        st.warning("No data yet. Go to **Data & KB** tab and download data first.")
    else:
        from data_manager import PAIRS as DMPAIRS

        bc1, bc2, bc3, bc4 = st.columns(4)
        bt_pair     = bc1.selectbox("Pair",      list(DMPAIRS.keys()),        key="bt_pair")
        bt_interval = bc2.selectbox("Timeframe", ["1d", "1h"],                key="bt_tf")
        bt_balance  = bc3.number_input("Starting balance ($)", 1000, 100000, 10000, 1000)
        bt_risk     = bc4.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.25)

        bc5, bc6, bc7 = st.columns(3)
        bt_minsig   = bc5.slider("Min signals (out of 7)", 2, 6, 4)
        bt_sl_atr   = bc6.slider("SL (× ATR)", 0.5, 4.0, 1.5, 0.25)
        bt_tp_atr   = bc7.slider("TP (× ATR)", 1.0, 6.0, 2.5, 0.25)

        use_wfo = st.checkbox(
            "✅ Use Walk-Forward Optimization (WFO)",
            value=False,
            help="Research-verified (3-0 vote): rolls IS/OOS windows, optimizes params on each IS period, validates on unseen OOS. Reduces overfitting vs single split."
        )
        if use_wfo:
            wfo_is  = st.slider("In-sample window (bars)", 63, 504, 252, 63)
            wfo_oos = st.slider("Out-of-sample window (bars)", 21, 126, 63, 21)

        col_bt1, col_bt2 = st.columns(2)
        run_single  = col_bt1.button("▶ Backtest This Pair",    type="primary",   use_container_width=True)
        run_all_bt  = col_bt2.button("▶ Backtest All 8 Pairs",  use_container_width=True)

        if run_single:
            prog_bt = st.progress(0, f"{'WFO' if use_wfo else 'Backtesting'} {bt_pair} {bt_interval}...")
            def _bt_cb(pct, msg=""): prog_bt.progress(min(float(pct), 1.0), msg or "Running...")
            if use_wfo:
                from backtester import run_wfo
                result = run_wfo(bt_pair, bt_interval, bt_balance, bt_risk,
                                  wfo_is, wfo_oos, progress_callback=_bt_cb)
            else:
                from backtester import run_backtest
                result = run_backtest(bt_pair, bt_interval, bt_balance, bt_risk,
                                       bt_sl_atr, bt_tp_atr, bt_minsig,
                                       progress_callback=lambda p: _bt_cb(p))
            prog_bt.progress(1.0, "Done!")

            if "error" in result:
                st.error(result["error"])
            else:
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                ret_color = "normal" if result["total_return_pct"] >= 0 else "inverse"
                m1.metric("Total Return",     f"{result['total_return_pct']:+.1f}%")
                m2.metric("Final Balance",    f"${result['final_balance']:,.0f}")
                m3.metric("Win Rate",         f"{result['win_rate']:.1f}%")
                m4.metric("Profit Factor",    f"{result['profit_factor']:.2f}")
                m5.metric("Max Drawdown",     f"{result['max_drawdown_pct']:.1f}%")
                m6.metric("Sharpe Ratio",     f"{result['sharpe_ratio']:.2f}")

                m1b, m2b, m3b, m4b = st.columns(4)
                m1b.metric("Total Trades",    result["total_trades"])
                m2b.metric("Avg Win",         f"${result['avg_win_usd']:.2f}")
                m3b.metric("Avg Loss",        f"${result['avg_loss_usd']:.2f}")
                m4b.metric("Avg Hold (bars)", result["avg_bars_held"])

                # Equity curve
                fig_bt = go.Figure()
                eq = result["equity_curve"]
                idx = list(range(len(eq)))
                fig_bt.add_trace(go.Scatter(
                    x=idx, y=eq, name="Equity",
                    fill="tozeroy", line=dict(color="#00D4FF", width=1.5),
                    fillcolor="rgba(0,212,255,0.08)",
                ))
                fig_bt.add_hline(y=bt_balance, line=dict(color="#888", dash="dash", width=1))
                fig_bt.update_layout(
                    title=f"{bt_pair} {bt_interval} Equity Curve",
                    template="plotly_dark", height=320,
                    xaxis_title="Bar", yaxis_title="Balance ($)",
                    margin=dict(t=40),
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                # Trade log
                if result["trades"]:
                    df_bt = pd.DataFrame(result["trades"])
                    st.markdown(f"**{len(df_bt)} trades** | {result['max_consecutive_wins']} max consecutive wins | {result['max_consecutive_losses']} max consecutive losses")

                    def _color_pnl(val):
                        try:
                            return "color:#00FF88" if float(val) > 0 else "color:#FF4444"
                        except Exception:
                            return ""

                    cols_show = ["entry_date","exit_date","direction","entry","exit","pnl_pips","pnl_usd","reason","bars_held"]
                    cols_show = [c for c in cols_show if c in df_bt.columns]
                    st.dataframe(
                        df_bt[cols_show].style.applymap(_color_pnl, subset=["pnl_usd"]),
                        use_container_width=True, hide_index=True,
                    )

        if run_all_bt:
            from backtester import run_all_pairs_summary
            prog_all = st.progress(0, "Backtesting all pairs...")
            def _all_cb(pct, msg): prog_all.progress(min(pct, 1.0), msg)
            summary = run_all_pairs_summary(
                list(DMPAIRS.keys()), bt_interval, bt_balance, bt_risk, bt_minsig, _all_cb
            )
            prog_all.progress(1.0, "Done!")
            if not summary.empty:
                def _color_ret(val):
                    try:
                        return "background:#0d2b0d;color:#00FF88" if float(val) > 0 else "background:#2b0d0d;color:#FF4444"
                    except Exception:
                        return ""
                st.dataframe(
                    summary.style.applymap(_color_ret, subset=["Return %"]),
                    use_container_width=True, hide_index=True,
                )

# ═══════════════════════════ PAPER TRADING ════════════════════════════════════
with tab_paper:
    st.markdown("### 📝 Paper Trading — Test Without MT5")
    st.caption("Simulates live trades using real Yahoo Finance prices. No MT5 connection needed.")

    if "paper_balance" not in st.session_state:
        st.session_state.paper_balance  = 10_000.0
        st.session_state.paper_trades   = []
        st.session_state.paper_positions= []

    pb1, pb2, pb3 = st.columns(3)
    pb1.metric("Paper Balance",   f"${st.session_state.paper_balance:,.2f}")

    open_pnl = sum(p.get("unrealized_pnl", 0) for p in st.session_state.paper_positions)
    pb2.metric("Open P/L",        f"${open_pnl:+,.2f}")
    pb3.metric("Open Positions",  len(st.session_state.paper_positions))

    pap_reset = st.button("🔄 Reset Paper Account ($10,000)")
    if pap_reset:
        st.session_state.paper_balance   = 10_000.0
        st.session_state.paper_trades    = []
        st.session_state.paper_positions = []
        st.rerun()

    st.divider()

    # Scan and signal
    pap_sym = st.selectbox("Symbol", list(DMPAIRS.keys()), key="pap_sym")
    pap_tf  = st.selectbox("Timeframe", ["1d","1h"], key="pap_tf", index=1)
    pap_risk= st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, key="pap_risk")

    if st.button("🔍 Get AI Signal (Paper)", type="primary"):
        with st.spinner("Fetching live data + AI analysis..."):
            try:
                import yfinance as yf
                from data_manager import PAIRS as DMPAIRS2
                ticker = DMPAIRS2[pap_sym]
                period_map = {"1d": "6mo", "1h": "60d"}
                raw = yf.download(ticker, period=period_map[pap_tf], interval=pap_tf,
                                   auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw[["Open","High","Low","Close","Volume"]].dropna()

                from strategies import calculate_all_indicators, run_all_strategies
                raw = calculate_all_indicators(raw)
                res = run_all_strategies(raw)

                last_price = float(raw["Close"].iloc[-1])
                atr        = float(raw["ATR"].iloc[-1])
                action     = res["overall"]

                st.session_state.paper_signal = {
                    "symbol": pap_sym, "action": action,
                    "price": last_price, "atr": atr,
                    "results": res, "raw": raw,
                }

                if action == "Buy":
                    st.success(f"🟢 Signal: BUY {pap_sym} @ {last_price:.5f}")
                elif action == "Sell":
                    st.error(f"🔴 Signal: SELL {pap_sym} @ {last_price:.5f}")
                else:
                    st.warning(f"🟡 Signal: HOLD — no trade")

                st.markdown("**Strategy breakdown:**")
                for name, d in res["individual"].items():
                    ic = "🟢" if d["signal"]=="Buy" else "🔴" if d["signal"]=="Sell" else "🟡"
                    st.markdown(f"{ic} {name}: {d['signal']} — {d['reason']}")

            except Exception as e:
                st.error(f"Error: {e}")

    sig = st.session_state.get("paper_signal")
    if sig and sig["action"] != "Hold":
        st.divider()
        from risk_manager import calculate_sl_tp

        p_action = sig["action"]
        p_price  = sig["price"]
        p_atr    = sig["atr"]
        sl_dist  = p_atr * 1.5
        tp_dist  = p_atr * 2.5
        sl_p, tp_p = calculate_sl_tp(p_price, p_action, sl_dist * 10000, tp_dist * 10000, sig["symbol"])
        risk_usd = st.session_state.paper_balance * (pap_risk / 100)

        c1, c2 = st.columns(2)
        c1.markdown(f"**Entry:** {p_price:.5f}")
        c1.markdown(f"**Stop Loss:** {sl_p:.5f}")
        c1.markdown(f"**Take Profit:** {tp_p:.5f}")
        c2.markdown(f"**Risk:** ${risk_usd:.2f}")
        c2.markdown(f"**Potential R:R:** 1:1.7")

        if st.button(f"✅ Execute Paper {p_action.upper()}", type="primary"):
            st.session_state.paper_positions.append({
                "symbol": sig["symbol"], "action": p_action,
                "entry": p_price, "sl": sl_p, "tp": tp_p,
                "risk_usd": risk_usd, "unrealized_pnl": 0.0,
                "opened": datetime.now().strftime("%H:%M:%S"),
            })
            st.success(f"Paper {p_action} opened on {sig['symbol']} @ {p_price:.5f}")
            st.session_state.paper_signal = None
            st.rerun()

    # Open paper positions
    if st.session_state.paper_positions:
        st.divider()
        st.markdown("**Open paper positions:**")
        for i, pos in enumerate(st.session_state.paper_positions):
            try:
                import yfinance as yf
                from data_manager import PAIRS as DMPAIRS3
                tick_data = yf.download(DMPAIRS3[pos["symbol"]], period="1d",
                                         interval="5m", auto_adjust=True, progress=False)
                if not tick_data.empty:
                    if isinstance(tick_data.columns, pd.MultiIndex):
                        tick_data.columns = tick_data.columns.get_level_values(0)
                    curr = float(tick_data["Close"].iloc[-1])
                    pip = 0.01 if "JPY" in pos["symbol"] else (0.1 if "XAU" in pos["symbol"] else 0.0001)
                    pnl_pips = (curr - pos["entry"]) / pip * (1 if pos["action"]=="Buy" else -1)
                    pnl_usd  = pos["risk_usd"] * (pnl_pips / ((pos["entry"] - pos["sl"]) / pip))
                    pos["unrealized_pnl"] = round(pnl_usd, 2)
                else:
                    curr = pos["entry"]
                    pnl_usd = 0
            except Exception:
                curr = pos["entry"]; pnl_usd = 0

            icon = "🟢" if pnl_usd >= 0 else "🔴"
            with st.expander(f"{icon} {pos['symbol']} {pos['action']} | P/L: ${pnl_usd:+.2f}"):
                st.write(f"Entry: {pos['entry']:.5f} | SL: {pos['sl']:.5f} | TP: {pos['tp']:.5f}")
                st.write(f"Current: {curr:.5f} | Opened: {pos['opened']}")
                if st.button("❌ Close paper position", key=f"pap_cl_{i}"):
                    st.session_state.paper_balance += pnl_usd
                    st.session_state.paper_trades.append({
                        **pos, "exit": curr, "pnl_usd": pnl_usd,
                        "closed": datetime.now().strftime("%H:%M:%S"),
                    })
                    st.session_state.paper_positions.pop(i)
                    st.rerun()

    # Closed trades
    if st.session_state.paper_trades:
        st.divider()
        st.markdown("**Closed paper trades:**")
        df_pt = pd.DataFrame(st.session_state.paper_trades)
        if "pnl_usd" in df_pt.columns:
            total_pt = df_pt["pnl_usd"].sum()
            st.metric("Total Paper P/L", f"${total_pt:+.2f}")
            st.dataframe(df_pt[["symbol","action","entry","exit","pnl_usd"]],
                          use_container_width=True, hide_index=True)

# ═══════════════════════════ AI CHAT ══════════════════════════════════════════
with tab_chat:
    st.markdown("### 🤖 AI Trading Assistant")
    st.caption("Ask about strategies, session timing, risk management, pair characteristics — anything trading-related.")

    has_key = bool(st.session_state.mt5_creds.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))

    if not has_key:
        st.warning("Enter your Anthropic API key in the sidebar to enable AI chat.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask anything about trading..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from ai_analyzer import answer_trading_question
                        ctx = ""
                        if st.session_state.mt5_connected:
                            from mt5_connector import get_account_info
                            from sessions import get_session_quality
                            acc = get_account_info()
                            q, qdesc = get_session_quality()
                            ctx = f"Account balance: {acc['currency']} {acc['balance']:.2f}. Session: {qdesc}."
                        reply = answer_trading_question(prompt, ctx)
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        err = f"Error: {e}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})

        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

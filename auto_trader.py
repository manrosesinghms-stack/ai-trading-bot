"""
Autonomous Paper Trading Bot
Starts with $100, scans markets, places & closes trades automatically.
Records every trade with full details. Run until you stop it.

Run:  python -m streamlit run auto_trader.py
"""
from __future__ import annotations
import os, json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Load API key from Streamlit secrets (cloud) or .env (local) ───────────────
try:
    _key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if _key:
        os.environ["ANTHROPIC_API_KEY"] = _key
except Exception:
    pass
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Trader",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#07071a; }
[data-testid="stSidebar"]          { background:#0a0a1f; }
section[data-testid="stMain"] > div { padding-top:.4rem; }

.stat-box {
  background:#0d0d22; border:1px solid #1a1a3a;
  border-radius:12px; padding:14px 18px; text-align:center;
}
.stat-label { color:#555; font-size:.72rem; letter-spacing:.1em;
              text-transform:uppercase; margin-bottom:4px; }
.stat-val   { color:#eee; font-size:1.5rem; font-weight:800; }
.stat-up    { color:#00FF88; }
.stat-dn    { color:#FF4444; }
.stat-neu   { color:#FFAA00; }

.trade-row {
  display:grid;
  grid-template-columns:36px 90px 60px 70px 80px 80px 80px 80px 1fr;
  gap:0 8px; align-items:center;
  background:#0d0d22; border:1px solid #1a1a3a;
  border-radius:8px; padding:8px 12px; margin:3px 0;
  font-size:.82rem;
}
.trade-win  { border-left:3px solid #00FF88; }
.trade-loss { border-left:3px solid #FF4444; }
.trade-open { border-left:3px solid #FFAA00; }

.lbl        { color:#555; font-size:.7rem; }
.buy-txt    { color:#00FF88; font-weight:700; }
.sell-txt   { color:#FF4444; font-weight:700; }
.pnl-pos    { color:#00FF88; font-weight:700; }
.pnl-neg    { color:#FF4444; font-weight:700; }

.big-btn button {
  font-size:1.2rem !important; font-weight:800 !important;
  height:3rem !important; border-radius:12px !important;
}
.stop-btn button {
  background:#FF4444 !important; color:#fff !important;
}
.start-btn button {
  background:linear-gradient(90deg,#00CC66,#00AA55) !important;
  color:#fff !important;
}

@keyframes pulse { 0%{opacity:1}50%{opacity:.4}100%{opacity:1} }
.live-dot {
  display:inline-block; width:10px; height:10px;
  border-radius:50%; background:#00FF88;
  animation:pulse 1s infinite; margin-right:6px;
}
.idle-dot {
  display:inline-block; width:10px; height:10px;
  border-radius:50%; background:#444; margin-right:6px;
}
</style>
""", unsafe_allow_html=True)

# ── Persistence: save/load bot state to file ───────────────────────────────────
DATA_FILE = Path(__file__).parent / "data" / "auto_trader_state.json"
DATA_FILE.parent.mkdir(exist_ok=True)

def save_state():
    state = {
        "running":       st.session_state.bot_running,
        "balance":       st.session_state.bot_balance,
        "start_balance": st.session_state.bot_start_balance,
        "positions":     st.session_state.bot_positions,
        "trades":        st.session_state.bot_trades,
        "started_at":    st.session_state.bot_started_at,
        "scan_count":    st.session_state.bot_scan_count,
    }
    DATA_FILE.write_text(json.dumps(state, indent=2))

def load_state():
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return None

# ── Session state ──────────────────────────────────────────────────────────────
saved = load_state()
defaults = {
    "bot_running":       saved["running"]       if saved else False,
    "bot_balance":       saved["balance"]       if saved else 100.0,
    "bot_start_balance": saved["start_balance"] if saved else 100.0,
    "bot_positions":     saved["positions"]     if saved else [],
    "bot_trades":        saved["trades"]        if saved else [],
    "bot_started_at":    saved["started_at"]    if saved else None,
    "bot_scan_count":    saved["scan_count"]    if saved else 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto refresh ───────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    if st.session_state.bot_running:
        st_autorefresh(interval=60_000, key="bot_refresh")  # every 60s
except ImportError:
    pass

# ── Helpers ────────────────────────────────────────────────────────────────────
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","XAUUSD","AUDUSD","USDCAD"]
TFS     = ["M15","M30","H1","H4","D1"]
PIP_MAP = {"EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
           "USDCHF":.0001,"USDJPY":.01,"EURJPY":.01,"GBPJPY":.01,"XAUUSD":.1}
TV_MAP  = {"EURUSD":"FX:EURUSD","GBPUSD":"FX:GBPUSD","USDJPY":"FX:USDJPY",
           "AUDUSD":"FX:AUDUSD","USDCAD":"FX:USDCAD","GBPJPY":"FX:GBPJPY",
           "EURJPY":"FX:EURJPY","XAUUSD":"TVC:GOLD"}
TV_TF   = {"M15":"15","M30":"30","H1":"60","H4":"240","D1":"D"}

def pip(sym): return PIP_MAP.get(sym, .0001)

def fmt(p, sym):
    d = 3 if "JPY" in sym else (2 if "XAU" in sym else 5)
    return f"{p:.{d}f}"

def now_str(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def elapsed(started_at):
    if not started_at:
        return "—"
    try:
        start = datetime.fromisoformat(started_at)
        delta = datetime.now(timezone.utc) - start
        h, m = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    except Exception:
        return "—"

@st.cache_data(ttl=55)
def fetch_data(symbol, timeframe):
    import yfinance as yf
    from data_manager import PAIRS as YF_PAIRS
    from strategies import calculate_all_indicators, run_all_strategies

    ticker  = YF_PAIRS.get(symbol, symbol + "=X")
    per_map = {"M15":"5d","M30":"10d","H1":"30d","H4":"90d","D1":"2y"}
    iv_map  = {"M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}

    df = yf.download(ticker, period=per_map.get(timeframe,"30d"),
                     interval=iv_map.get(timeframe,"1h"),
                     auto_adjust=True, progress=False)
    if df.empty or len(df) < 30:
        return None, None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df = calculate_all_indicators(df)
    res = run_all_strategies(df)
    price = float(df["Close"].iloc[-1])
    atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else price * 0.001
    return df, res, price, atr

def get_current_price(symbol):
    try:
        import yfinance as yf
        from data_manager import PAIRS as YF_PAIRS
        ticker = YF_PAIRS.get(symbol, symbol+"=X")
        df = yf.download(ticker, period="1d", interval="1m",
                          auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None

# ── Core trading logic ─────────────────────────────────────────────────────────
def open_position(symbol, direction, entry, sl_price, tp_price, risk_usd, atr):
    pos = {
        "id":        len(st.session_state.bot_trades) + len(st.session_state.bot_positions) + 1,
        "symbol":    symbol,
        "direction": direction,
        "entry":     entry,
        "sl":        sl_price,
        "tp":        tp_price,
        "risk_usd":  risk_usd,
        "atr":       atr,
        "opened_at": now_str(),
        "pnl":       0.0,
    }
    st.session_state.bot_positions.append(pos)
    return pos

def close_position(pos, exit_price, reason):
    p         = pip(pos["symbol"])
    sl_dist   = abs(pos["entry"] - pos["sl"])
    sl_pips   = sl_dist / p
    exit_dist = exit_price - pos["entry"]
    if pos["direction"] == "Sell":
        exit_dist = -exit_dist
    pnl_pips  = exit_dist / p - 1.5        # subtract spread
    r_mult     = pnl_pips / max(sl_pips, 1)
    pnl        = pos["risk_usd"] * r_mult

    st.session_state.bot_balance += pnl

    trade = {
        **pos,
        "exit":       exit_price,
        "closed_at":  now_str(),
        "pnl":        round(pnl, 4),
        "pnl_pct":    round(pnl / pos["risk_usd"] * (pos["risk_usd"] / st.session_state.bot_start_balance) * 100, 2),
        "reason":     reason,
        "balance_after": round(st.session_state.bot_balance, 4),
        "return_pct": round((st.session_state.bot_balance / st.session_state.bot_start_balance - 1) * 100, 2),
    }
    st.session_state.bot_trades.insert(0, trade)
    return trade

def smart_filters(sym, timeframe, action, conf_pct, price, atr, smart_cfg):
    """
    Run advanced confluence filters on a candidate trade.
    Returns (approved: bool, reason: str, overrides: dict).
    overrides may contain ai_sl_pips / ai_tp_pips from Claude.
    """
    overrides = {}

    # ── 1. News blocking ──
    if smart_cfg.get("news"):
        try:
            from news_calendar import should_block_trade
            blocked, reason = should_block_trade(sym, 30)
            if blocked:
                return False, f"🚫 News block: {reason[:50]}", overrides
        except Exception:
            pass

    # ── 2. Multi-timeframe confluence ──
    if smart_cfg.get("mtf"):
        try:
            from mtf_analyzer import analyze_mtf
            mtf = analyze_mtf(sym, timeframe)
            htf = mtf.get("higher_tf_trend", "Hold")
            # Veto if higher timeframe strongly opposes the trade direction
            if action == "Buy" and htf == "Sell":
                return False, f"🚫 MTF veto: higher TF bearish (score {mtf['confluence_score']:+.2f})", overrides
            if action == "Sell" and htf == "Buy":
                return False, f"🚫 MTF veto: higher TF bullish (score {mtf['confluence_score']:+.2f})", overrides
        except Exception:
            pass

    # ── 3. COT institutional positioning ──
    if smart_cfg.get("cot"):
        try:
            from cot_feed import get_cot_signal
            cot = get_cot_signal(sym)
            direction = cot.get("direction", "")
            if direction:
                if action == "Buy" and "Bearish" in direction and cot.get("strength", 0) > 0.6:
                    return False, f"🚫 COT veto: institutions heavily {direction}", overrides
                if action == "Sell" and "Bullish" in direction and cot.get("strength", 0) > 0.6:
                    return False, f"🚫 COT veto: institutions heavily {direction}", overrides
        except Exception:
            pass

    # ── 4. Claude AI confirmation (final gate) ──
    if smart_cfg.get("ai"):
        try:
            import os
            if os.environ.get("ANTHROPIC_API_KEY"):
                from strategies import run_all_strategies, calculate_all_indicators
                from ai_analyzer import analyze_trade_opportunity
                # Re-fetch a strategy result dict for the prompt
                acc = {"balance": st.session_state.bot_balance, "equity": st.session_state.bot_balance,
                       "free_margin": st.session_state.bot_balance, "currency": "USD"}
                _, res2, _, _ = fetch_data(sym, timeframe)
                if res2:
                    ai = analyze_trade_opportunity(sym, timeframe, res2, acc, {"open_positions_count": len(st.session_state.bot_positions)})
                    ai_action = ai.get("action", "Hold")
                    ai_conf   = ai.get("confidence", 0)
                    # AI must agree with direction and have reasonable confidence
                    if ai_action != action:
                        return False, f"🚫 AI veto: Claude says {ai_action} (not {action})", overrides
                    if ai_conf < 0.55:
                        return False, f"🚫 AI veto: Claude confidence only {int(ai_conf*100)}%", overrides
                    # Use Claude's SL/TP if provided
                    if ai.get("stop_loss_pips"):
                        overrides["ai_sl_pips"] = ai["stop_loss_pips"]
                    if ai.get("take_profit_pips"):
                        overrides["ai_tp_pips"] = ai["take_profit_pips"]
                    overrides["ai_conf"] = int(ai_conf * 100)
        except Exception as e:
            # AI errors should not block trading — just note it
            overrides["ai_note"] = f"AI check skipped ({str(e)[:30]})"

    return True, "✅ All filters passed", overrides


def run_scan(symbols, timeframe, min_conf, risk_pct, max_positions, sl_mult, tp_mult, smart_cfg=None):
    """One scan cycle: check positions + look for new entries."""
    results = []
    current_prices = {}
    smart_cfg = smart_cfg or {}

    # ── 1. Fetch current prices for all open positions ──
    open_syms = set(p["symbol"] for p in st.session_state.bot_positions)
    for sym in open_syms:
        price = get_current_price(sym)
        if price:
            current_prices[sym] = price

    # ── 2. Check SL / TP on open positions ──
    closed_ids = []
    for pos in st.session_state.bot_positions:
        price = current_prices.get(pos["symbol"])
        if not price:
            continue
        sl, tp = pos["sl"], pos["tp"]
        closed = False
        if pos["direction"] == "Buy":
            if price <= sl:
                trade = close_position(pos, sl, "Stop Loss ❌")
                results.append(f"🔴 SL hit: {pos['symbol']} Buy → ${trade['pnl']:+.2f}")
                closed = True
            elif price >= tp:
                trade = close_position(pos, tp, "Take Profit ✅")
                results.append(f"🟢 TP hit: {pos['symbol']} Buy → ${trade['pnl']:+.2f}")
                closed = True
        else:
            if price >= sl:
                trade = close_position(pos, sl, "Stop Loss ❌")
                results.append(f"🔴 SL hit: {pos['symbol']} Sell → ${trade['pnl']:+.2f}")
                closed = True
            elif price <= tp:
                trade = close_position(pos, tp, "Take Profit ✅")
                results.append(f"🟢 TP hit: {pos['symbol']} Sell → ${trade['pnl']:+.2f}")
                closed = True
        if closed:
            closed_ids.append(pos["id"])

    st.session_state.bot_positions = [
        p for p in st.session_state.bot_positions if p["id"] not in closed_ids
    ]

    # ── 3. Look for new entries ──
    if len(st.session_state.bot_positions) >= max_positions:
        results.append(f"ℹ️ Max positions ({max_positions}) reached — skipping entry scan")
    else:
        already_open = {p["symbol"] for p in st.session_state.bot_positions}

        for sym in symbols:
            if sym in already_open:
                continue
            try:
                df, res, price, atr = fetch_data(sym, timeframe)
                if df is None:
                    continue
                current_prices[sym] = price

                action     = res["overall"]
                confidence = res["overall_strength"]
                conf_pct   = int(confidence * 100)

                if action == "Hold" or conf_pct < min_conf:
                    continue

                # ── Smart confluence filters (MTF, news, COT, Claude AI) ──
                approved, reason, overrides = smart_filters(
                    sym, timeframe, action, conf_pct, price, atr, smart_cfg
                )
                if not approved:
                    results.append(f"⊘ {sym} {action} {conf_pct}% — {reason}")
                    continue

                risk_usd = st.session_state.bot_balance * (risk_pct / 100)
                pip_sz   = pip(sym)

                # Use Claude's SL/TP if it provided them, else ATR-based
                if overrides.get("ai_sl_pips"):
                    sl_p = overrides["ai_sl_pips"] * pip_sz
                    tp_p = overrides.get("ai_tp_pips", overrides["ai_sl_pips"] * 2) * pip_sz
                else:
                    sl_p = atr * sl_mult
                    tp_p = atr * tp_mult

                if action == "Buy":
                    sl_price = round(price - sl_p, 5)
                    tp_price = round(price + tp_p, 5)
                else:
                    sl_price = round(price + sl_p, 5)
                    tp_price = round(price - tp_p, 5)

                pos = open_position(sym, action, price, sl_price, tp_price, risk_usd, atr)
                buy_c  = res["buy_count"]
                sell_c = res["sell_count"]
                ai_tag = f" | 🤖 AI {overrides['ai_conf']}%" if overrides.get("ai_conf") else ""
                results.append(
                    f"{'🟢' if action=='Buy' else '🔴'} {action} {sym} "
                    f"@ {fmt(price,sym)} | SL {fmt(sl_price,sym)} | TP {fmt(tp_price,sym)} "
                    f"| Conf {conf_pct}% ({buy_c}B/{sell_c}S){ai_tag}"
                )
            except Exception as e:
                results.append(f"⚠️ {sym}: {e}")

    # Update unrealized P&L on remaining positions
    for pos in st.session_state.bot_positions:
        price = current_prices.get(pos["symbol"])
        if not price:
            continue
        p       = pip(pos["symbol"])
        sl_pips = abs(pos["entry"] - pos["sl"]) / p
        exit_d  = (price - pos["entry"]) if pos["direction"]=="Buy" else (pos["entry"] - price)
        pnl_pips = exit_d / p - 1.5
        pos["pnl"] = round(pos["risk_usd"] * (pnl_pips / max(sl_pips,1)), 4)
        pos["current_price"] = price

    st.session_state.bot_scan_count += 1
    save_state()
    return results

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────── SIDEBAR ─────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Bot Settings")

    selected_symbols = st.multiselect(
        "Scan Symbols", SYMBOLS,
        default=["EURUSD","GBPUSD","XAUUSD","USDJPY","AUDUSD","USDCAD"],
    )
    timeframe    = st.selectbox("Timeframe", TFS, index=0)   # M15 = more signals
    min_conf_pct = st.slider("Min signal confidence %", 30, 90, 45, 5)
    risk_pct     = st.slider("Risk per trade (% of balance)", 1, 10, 2)
    max_positions= st.slider("Max open positions", 1, 12, 8)
    sl_mult      = st.slider("SL × ATR", 0.5, 3.0, 1.0, 0.25)   # tighter → closes faster
    tp_mult      = st.slider("TP × ATR", 1.0, 5.0, 1.5, 0.25)   # tighter → hits faster

    st.markdown("### 🧠 Smart Filters")
    st.caption("Extra confluence checks before every trade")
    f_mtf  = st.toggle("Multi-timeframe confluence", value=True,
                        help="Veto trades that fight the higher-timeframe trend")
    f_news = st.toggle("News blocking", value=True,
                        help="Skip trades within 30 min of high-impact news")
    f_cot  = st.toggle("COT institutional bias", value=True,
                        help="Veto trades against heavy institutional positioning (needs COT data built)")
    f_ai   = st.toggle("Claude AI confirmation", value=False,
                        help="Claude reviews & can veto each trade (~1.5¢ per check). Off by default for speed/volume.")
    smart_cfg = {"mtf": f_mtf, "news": f_news, "cot": f_cot, "ai": f_ai}

    st.divider()
    start_bal = st.number_input("Starting balance ($)", 10.0, 10000.0,
                                  value=st.session_state.bot_start_balance,
                                  step=10.0, format="%.2f")
    if st.button("🔄 Reset Everything", use_container_width=True):
        st.session_state.bot_running       = False
        st.session_state.bot_balance       = start_bal
        st.session_state.bot_start_balance = start_bal
        st.session_state.bot_positions     = []
        st.session_state.bot_trades        = []
        st.session_state.bot_started_at    = None
        st.session_state.bot_scan_count    = 0
        save_state()
        st.rerun()

    st.divider()
    st.markdown("### 📺 Chart")
    tv_sym_select = st.selectbox("Chart symbol", selected_symbols or SYMBOLS, key="chart_sym_sel")
    tv_tf_select  = st.selectbox("Chart TF", TFS, index=2, key="chart_tf_sel")

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────── MAIN UI ─────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# ── Title ──────────────────────────────────────────────────────────────────────
t1, t2 = st.columns([4, 1])
with t1:
    dot = '<span class="live-dot"></span>' if st.session_state.bot_running else '<span class="idle-dot"></span>'
    status_txt = f"{dot}<b style='color:#00FF88'>RUNNING</b>" if st.session_state.bot_running else f"{dot}<b style='color:#555'>STOPPED</b>"
    st.markdown(f"## 🤖 Auto Paper Trader &nbsp;&nbsp; {status_txt}", unsafe_allow_html=True)
with t2:
    st.markdown(f"<div style='text-align:right; color:#444; font-size:.8rem; padding-top:14px'>Scan #{st.session_state.bot_scan_count} &nbsp;|&nbsp; {elapsed(st.session_state.bot_started_at)}</div>", unsafe_allow_html=True)

# ── START / STOP ───────────────────────────────────────────────────────────────
b1, b2, b3 = st.columns([2, 2, 4])
with b1:
    if not st.session_state.bot_running:
        st.markdown('<div class="big-btn start-btn">', unsafe_allow_html=True)
        if st.button("▶  START BOT", use_container_width=True, type="primary"):
            st.session_state.bot_running    = True
            st.session_state.bot_started_at = datetime.now(timezone.utc).isoformat()
            save_state()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="big-btn stop-btn">', unsafe_allow_html=True)
        if st.button("⏹  STOP BOT", use_container_width=True):
            st.session_state.bot_running = False
            save_state()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

with b2:
    if st.session_state.bot_running:
        if st.button("🔍 Scan Now", use_container_width=True):
            with st.spinner("Scanning markets..."):
                scan_log = run_scan(
                    selected_symbols, timeframe, min_conf_pct,
                    risk_pct, max_positions, sl_mult, tp_mult, smart_cfg,
                )
            st.cache_data.clear()
            for msg in scan_log:
                st.toast(msg)
            st.rerun()

# ── Stats bar ──────────────────────────────────────────────────────────────────
balance   = st.session_state.bot_balance
start_bal = st.session_state.bot_start_balance
total_ret = (balance / start_bal - 1) * 100
open_pnl  = sum(p.get("pnl", 0) for p in st.session_state.bot_positions)
equity    = balance + open_pnl

trades    = st.session_state.bot_trades
wins      = [t for t in trades if t.get("pnl", 0) > 0]
losses    = [t for t in trades if t.get("pnl", 0) <= 0]
win_rate  = len(wins) / len(trades) * 100 if trades else 0
best      = max((t["pnl"] for t in trades), default=0)
worst     = min((t["pnl"] for t in trades), default=0)
total_pnl = sum(t["pnl"] for t in trades)
max_bal   = max([start_bal] + [t["balance_after"] for t in trades])
min_eq    = min([start_bal] + [t["balance_after"] for t in trades])
max_dd    = (max_bal - min_eq) / max_bal * 100

ret_color  = "#00FF88" if total_ret >= 0 else "#FF4444"
pnl_color  = "#00FF88" if total_pnl >= 0 else "#FF4444"
opnl_color = "#00FF88" if open_pnl  >= 0 else "#FF4444"

stats = [
    ("Starting", f"${start_bal:.2f}", ""),
    ("Balance", f"${balance:.2f}", ret_color),
    ("Equity", f"${equity:.2f}", opnl_color),
    ("Total Return", f"{total_ret:+.2f}%", ret_color),
    ("Closed P/L", f"${total_pnl:+.2f}", pnl_color),
    ("Open P/L", f"${open_pnl:+.2f}", opnl_color),
    ("Win Rate", f"{win_rate:.0f}%", "#00FF88" if win_rate>=50 else "#FF4444"),
    ("Trades", str(len(trades)), "#eee"),
    ("Best Trade", f"${best:+.2f}", "#00FF88"),
    ("Worst Trade", f"${worst:+.2f}", "#FF4444"),
    ("Max Drawdown", f"{max_dd:.1f}%", "#FF4444" if max_dd>10 else "#FFAA00"),
    ("Open Pos.", str(len(st.session_state.bot_positions)), "#eee"),
]

cols = st.columns(len(stats))
for col, (label, value, color) in zip(cols, stats):
    color = color or "#eee"
    col.markdown(f"""
<div class="stat-box">
  <div class="stat-label">{label}</div>
  <div class="stat-val" style="color:{color}">{value}</div>
</div>""", unsafe_allow_html=True)

st.markdown("")

# ── Auto-run scan on each refresh if bot is running ────────────────────────────
if st.session_state.bot_running and st.session_state.bot_scan_count > 0:
    # Run silently in background on each autorefresh
    try:
        run_scan(selected_symbols, timeframe, min_conf_pct,
                 risk_pct, max_positions, sl_mult, tp_mult, smart_cfg)
    except Exception:
        pass

# ── Main 2-column ──────────────────────────────────────────────────────────────
chart_col, info_col = st.columns([3, 2], gap="medium")

# ── TradingView Chart ──────────────────────────────────────────────────────────
with chart_col:
    tv_sym = TV_MAP.get(tv_sym_select, f"FX:{tv_sym_select}")
    tv_tf  = TV_TF.get(tv_tf_select, "60")
    tv_html = f"""
<div class="tradingview-widget-container" style="height:460px;border-radius:12px;overflow:hidden">
  <div id="tv_bot_chart" style="height:460px"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
  new TradingView.widget({{
    "container_id":       "tv_bot_chart",
    "width":              "100%",
    "height":             460,
    "symbol":             "{tv_sym}",
    "interval":           "{tv_tf}",
    "timezone":           "Etc/UTC",
    "theme":              "dark",
    "style":              "1",
    "locale":             "en",
    "toolbar_bg":         "#0d0d22",
    "hide_top_toolbar":   false,
    "allow_symbol_change":true,
    "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies"],
    "show_popup_button":  false
  }});
  </script>
</div>"""
    components.html(tv_html, height=470, scrolling=False)

# ── Open positions + recent trades ────────────────────────────────────────────
with info_col:
    # Open positions
    st.markdown("#### 📊 Open Positions")
    if not st.session_state.bot_positions:
        st.markdown("<div style='color:#444; font-size:.85rem; padding:10px'>No open positions.</div>", unsafe_allow_html=True)
    else:
        for i, pos in enumerate(st.session_state.bot_positions):
            pnl   = pos.get("pnl", 0)
            curr  = pos.get("current_price", pos["entry"])
            color = "#00FF88" if pnl >= 0 else "#FF4444"
            dir_c = "buy-txt" if pos["direction"]=="Buy" else "sell-txt"
            st.markdown(f"""
<div style="background:#0d0d22; border:1px solid #1a1a3a; border-left:3px solid {'#00FF88' if pnl>=0 else '#FF4444'};
            border-radius:8px; padding:10px 14px; margin:4px 0; font-size:.82rem;">
  <div style="display:flex; justify-content:space-between; margin-bottom:4px">
    <b style="color:#eee">{pos['symbol']}</b>
    <span class="{dir_c}">{'▲ Buy' if pos['direction']=='Buy' else '▼ Sell'}</span>
    <b style="color:{color}">${pnl:+.4f}</b>
  </div>
  <div style="color:#555; font-size:.75rem; display:flex; gap:12px">
    <span>Entry: {fmt(pos['entry'],pos['symbol'])}</span>
    <span>Now: {fmt(curr,pos['symbol'])}</span>
    <span>SL: {fmt(pos['sl'],pos['symbol'])}</span>
    <span>TP: {fmt(pos['tp'],pos['symbol'])}</span>
  </div>
  <div style="color:#444; font-size:.72rem; margin-top:2px">Opened: {pos['opened_at']}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"Close #{pos['id']}", key=f"man_close_{i}_{pos['id']}"):
                price = get_current_price(pos["symbol"]) or pos.get("current_price", pos["entry"])
                close_position(pos, price, "Manual Close")
                st.session_state.bot_positions = [p for p in st.session_state.bot_positions if p["id"] != pos["id"]]
                save_state()
                st.rerun()

    st.markdown("#### 📜 Trade History")
    if not st.session_state.bot_trades:
        st.markdown("<div style='color:#444; font-size:.85rem; padding:10px'>No closed trades yet.</div>", unsafe_allow_html=True)
    else:
        for t in st.session_state.bot_trades[:20]:
            pnl    = t.get("pnl", 0)
            color  = "#00FF88" if pnl >= 0 else "#FF4444"
            border = "#00FF88" if pnl >= 0 else "#FF4444"
            icon   = "✅" if pnl >= 0 else "❌"
            dir_c  = "buy-txt" if t["direction"]=="Buy" else "sell-txt"
            st.markdown(f"""
<div style="background:#0d0d22; border:1px solid #1a1a3a; border-left:3px solid {border};
            border-radius:8px; padding:9px 12px; margin:3px 0; font-size:.8rem;">
  <div style="display:flex; justify-content:space-between; align-items:center">
    <span style="color:#eee; font-weight:700">#{t['id']} {t['symbol']}</span>
    <span class="{dir_c}">{'▲' if t['direction']=='Buy' else '▼'} {t['direction']}</span>
    <span style="color:{color}; font-weight:800">{icon} ${pnl:+.4f}</span>
    <span style="color:{color}; font-size:.75rem">{t.get('return_pct',0):+.2f}% acct</span>
  </div>
  <div style="color:#555; font-size:.73rem; margin-top:3px; display:flex; gap:10px; flex-wrap:wrap">
    <span>In: {fmt(t['entry'],t['symbol'])}</span>
    <span>Out: {fmt(t.get('exit',0),t['symbol'])}</span>
    <span>{t.get('reason','—')}</span>
    <span>Bal after: ${t.get('balance_after',0):.2f}</span>
  </div>
  <div style="color:#333; font-size:.7rem">{t.get('opened_at','')[:19]} → {t.get('closed_at','')[:19]}</div>
</div>""", unsafe_allow_html=True)

# ── Equity curve ───────────────────────────────────────────────────────────────
if st.session_state.bot_trades:
    st.divider()
    st.markdown("#### 📈 Equity Curve")
    import plotly.graph_objects as go
    balances = [start_bal] + [t["balance_after"] for t in reversed(st.session_state.bot_trades)]
    labels   = ["Start"] + [f"#{t['id']}" for t in reversed(st.session_state.bot_trades)]
    colors   = ["#00FF88" if b >= start_bal else "#FF4444" for b in balances]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=balances, mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        marker=dict(color=colors, size=7),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        name="Balance",
    ))
    fig.add_hline(y=start_bal, line=dict(color="#555", dash="dash", width=1))
    fig.update_layout(
        height=220, template="plotly_dark", paper_bgcolor="#07071a",
        plot_bgcolor="#07071a", margin=dict(t=10,b=30,l=0,r=0),
        xaxis_title="Trade #", yaxis_title="Balance ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; color:#333; font-size:.72rem; margin-top:8px">
  Auto Paper Trader · {'🟢 Running' if st.session_state.bot_running else '⏸ Stopped'} ·
  Refreshes every 60s when running · {now_str()}
</div>
""", unsafe_allow_html=True)

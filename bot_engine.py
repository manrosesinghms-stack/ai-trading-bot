"""
Headless trading engine — runs ONE scan cycle and exits (for GitHub Actions cron),
or loops locally with --loop. Reads/writes data/bot_state.json so the dashboard
and the cron job share one source of truth.

Uses the backtest-optimized config: edge pairs only, per-pair SL/TP, H1 timeframe
(the timeframe the parameters were validated on), consensus-weighted confidence.

Run once:   python bot_engine.py            (one scan, used by cron)
Run loop:   python bot_engine.py --loop     (continuous, local use)
"""
from __future__ import annotations
import json, sys, time, logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("engine")

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
STATE_FILE  = DATA / "bot_state.json"
CONFIG_FILE = DATA / "bot_config.json"

# ── Optimized config (validated out-of-sample on 2yr H1 data) ──────────────────
PER_PAIR = {
    "XAUUSD": {"sl_atr": 1.5, "tp_atr": 1.0},
    "USDCAD": {"sl_atr": 2.0, "tp_atr": 1.0},
    "EURJPY": {"sl_atr": 1.0, "tp_atr": 1.0},
    "AUDUSD": {"sl_atr": 2.0, "tp_atr": 1.0},
}
DEFAULT_CONFIG = {
    "running":       True,
    "symbols":       list(PER_PAIR.keys()),
    "timeframe":     "H1",
    "min_conf_pct":  45,
    "risk_pct":      2,
    "max_positions": 8,
    "news_filter":   True,
}

PIP = {"EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
       "USDCHF":.0001,"USDJPY":.01,"EURJPY":.01,"GBPJPY":.01,"XAUUSD":.1}
def pip(s): return PIP.get(s, .0001)
def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_config():
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"balance": 100.0, "start_balance": 100.0, "positions": [],
            "trades": [], "scan_count": 0, "started_at": now(), "last_scan": None}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def fetch_price(sym):
    try:
        import yfinance as yf
        from data_manager import PAIRS
        df = yf.download(PAIRS.get(sym, sym+"=X"), period="1d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty: return None
        if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"price {sym}: {e}"); return None


def analyze(sym, tf):
    try:
        import yfinance as yf
        from data_manager import PAIRS
        from strategies import calculate_all_indicators, run_all_strategies
        per = {"M15":"5d","M30":"10d","H1":"30d","H4":"90d","D1":"2y"}
        iv  = {"M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}
        df = yf.download(PAIRS.get(sym, sym+"=X"), period=per.get(tf,"30d"),
                         interval=iv.get(tf,"1h"), auto_adjust=True, progress=False)
        if df.empty or len(df) < 60: return None, None, None
        if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df = calculate_all_indicators(df)
        r = run_all_strategies(df)
        return r, float(df["Close"].iloc[-1]), float(df["ATR"].iloc[-1])
    except Exception as e:
        log.warning(f"analyze {sym}: {e}"); return None, None, None


def news_blocked(sym):
    try:
        from news_calendar import should_block_trade
        b, _ = should_block_trade(sym, 30)
        return b
    except Exception:
        return False


def close_pos(state, pos, exit_px, reason):
    p = pip(pos["symbol"])
    sl_pips = abs(pos["entry"] - pos["sl"]) / p
    d = 1 if pos["direction"] == "Buy" else -1
    pnl_pips = (exit_px - pos["entry"]) / p * d - 1.5
    pnl = pos["risk_usd"] * (pnl_pips / max(sl_pips, 1))
    state["balance"] += pnl
    state["trades"].insert(0, {**pos, "exit": exit_px, "closed_at": now(),
        "pnl": round(pnl, 4), "reason": reason,
        "balance_after": round(state["balance"], 4),
        "total_return": round((state["balance"]/state["start_balance"]-1)*100, 2)})
    log.info(f"CLOSE {pos['symbol']} {pos['direction']} {reason} ${pnl:+.2f} -> bal ${state['balance']:.2f}")


def scan(state, cfg):
    # 1. manage open positions
    keep = []
    for pos in state["positions"]:
        px = fetch_price(pos["symbol"])
        if not px:
            keep.append(pos); continue
        d = 1 if pos["direction"] == "Buy" else -1
        hit_sl = px <= pos["sl"] if d == 1 else px >= pos["sl"]
        hit_tp = px >= pos["tp"] if d == 1 else px <= pos["tp"]
        if hit_sl:
            close_pos(state, pos, pos["sl"], "Stop Loss")
        elif hit_tp:
            close_pos(state, pos, pos["tp"], "Take Profit")
        else:
            p = pip(pos["symbol"]); sl_pips = abs(pos["entry"]-pos["sl"])/p
            pnl_pips = (px-pos["entry"])/p*d - 1.5
            pos["pnl"] = round(pos["risk_usd"]*(pnl_pips/max(sl_pips,1)), 4)
            pos["current_price"] = px
            keep.append(pos)
    state["positions"] = keep

    # 2. open new positions
    open_syms = {p["symbol"] for p in state["positions"]}
    for sym in cfg["symbols"]:
        if len(state["positions"]) >= cfg["max_positions"]: break
        if sym in open_syms: continue
        r, price, atr = analyze(sym, cfg["timeframe"])
        if r is None or atr <= 0: continue
        action = r["overall"]; conf = int(r["overall_strength"]*100)
        if action == "Hold" or conf < cfg["min_conf_pct"]: continue
        if cfg.get("news_filter") and news_blocked(sym):
            log.info(f"skip {sym}: news"); continue
        pp = PER_PAIR.get(sym, {"sl_atr":1.5,"tp_atr":1.5})
        d = 1 if action == "Buy" else -1
        sl = round(price - d*atr*pp["sl_atr"], 5)
        tp = round(price + d*atr*pp["tp_atr"], 5)
        risk = state["balance"] * (cfg["risk_pct"]/100)
        tid = state["scan_count"]*100 + len(state["trades"]) + len(state["positions"]) + 1
        state["positions"].append({"id":tid,"symbol":sym,"direction":action,
            "entry":price,"sl":sl,"tp":tp,"risk_usd":risk,"atr":atr,
            "opened_at":now(),"pnl":0.0,"current_price":price})
        open_syms.add(sym)
        log.info(f"OPEN {action} {sym} @ {price} SL {sl} TP {tp} conf {conf}%")

    state["scan_count"] += 1
    state["last_scan"] = now()
    state["trades"] = state["trades"][:500]
    save_state(state)


def main():
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    loop = "--loop" in sys.argv
    while True:
        cfg = load_config()
        state = load_state()
        if cfg.get("running", True):
            log.info(f"Scan #{state['scan_count']+1} | bal ${state['balance']:.2f} | open {len(state['positions'])}")
            try:
                scan(state, cfg)
            except Exception as e:
                log.error(f"scan error: {e}")
        else:
            log.info("paused (running=false)")
        if not loop:
            break
        time.sleep(cfg.get("scan_interval", 60))


if __name__ == "__main__":
    main()

"""
Autonomous Bot Engine — runs 24/7 with no UI.
Scans markets, opens/closes paper trades, saves everything to JSON.
Can be run locally or on a cloud server (Railway, Render, VPS).

Run:  python bot_engine.py
"""
from __future__ import annotations
import json, time, os, sys, logging
from datetime import datetime, timezone
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/bot_engine.log", mode="a"),
    ]
)
log = logging.getLogger("bot")

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE  = DATA_DIR / "bot_state.json"
CONFIG_FILE = DATA_DIR / "bot_config.json"

# ── Default config (editable via dashboard or config file) ────────────────────
DEFAULT_CONFIG = {
    "running":        True,
    "symbols":        ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"],
    "timeframe":      "H1",
    "min_conf_pct":   60,
    "risk_pct":       3,
    "max_positions":  3,
    "sl_mult":        1.5,
    "tp_mult":        2.5,
    "start_balance":  100.0,
    "scan_interval":  60,   # seconds
}

PIP_MAP = {
    "EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
    "USDCHF":.0001,"USDJPY":.01,"EURJPY":.01,"GBPJPY":.01,"XAUUSD":.1
}

def pip(sym): return PIP_MAP.get(sym, .0001)
def now_utc(): return datetime.now(timezone.utc).isoformat()
def now_str(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ── Config helpers ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

# ── State helpers ──────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    cfg = load_config()
    return {
        "balance":        cfg["start_balance"],
        "start_balance":  cfg["start_balance"],
        "positions":      [],
        "trades":         [],
        "scan_count":     0,
        "started_at":     now_utc(),
        "last_scan":      None,
    }

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

# ── Price fetch ────────────────────────────────────────────────────────────────
def fetch_price(symbol: str) -> float | None:
    try:
        import yfinance as yf
        from data_manager import PAIRS as YF_PAIRS
        ticker = YF_PAIRS.get(symbol, symbol + "=X")
        df = yf.download(ticker, period="1d", interval="1m",
                          auto_adjust=True, progress=False)
        if df.empty: return None
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"Price fetch failed for {symbol}: {e}")
        return None

# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze(symbol: str, timeframe: str):
    try:
        import yfinance as yf
        from data_manager import PAIRS as YF_PAIRS
        from strategies import calculate_all_indicators, run_all_strategies
        import pandas as pd

        ticker  = YF_PAIRS.get(symbol, symbol + "=X")
        per_map = {"M15":"5d","M30":"10d","H1":"30d","H4":"90d","D1":"2y"}
        iv_map  = {"M15":"15m","M30":"30m","H1":"1h","H4":"4h","D1":"1d"}

        df = yf.download(ticker, period=per_map.get(timeframe,"30d"),
                          interval=iv_map.get(timeframe,"1h"),
                          auto_adjust=True, progress=False)
        if df.empty or len(df) < 30: return None, None, None, None

        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df = calculate_all_indicators(df)
        res = run_all_strategies(df)
        price = float(df["Close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else price * 0.001
        return df, res, price, atr
    except Exception as e:
        log.warning(f"Analysis failed for {symbol}: {e}")
        return None, None, None, None

# ── Trade management ───────────────────────────────────────────────────────────
def open_trade(state, symbol, direction, price, sl, tp, risk_usd, atr):
    trade_id = state["scan_count"] * 100 + len(state["positions"]) + len(state["trades"]) + 1
    pos = {
        "id":        trade_id,
        "symbol":    symbol,
        "direction": direction,
        "entry":     price,
        "sl":        sl,
        "tp":        tp,
        "risk_usd":  risk_usd,
        "atr":       atr,
        "opened_at": now_str(),
        "pnl":       0.0,
    }
    state["positions"].append(pos)
    log.info(f"OPEN  {direction:4s} {symbol} @ {price:.5f} | SL {sl:.5f} | TP {tp:.5f} | Risk ${risk_usd:.2f}")
    return pos

def close_trade(state, pos, exit_price, reason):
    p        = pip(pos["symbol"])
    sl_pips  = abs(pos["entry"] - pos["sl"]) / p
    if pos["direction"] == "Buy":
        pnl_pips = (exit_price - pos["entry"]) / p
    else:
        pnl_pips = (pos["entry"] - exit_price) / p
    pnl_pips -= 1.5  # spread cost
    r_mult    = pnl_pips / max(sl_pips, 1)
    pnl       = pos["risk_usd"] * r_mult

    state["balance"] += pnl
    total_ret = (state["balance"] / state["start_balance"] - 1) * 100

    closed = {
        **pos,
        "exit":          exit_price,
        "closed_at":     now_str(),
        "pnl":           round(pnl, 4),
        "pnl_pct":       round(pnl / pos["risk_usd"] * 100, 2),
        "reason":        reason,
        "balance_after": round(state["balance"], 4),
        "total_return":  round(total_ret, 2),
    }
    state["trades"].insert(0, closed)
    state["positions"] = [p for p in state["positions"] if p["id"] != pos["id"]]

    icon = "✅ WIN " if pnl > 0 else "❌ LOSS"
    log.info(f"CLOSE {icon} {pos['symbol']} {pos['direction']} @ {exit_price:.5f} | "
             f"P/L ${pnl:+.4f} | Balance ${state['balance']:.4f} ({total_ret:+.2f}%)")
    return closed

# ── Main scan ──────────────────────────────────────────────────────────────────
def run_scan(state: dict, config: dict):
    symbols       = config["symbols"]
    timeframe     = config["timeframe"]
    min_conf      = config["min_conf_pct"]
    risk_pct      = config["risk_pct"]
    max_pos       = config["max_positions"]
    sl_mult       = config["sl_mult"]
    tp_mult       = config["tp_mult"]

    log.info(f"=== Scan #{state['scan_count']+1} | Balance ${state['balance']:.4f} | "
             f"Open: {len(state['positions'])} ===")

    # ── Check SL/TP on open positions ──
    closed_ids = []
    for pos in list(state["positions"]):
        price = fetch_price(pos["symbol"])
        if not price:
            continue
        # Update unrealized P/L
        p        = pip(pos["symbol"])
        sl_pips  = abs(pos["entry"] - pos["sl"]) / p
        exit_d   = (price - pos["entry"]) if pos["direction"]=="Buy" else (pos["entry"] - price)
        pos["pnl"]           = round(pos["risk_usd"] * ((exit_d/p - 1.5) / max(sl_pips,1)), 4)
        pos["current_price"] = price

        if pos["direction"] == "Buy":
            if price <= pos["sl"]:
                close_trade(state, pos, pos["sl"], "Stop Loss ❌")
                closed_ids.append(pos["id"])
            elif price >= pos["tp"]:
                close_trade(state, pos, pos["tp"], "Take Profit ✅")
                closed_ids.append(pos["id"])
        else:
            if price >= pos["sl"]:
                close_trade(state, pos, pos["sl"], "Stop Loss ❌")
                closed_ids.append(pos["id"])
            elif price <= pos["tp"]:
                close_trade(state, pos, pos["tp"], "Take Profit ✅")
                closed_ids.append(pos["id"])

    # ── Look for new entries ──
    open_syms = {p["symbol"] for p in state["positions"]}
    if len(state["positions"]) < max_pos:
        for sym in symbols:
            if sym in open_syms:
                continue
            if len(state["positions"]) >= max_pos:
                break

            _, res, price, atr = analyze(sym, timeframe)
            if res is None:
                continue

            action   = res["overall"]
            conf_pct = int(res["overall_strength"] * 100)

            if action == "Hold" or conf_pct < min_conf:
                log.info(f"  {sym}: {action} ({conf_pct}%) — skip")
                continue

            risk_usd = state["balance"] * (risk_pct / 100)
            sl_d = atr * sl_mult
            tp_d = atr * tp_mult

            if action == "Buy":
                sl = round(price - sl_d, 5)
                tp = round(price + tp_d, 5)
            else:
                sl = round(price + sl_d, 5)
                tp = round(price - tp_d, 5)

            open_trade(state, sym, action, price, sl, tp, risk_usd, atr)
            open_syms.add(sym)

    state["scan_count"] += 1
    state["last_scan"]   = now_str()

    # Keep trades list to last 500
    state["trades"] = state["trades"][:500]
    save_state(state)

# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    log.info("🤖 Bot Engine started")

    # Ensure default config exists
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        log.info(f"Created default config at {CONFIG_FILE}")

    state = load_state()
    log.info(f"Loaded state: balance=${state['balance']:.4f}, "
             f"trades={len(state['trades'])}, positions={len(state['positions'])}")

    while True:
        try:
            config = load_config()
            interval = config.get("scan_interval", 60)

            if config.get("running", True):
                run_scan(state, config)
            else:
                log.info("Bot paused (running=false in config). Sleeping...")

            time.sleep(interval)

        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception as e:
            log.error(f"Scan error: {e}", exc_info=True)
            time.sleep(30)  # wait before retrying

if __name__ == "__main__":
    main()

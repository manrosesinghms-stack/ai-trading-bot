"""
TradingView Webhook Server
Receives alerts from TradingView Pine Script strategies and places trades on MT5.

Run:  python webhook_server.py
Then use ngrok to get a public URL:  ngrok http 8000
Paste the ngrok URL into your TradingView alert webhook field.

TradingView alert message format (JSON):
  Simple:  {"action": "buy", "symbol": "EURUSD"}
  Full:    {"action": "buy", "symbol": "EURUSD", "lots": 0.1, "sl_pips": 50, "tp_pips": 100}
  Close:   {"action": "close", "symbol": "EURUSD"}
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

# ── FastAPI ────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("Installing fastapi + uvicorn...")
    os.system(f"{sys.executable} -m pip install fastapi uvicorn -q")
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn

DATA_DIR     = Path(__file__).parent / "data"
SIGNALS_FILE = DATA_DIR / "webhook_signals.json"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="TradingView Webhook Receiver")

# ── Symbol mapping TV → our format ────────────────────────────────────────────
TV_SYMBOL_MAP = {
    "EURUSD": "EURUSD", "FX:EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD", "FX:GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY", "FX:USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "FX:AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD", "FX:USDCAD": "USDCAD",
    "GBPJPY": "GBPJPY", "FX:GBPJPY": "GBPJPY",
    "EURJPY": "EURJPY", "FX:EURJPY": "EURJPY",
    "XAUUSD": "XAUUSD", "TVC:GOLD":  "XAUUSD",
    "GOLD":   "XAUUSD", "OANDA:XAUUSD": "XAUUSD",
}

# ── Signal store (shared with Streamlit via JSON file) ─────────────────────────
def load_signals() -> list:
    if SIGNALS_FILE.exists():
        try:
            return json.loads(SIGNALS_FILE.read_text())
        except Exception:
            pass
    return []

def save_signals(signals: list):
    SIGNALS_FILE.write_text(json.dumps(signals[-50:], indent=2))  # keep last 50

# ── MT5 trade execution ────────────────────────────────────────────────────────
def execute_on_mt5(symbol: str, action: str, lots: float,
                   sl_pips: float, tp_pips: float, comment: str) -> dict:
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return {"ok": False, "msg": "MT5 not running"}

        pip_map = {
            "EURUSD":.0001,"GBPUSD":.0001,"AUDUSD":.0001,"USDCAD":.0001,
            "USDJPY":.01,"EURJPY":.01,"GBPJPY":.01,"XAUUSD":.1
        }
        pip_size = pip_map.get(symbol, .0001)

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"ok": False, "msg": f"No tick for {symbol}"}

        if action == "buy":
            entry = tick.ask
            sl    = round(entry - sl_pips * pip_size, 5)
            tp    = round(entry + tp_pips * pip_size, 5)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry = tick.bid
            sl    = round(entry + sl_pips * pip_size, 5)
            tp    = round(entry - tp_pips * pip_size, 5)
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      float(lots),
            "type":        order_type,
            "price":       entry,
            "sl":          sl,
            "tp":          tp,
            "deviation":   20,
            "magic":       999999,
            "comment":     comment,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"ok": True, "ticket": result.order, "entry": entry}
        return {"ok": False, "msg": f"MT5 error {result.retcode}: {result.comment}"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def close_all_on_mt5(symbol: str) -> dict:
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return {"ok": False, "msg": "MT5 not running"}
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return {"ok": True, "msg": "No positions to close"}
        closed = 0
        for pos in positions:
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if pos.type == 0 else tick.ask
            order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            req = {
                "action":   mt5.TRADE_ACTION_DEAL,
                "symbol":   symbol,
                "volume":   pos.volume,
                "type":     order_type,
                "position": pos.ticket,
                "price":    price,
                "deviation": 20,
                "magic":    999999,
                "comment":  "TV Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            r = mt5.order_send(req)
            if r.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
        return {"ok": True, "msg": f"Closed {closed} positions"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ── Webhook endpoint ────────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        try:
            body = json.loads(raw.decode())
        except Exception:
            raise HTTPException(400, "Invalid JSON body")

    # Normalise
    action  = str(body.get("action", "")).lower().strip()
    symbol  = TV_SYMBOL_MAP.get(str(body.get("symbol", "EURUSD")).upper(), "EURUSD")
    lots    = float(body.get("lots",    body.get("qty", 0.10)))
    sl_pips = float(body.get("sl_pips", body.get("sl",  50)))
    tp_pips = float(body.get("tp_pips", body.get("tp",  100)))
    comment = str(body.get("comment", "TradingView"))

    if action not in ("buy", "sell", "close", "closebuy", "closesell"):
        raise HTTPException(400, f"Unknown action: {action}")

    # Execute
    if action in ("buy", "sell"):
        result = execute_on_mt5(symbol, action, lots, sl_pips, tp_pips, comment)
    else:
        result = close_all_on_mt5(symbol)

    # Log signal
    entry = {
        "time":    datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "symbol":  symbol,
        "action":  action.upper(),
        "lots":    lots,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "result":  result,
        "raw":     body,
    }
    signals = load_signals()
    signals.insert(0, entry)
    save_signals(signals)

    status = "✅ Executed" if result.get("ok") else f"❌ Failed: {result.get('msg')}"
    print(f"[{entry['time']}] {symbol} {action.upper()} → {status}")
    return JSONResponse({"status": "ok", "result": result, "signal": entry})


@app.get("/")
async def health():
    signals = load_signals()
    last = signals[0] if signals else None
    return {
        "status":       "running",
        "signals_received": len(signals),
        "last_signal":  last,
        "webhook_url":  "POST /webhook",
        "example_payload": {
            "action": "buy",
            "symbol": "EURUSD",
            "lots":   0.1,
            "sl_pips": 50,
            "tp_pips": 100,
        }
    }

@app.get("/signals")
async def get_signals():
    return load_signals()


# ── ngrok auto-tunnel ──────────────────────────────────────────────────────────
def start_ngrok(port: int = 8000):
    try:
        from pyngrok import ngrok, conf
        tunnel = ngrok.connect(port, "http")
        url    = tunnel.public_url
        print("\n" + "="*60)
        print(f"  ✅ PUBLIC WEBHOOK URL:")
        print(f"  {url}/webhook")
        print(f"\n  Paste this URL into TradingView:")
        print(f"  Alerts → Notifications → Webhook URL")
        print("="*60 + "\n")
        SIGNALS_FILE.parent.mkdir(exist_ok=True)
        (SIGNALS_FILE.parent / "webhook_url.txt").write_text(f"{url}/webhook")
        return url
    except ImportError:
        print("pyngrok not installed. Run: pip install pyngrok")
        print(f"Local URL (not accessible from TradingView): http://localhost:{port}/webhook")
        return None
    except Exception as e:
        print(f"ngrok failed: {e}")
        print(f"Webhook available locally at: http://localhost:{port}/webhook")
        return None


if __name__ == "__main__":
    print("\n🚀 Starting TradingView Webhook Server...")
    start_ngrok(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

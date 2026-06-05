from __future__ import annotations
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def connect(login: int | None = None, password: str = "", server: str = "") -> tuple[bool, str]:
    if not mt5.initialize():
        return False, f"MT5 initialize() failed: {mt5.last_error()}"
    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            return False, f"MT5 login failed: {mt5.last_error()}"
    return True, "Connected successfully"


def disconnect():
    mt5.shutdown()


def get_account_info() -> dict | None:
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "balance":     info.balance,
        "equity":      info.equity,
        "margin":      info.margin,
        "free_margin": info.margin_free,
        "margin_level": info.margin_level,
        "profit":      info.profit,
        "currency":    info.currency,
        "leverage":    info.leverage,
        "name":        info.name,
        "login":       info.login,
        "server":      info.server,
    }


def get_ohlcv(symbol: str, timeframe: str = "H1", bars: int = 500) -> pd.DataFrame | None:
    tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_H1)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    df.set_index("time", inplace=True)
    return df


def get_current_price(symbol: str) -> dict | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    spread_pips = round((tick.ask - tick.bid) / mt5.symbol_info(symbol).point / 10, 1)
    return {"bid": tick.bid, "ask": tick.ask, "spread": spread_pips}


def get_pip_value(symbol: str) -> float:
    """Return tick value in account currency for 1 standard lot."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return 10.0
    return info.trade_tick_value * (info.point / info.trade_tick_size)


def get_open_positions() -> list[dict]:
    positions = mt5.positions_get()
    if positions is None:
        return []
    return [
        {
            "ticket":        p.ticket,
            "symbol":        p.symbol,
            "type":          "Buy" if p.type == 0 else "Sell",
            "volume":        p.volume,
            "open_price":    p.price_open,
            "current_price": p.price_current,
            "sl":            p.sl,
            "tp":            p.tp,
            "profit":        p.profit,
            "swap":          p.swap,
            "open_time":     datetime.fromtimestamp(p.time),
            "comment":       p.comment,
        }
        for p in positions
    ]


def get_trade_history(days: int = 30) -> list[dict]:
    date_from = datetime.now() - timedelta(days=days)
    deals = mt5.history_deals_get(date_from, datetime.now())
    if deals is None:
        return []
    return [
        {
            "ticket":  d.ticket,
            "symbol":  d.symbol,
            "type":    "Buy" if d.type == 0 else "Sell",
            "volume":  d.volume,
            "price":   d.price,
            "profit":  d.profit,
            "commission": d.commission,
            "swap":    d.swap,
            "time":    datetime.fromtimestamp(d.time),
            "comment": d.comment,
        }
        for d in deals
        if d.entry == 1  # exit deals only
    ]


def place_order(
    symbol: str,
    order_type: str,
    volume: float,
    sl_price: float,
    tp_price: float,
    comment: str = "AI Trade",
) -> tuple[int | None, str]:
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return None, "Symbol not found"

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, "Failed to get tick"

    digits = sym_info.digits
    action_price = tick.ask if order_type == "Buy" else tick.bid
    mt5_type = mt5.ORDER_TYPE_BUY if order_type == "Buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      float(volume),
        "type":        mt5_type,
        "price":       action_price,
        "sl":          round(sl_price, digits),
        "tp":          round(tp_price, digits),
        "deviation":   20,
        "magic":       234000,
        "comment":     comment,
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return None, f"Order failed — {result.comment} (code {result.retcode})"
    return result.order, "Order placed successfully"


def close_position(ticket: int) -> tuple[bool, str]:
    pos_list = mt5.positions_get(ticket=ticket)
    if not pos_list:
        return False, "Position not found"

    pos = pos_list[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return False, "Failed to get tick"

    if pos.type == mt5.ORDER_TYPE_BUY:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      pos.symbol,
        "volume":      pos.volume,
        "type":        order_type,
        "position":    ticket,
        "price":       price,
        "deviation":   20,
        "magic":       234000,
        "comment":     "Close by AI",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Close failed — {result.comment}"
    return True, "Position closed"

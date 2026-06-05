"""
Risk management — position sizing, SL/TP, drawdown guard, Kelly Criterion.
"""
from __future__ import annotations
import math


# ─────────────────────────── Basic position sizing ───────────────────────────

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    sl_pips: float,
    pip_value_per_lot: float = 10.0,
) -> float:
    """Return lot size (0.01 step) that risks risk_percent% of balance."""
    if sl_pips <= 0 or pip_value_per_lot <= 0:
        return 0.01
    risk_amount = account_balance * (risk_percent / 100)
    lots = risk_amount / (sl_pips * pip_value_per_lot)
    return max(0.01, min(round(lots, 2), 10.0))


# ─────────────────────────── Kelly Criterion sizing ──────────────────────────

def kelly_position_size(
    account_balance: float,
    win_rate: float,          # 0.0–1.0
    avg_rr: float,            # average win / average loss (e.g. 2.0 for 2:1)
    kelly_fraction: float = 0.25,   # use 25% Kelly for safety
    max_risk_pct: float = 3.0,
    min_risk_pct: float = 0.3,
    sl_pips: float = 50.0,
    pip_value_per_lot: float = 10.0,
) -> dict:
    """
    Kelly Criterion position sizing.
    f* = (p * b - q) / b  where p=win_rate, q=1-p, b=avg_rr

    Uses fractional Kelly (default 25%) to reduce volatility of the Kelly bet.
    Caps at max_risk_pct and floors at min_risk_pct.
    """
    if win_rate <= 0 or win_rate >= 1 or avg_rr <= 0:
        return {"lots": 0.01, "risk_pct": min_risk_pct, "method": "fallback"}

    p = win_rate
    q = 1.0 - p
    b = avg_rr

    full_kelly = (p * b - q) / b   # fraction of account to risk

    if full_kelly <= 0:
        # Negative edge — don't trade
        return {"lots": 0.01, "risk_pct": 0.0, "method": "negative_edge",
                "full_kelly": round(full_kelly, 4), "note": "Strategy has negative edge — do not trade"}

    fractional_kelly = full_kelly * kelly_fraction
    risk_pct = min(max(fractional_kelly * 100, min_risk_pct), max_risk_pct)

    lots = calculate_position_size(account_balance, risk_pct, sl_pips, pip_value_per_lot)

    return {
        "lots":          lots,
        "risk_pct":      round(risk_pct, 2),
        "full_kelly":    round(full_kelly * 100, 2),
        "frac_kelly":    round(fractional_kelly * 100, 2),
        "kelly_fraction": kelly_fraction,
        "method":        "kelly",
        "note":          f"Full Kelly={full_kelly*100:.1f}%, using {kelly_fraction*100:.0f}% fraction → {risk_pct:.2f}% risk",
    }


# ─────────────────────────── Drawdown-adaptive sizing ────────────────────────

DRAWDOWN_SCALE_TABLE = [
    (0.0,  3.0,  1.00),   # <3% DD → full size
    (3.0,  5.0,  0.75),   # 3–5% → 75%
    (5.0,  7.0,  0.50),   # 5–7% → 50%
    (7.0,  9.0,  0.25),   # 7–9% → 25%
    (9.0,  999., 0.00),   # >9% → stop trading
]


def drawdown_scale_factor(current_drawdown_pct: float) -> tuple[float, str]:
    """Return (scale_factor, description) based on current drawdown."""
    for low, high, factor in DRAWDOWN_SCALE_TABLE:
        if low <= current_drawdown_pct < high:
            pct = int(factor * 100)
            desc = (
                f"Full size (DD {current_drawdown_pct:.1f}%)" if factor == 1.0 else
                f"Reduced to {pct}% of normal size (DD {current_drawdown_pct:.1f}%)" if factor > 0 else
                f"TRADING HALTED — drawdown {current_drawdown_pct:.1f}% exceeds limit"
            )
            return factor, desc
    return 0.0, f"TRADING HALTED — drawdown {current_drawdown_pct:.1f}%"


def adaptive_position_size(
    account_balance: float,
    equity: float,
    base_risk_pct: float,
    sl_pips: float,
    pip_value_per_lot: float = 10.0,
    use_kelly: bool = False,
    win_rate: float = 0.55,
    avg_rr: float = 2.0,
) -> dict:
    """
    Combine drawdown scaling with optional Kelly sizing.
    Returns the final recommended lot size.
    """
    drawdown_pct = max(0.0, (account_balance - equity) / max(account_balance, 1) * 100)
    dd_factor, dd_desc = drawdown_scale_factor(drawdown_pct)

    if dd_factor == 0.0:
        return {
            "lots": 0.0, "risk_pct": 0.0, "can_trade": False,
            "dd_factor": 0.0, "dd_desc": dd_desc,
            "drawdown_pct": round(drawdown_pct, 2),
        }

    if use_kelly:
        kelly = kelly_position_size(
            account_balance, win_rate, avg_rr,
            kelly_fraction=0.25,
            max_risk_pct=base_risk_pct * 1.5,
            sl_pips=sl_pips, pip_value_per_lot=pip_value_per_lot,
        )
        if kelly.get("method") == "negative_edge":
            return {**kelly, "can_trade": False, "drawdown_pct": round(drawdown_pct, 2)}
        effective_risk = kelly["risk_pct"] * dd_factor
        lots = calculate_position_size(account_balance, effective_risk, sl_pips, pip_value_per_lot)
        return {
            "lots":           lots,
            "risk_pct":       round(effective_risk, 2),
            "can_trade":      True,
            "method":         "kelly+dd",
            "dd_factor":      dd_factor,
            "dd_desc":        dd_desc,
            "drawdown_pct":   round(drawdown_pct, 2),
            "kelly_note":     kelly["note"],
        }
    else:
        effective_risk = base_risk_pct * dd_factor
        lots = calculate_position_size(account_balance, effective_risk, sl_pips, pip_value_per_lot)
        return {
            "lots":         lots,
            "risk_pct":     round(effective_risk, 2),
            "can_trade":    True,
            "method":       "fixed+dd",
            "dd_factor":    dd_factor,
            "dd_desc":      dd_desc,
            "drawdown_pct": round(drawdown_pct, 2),
        }


# ─────────────────────────── SL / TP calculation ─────────────────────────────

def calculate_sl_tp(
    entry_price: float,
    direction: str,
    sl_pips: float,
    tp_pips: float,
    symbol: str = "EURUSD",
) -> tuple[float, float]:
    """Return (stop_loss_price, take_profit_price)."""
    if "JPY" in symbol:
        pip_size = 0.01
    elif "XAU" in symbol:
        pip_size = 0.1
    else:
        pip_size = 0.0001

    sl_dist = sl_pips * pip_size
    tp_dist = tp_pips * pip_size

    if direction == "Buy":
        return round(entry_price - sl_dist, 5), round(entry_price + tp_dist, 5)
    return round(entry_price + sl_dist, 5), round(entry_price - tp_dist, 5)


def multi_tp_levels(
    entry_price: float,
    direction: str,
    sl_pips: float,
    symbol: str = "EURUSD",
    rr_levels: list[float] = (1.5, 2.5, 4.0),
) -> list[dict]:
    """Return multiple take-profit levels at given R:R multiples."""
    if "JPY" in symbol:
        pip_size = 0.01
    elif "XAU" in symbol:
        pip_size = 0.1
    else:
        pip_size = 0.0001

    results = []
    for rr in rr_levels:
        tp_pips = sl_pips * rr
        tp_dist = tp_pips * pip_size
        tp_price = round(
            entry_price + tp_dist if direction == "Buy" else entry_price - tp_dist,
            5,
        )
        results.append({"rr": rr, "tp_pips": round(tp_pips, 1), "tp_price": tp_price})
    return results


def trailing_stop_price(
    current_price: float,
    direction: str,
    trail_pips: float,
    symbol: str = "EURUSD",
) -> float:
    """Return trailing stop price at trail_pips distance from current price."""
    pip_size = 0.01 if "JPY" in symbol else (0.1 if "XAU" in symbol else 0.0001)
    dist = trail_pips * pip_size
    return round(current_price - dist if direction == "Buy" else current_price + dist, 5)


def breakeven_stop(
    entry_price: float,
    direction: str,
    breakeven_pips: float = 5.0,
    symbol: str = "EURUSD",
) -> float:
    """Move SL to entry + breakeven_pips to lock in some profit."""
    pip_size = 0.01 if "JPY" in symbol else (0.1 if "XAU" in symbol else 0.0001)
    dist = breakeven_pips * pip_size
    return round(
        entry_price + dist if direction == "Buy" else entry_price - dist,
        5,
    )


# ─────────────────────────── Portfolio risk guard ────────────────────────────

def assess_risk(
    account_info: dict,
    open_positions: list[dict],
    max_positions: int = 5,
    max_drawdown_pct: float = 10.0,
) -> dict:
    balance  = account_info.get("balance", 1)
    equity   = account_info.get("equity", balance)
    drawdown = max(0.0, (balance - equity) / balance * 100)
    dd_factor, dd_desc = drawdown_scale_factor(drawdown)
    pos_count = len(open_positions)

    warnings = []
    can_trade = True

    if pos_count >= max_positions:
        warnings.append(f"Max positions reached ({max_positions})")
        can_trade = False

    if drawdown >= max_drawdown_pct:
        warnings.append(f"Drawdown {drawdown:.1f}% — trading halted until equity recovers")
        can_trade = False
    elif drawdown >= max_drawdown_pct * 0.7:
        warnings.append(f"High drawdown warning: {drawdown:.1f}%")

    if equity < balance * 0.85:
        warnings.append("Equity well below balance — consider closing losing positions")

    if dd_desc and dd_factor < 1.0 and dd_factor > 0:
        warnings.append(f"Position size scaled: {dd_desc}")

    return {
        "can_trade":        can_trade,
        "warnings":         warnings,
        "current_drawdown": round(drawdown, 2),
        "position_count":   pos_count,
        "dd_scale_factor":  dd_factor,
        "dd_scale_desc":    dd_desc,
        "risk_level": "High" if drawdown > 7 else "Medium" if drawdown > 3 else "Low",
    }


# ─────────────────────────── Strategy edge metrics ───────────────────────────

def calculate_edge_metrics(trades: list[dict]) -> dict:
    """Calculate win rate, avg R:R and expected value from a list of closed trades."""
    if not trades:
        return {"win_rate": 0.5, "avg_rr": 2.0, "ev_per_trade": 0.0, "trades": 0}

    pnls  = [t.get("pnl_usd", 0) for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.5
    avg_win  = sum(wins)   / len(wins)   if wins   else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1
    avg_rr   = avg_win / max(avg_loss, 1e-6)
    ev       = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "win_rate":     round(win_rate, 3),
        "avg_rr":       round(avg_rr, 2),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "ev_per_trade": round(ev, 2),
        "trades":       len(pnls),
    }

from datetime import datetime, timezone

SESSIONS = {
    "Sydney":   {"open": 21, "close": 6},
    "Tokyo":    {"open": 23, "close": 8},
    "London":   {"open": 7,  "close": 16},
    "New York": {"open": 13, "close": 22},
}

SESSION_COLORS = {
    "Sydney":   "#1f77b4",
    "Tokyo":    "#ff7f0e",
    "London":   "#2ca02c",
    "New York": "#d62728",
}

BEST_PAIRS = {
    "Sydney":   ["AUDUSD", "AUDNZD", "NZDUSD"],
    "Tokyo":    ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"],
    "London":   ["EURUSD", "GBPUSD", "EURGBP", "XAUUSD"],
    "New York": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD"],
}


def _is_active(open_h: int, close_h: int, hour: int) -> bool:
    if open_h > close_h:  # crosses midnight
        return hour >= open_h or hour < close_h
    return open_h <= hour < close_h


def get_active_sessions() -> list[str]:
    hour = datetime.now(timezone.utc).hour
    return [s for s, t in SESSIONS.items() if _is_active(t["open"], t["close"], hour)]


def get_session_quality() -> tuple[str, str]:
    active = get_active_sessions()
    if "London" in active and "New York" in active:
        return "Excellent", "London / NY Overlap — peak liquidity & volatility"
    if "London" in active:
        return "Good", "London Session — high liquidity, EUR/GBP pairs ideal"
    if "New York" in active:
        return "Good", "New York Session — USD pairs most active"
    if "Tokyo" in active:
        return "Fair", "Tokyo Session — JPY pairs ideal, moderate liquidity"
    if "Sydney" in active:
        return "Poor", "Sydney Session — low liquidity, AUD/NZD pairs only"
    return "Poor", "No major session active — avoid trading"


def get_session_status() -> list[dict]:
    hour = datetime.now(timezone.utc).hour
    active = get_active_sessions()
    result = []
    for session, times in SESSIONS.items():
        is_active = session in active
        open_h, close_h = times["open"], times["close"]
        if is_active:
            if close_h <= hour:
                hours_left = close_h + 24 - hour
            else:
                hours_left = close_h - hour
            state = f"ACTIVE — {hours_left}h remaining"
        else:
            if open_h > hour:
                hours_until = open_h - hour
            else:
                hours_until = 24 - hour + open_h
            state = f"Closed — opens in {hours_until}h"
        result.append({
            "session": session,
            "active": is_active,
            "state": state,
            "color": SESSION_COLORS[session],
            "best_pairs": BEST_PAIRS[session],
        })
    return result

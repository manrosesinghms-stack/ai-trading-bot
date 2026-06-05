"""
Economic calendar — fetches high-impact forex news events.
Source: ForexFactory public JSON feed (used by thousands of MT4/MT5 EAs).
Warns the AI and UI before high-impact events so the bot avoids trading.
"""
from __future__ import annotations
import json
import os
import requests
from datetime import datetime, timezone, timedelta
import pandas as pd

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "news_calendar.json")
CACHE_TTL_HOURS = 6   # re-fetch every 6 hours

FF_THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_NEXT_WEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

# Currencies whose events affect which pairs
CURRENCY_TO_PAIRS: dict[str, list[str]] = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD"],
    "EUR": ["EURUSD", "EURJPY"],
    "GBP": ["GBPUSD", "GBPJPY"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY"],
    "AUD": ["AUDUSD"],
    "CAD": ["USDCAD"],
    "CHF": ["USDCHF"],
    "NZD": ["NZDUSD"],
    "XAU": ["XAUUSD"],
}

HIGH_IMPACT_KEYWORDS = [
    "nfp", "non-farm", "interest rate", "fomc", "fed", "rate decision",
    "cpi", "inflation", "gdp", "unemployment", "payroll", "ecb", "boe",
    "boj", "rba", "rbnz", "bank of", "central bank", "monetary policy",
    "press conference", "speech", "testimony", "pmi flash",
]


def _is_high_impact(title: str, impact: str) -> bool:
    if impact.lower() in ("high", "3"):
        return True
    tl = title.lower()
    return any(kw in tl for kw in HIGH_IMPACT_KEYWORDS)


def _fetch_calendar() -> list[dict]:
    """Fetch this week + next week calendar from ForexFactory."""
    events = []
    for url in [FF_THIS_WEEK, FF_NEXT_WEEK]:
        try:
            r = requests.get(url, timeout=8,
                              headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    events.extend(data)
        except Exception:
            pass
    return events


def _load_cached() -> list[dict] | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            blob = json.load(f)
        fetched_at = datetime.fromisoformat(blob.get("fetched_at", "2000-01-01"))
        age_hours  = (datetime.now() - fetched_at).total_seconds() / 3600
        if age_hours < CACHE_TTL_HOURS:
            return blob.get("events", [])
    except Exception:
        pass
    return None


def get_events(force_refresh: bool = False) -> list[dict]:
    if not force_refresh:
        cached = _load_cached()
        if cached is not None:
            return cached

    events = _fetch_calendar()
    if events:
        with open(CACHE_FILE, "w") as f:
            json.dump({"fetched_at": datetime.now().isoformat(), "events": events}, f)
    return events


def get_upcoming_events(
    pair: str,
    minutes_window: int = 120,   # ± 2 hours from now
) -> list[dict]:
    """Return high-impact events for a pair within ±window minutes of now."""
    events = get_events()
    if not events:
        return []

    now   = datetime.now(timezone.utc)
    start = now - timedelta(minutes=30)
    end   = now + timedelta(minutes=minutes_window)

    # Currency codes relevant to this pair
    relevant_currencies: set[str] = set()
    for ccy, pairs in CURRENCY_TO_PAIRS.items():
        if pair in pairs:
            relevant_currencies.add(ccy)

    found = []
    for ev in events:
        try:
            # ForexFactory date format: "01-06-2025"
            dt_str = ev.get("date", "") + " " + ev.get("time", "12:00am")
            dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
            dt = dt.replace(tzinfo=timezone.utc)  # FF times are New York — treat as UTC approx
        except Exception:
            continue

        ccy    = ev.get("currency", "").upper()
        title  = ev.get("title", "")
        impact = ev.get("impact", "")

        if ccy not in relevant_currencies:
            continue
        if not _is_high_impact(title, impact):
            continue
        if not (start <= dt <= end):
            continue

        mins_away = int((dt - now).total_seconds() / 60)
        found.append({
            "title":     title,
            "currency":  ccy,
            "datetime":  dt.strftime("%Y-%m-%d %H:%M UTC"),
            "mins_away": mins_away,
            "impact":    impact,
        })

    return sorted(found, key=lambda x: abs(x["mins_away"]))


def should_block_trade(pair: str, block_window_mins: int = 30) -> tuple[bool, str]:
    """Return (block, reason). Block if high-impact event within ±block_window_mins."""
    upcoming = get_upcoming_events(pair, minutes_window=block_window_mins)
    if not upcoming:
        return False, ""
    ev = upcoming[0]
    return True, f"HIGH-IMPACT NEWS in {ev['mins_away']} min: {ev['currency']} {ev['title']} ({ev['datetime']})"


def format_calendar_for_ai(pair: str) -> str:
    """Return news warning block for AI prompt injection."""
    upcoming = get_upcoming_events(pair, minutes_window=240)  # 4-hour window
    if not upcoming:
        return "ECONOMIC CALENDAR: No high-impact events for this pair in the next 4 hours. ✅"

    lines = ["ECONOMIC CALENDAR — UPCOMING HIGH-IMPACT EVENTS (next 4h):"]
    for ev in upcoming[:5]:
        prefix = "🚨 IMMINENT" if abs(ev["mins_away"]) <= 30 else "⚠️ Upcoming"
        lines.append(f"  {prefix} [{ev['mins_away']:+d} min] {ev['currency']} — {ev['title']} ({ev['datetime']})")
    lines.append("")
    lines.append("  ⚠️  Avoid entering new positions within 30 minutes before/after these events.")
    lines.append("  Consider widening SL or waiting for post-news candle to close.")
    return "\n".join(lines)


def get_week_schedule(pair: str) -> pd.DataFrame:
    """Return a DataFrame of all high-impact events this week for a pair."""
    events = get_events()
    relevant_currencies: set[str] = set()
    for ccy, pairs in CURRENCY_TO_PAIRS.items():
        if pair in pairs:
            relevant_currencies.add(ccy)

    rows = []
    for ev in events:
        ccy    = ev.get("currency", "").upper()
        title  = ev.get("title", "")
        impact = ev.get("impact", "")
        if ccy not in relevant_currencies:
            continue
        if not _is_high_impact(title, impact):
            continue
        rows.append({
            "Date":     ev.get("date", ""),
            "Time":     ev.get("time", ""),
            "Currency": ccy,
            "Event":    title,
            "Impact":   impact,
            "Forecast": ev.get("forecast", ""),
            "Previous": ev.get("previous", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

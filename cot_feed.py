"""
COT (Commitment of Traders) institutional positioning feed.
Source: CFTC weekly reports via cot_reports library.
Verified finding: Managed Money net position is the best proxy for speculative
institutional sentiment (research vote 3-0 for library, 2-1 for signal value).

Usage:
    from cot_feed import get_cot_signal, get_cot_context_for_ai
"""
from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

COT_CACHE = os.path.join(os.path.dirname(__file__), "data", "cot")
os.makedirs(COT_CACHE, exist_ok=True)

# Map our pair names → COT market name search strings (Legacy / Disaggregated reports)
PAIR_TO_COT: dict[str, str] = {
    "EURUSD": "EURO FX",
    "GBPUSD": "BRITISH POUND",
    "USDJPY": "JAPANESE YEN",
    "AUDUSD": "AUSTRALIAN DOLLAR",
    "USDCAD": "CANADIAN DOLLAR",
    "USDCHF": "SWISS FRANC",
    "XAUUSD": "GOLD",
    "EURJPY": "EURO FX",        # EUR proxy
    "GBPJPY": "BRITISH POUND",  # GBP proxy
    "NZDUSD": "NEW ZEALAND DOLLAR",
}

# Managed Money columns in CFTC Disaggregated Futures report
MM_LONG_COL  = "M_Money_Positions_Long_All"
MM_SHORT_COL = "M_Money_Positions_Short_All"
# Commercial columns in Legacy report (fallback)
COMM_LONG    = "Comm_Positions_Long_All"
COMM_SHORT   = "Comm_Positions_Short_All"
NON_COMM_L   = "NonComm_Positions_Long_All"
NON_COMM_S   = "NonComm_Positions_Short_All"

# Date column name
DATE_COL = "As_of_Date_In_Form_YYMMDD"


def _cache_path(report_type: str, year: int) -> str:
    return os.path.join(COT_CACHE, f"{report_type}_{year}.csv")


def _download_cot_data(years: int = 3) -> pd.DataFrame | None:
    """Download and cache COT disaggregated futures data."""
    try:
        import cot_reports as cot
    except ImportError:
        print("cot-reports not installed. Run: pip install cot-reports")
        return None

    frames = []
    current_year = datetime.now().year

    for yr in range(current_year - years, current_year + 1):
        cache = _cache_path("disagg", yr)
        if os.path.exists(cache):
            df = pd.read_csv(cache, low_memory=False)
        else:
            try:
                df = cot.cot_year(yr, cot_report_type="disaggregated_futures_only_long")
                if df is not None and not df.empty:
                    df.to_csv(cache, index=False)
            except Exception as e:
                print(f"  COT download failed for {yr}: {e}")
                df = pd.DataFrame()
        if not df.empty:
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Parse date
    if DATE_COL in combined.columns:
        combined["Date"] = pd.to_datetime(combined[DATE_COL], format="%y%m%d", errors="coerce")
    elif "Report_Date_as_YYYY-MM-DD" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Report_Date_as_YYYY-MM-DD"], errors="coerce")
    else:
        # Try to find any date column
        date_cols = [c for c in combined.columns if "date" in c.lower() or "Date" in c]
        if date_cols:
            combined["Date"] = pd.to_datetime(combined[date_cols[0]], errors="coerce")

    combined = combined.dropna(subset=["Date"]).sort_values("Date")
    return combined


def _get_market_data(pair: str, df_full: pd.DataFrame) -> pd.DataFrame | None:
    """Extract rows for a specific pair from the full COT DataFrame."""
    search = PAIR_TO_COT.get(pair, "")
    if not search or "Market_and_Exchange_Names" not in df_full.columns:
        return None

    mask = df_full["Market_and_Exchange_Names"].str.upper().str.contains(search.upper(), na=False)
    filtered = df_full[mask].copy()
    return filtered if not filtered.empty else None


def _calc_net_mm(df: pd.DataFrame) -> pd.Series | None:
    """Calculate Managed Money net position (longs - shorts)."""
    if MM_LONG_COL in df.columns and MM_SHORT_COL in df.columns:
        longs  = pd.to_numeric(df[MM_LONG_COL],  errors="coerce")
        shorts = pd.to_numeric(df[MM_SHORT_COL], errors="coerce")
        return (longs - shorts).values
    # Fallback: non-commercial as proxy
    if NON_COMM_L in df.columns and NON_COMM_S in df.columns:
        longs  = pd.to_numeric(df[NON_COMM_L], errors="coerce")
        shorts = pd.to_numeric(df[NON_COMM_S], errors="coerce")
        return (longs - shorts).values
    return None


def build_cot_signals(progress_callback=None) -> dict:
    """Download all COT data and build pair-level signals. Saves to data/cot/signals.json."""
    if progress_callback:
        progress_callback(0.05, "Downloading COT disaggregated data (3 years)...")

    df_full = _download_cot_data(years=3)
    if df_full is None:
        return {"error": "COT data unavailable"}

    signals: dict = {}
    pairs = list(PAIR_TO_COT.keys())

    for i, pair in enumerate(pairs):
        if progress_callback:
            progress_callback(0.1 + 0.8 * i / len(pairs), f"Processing {pair}...")

        mkt = _get_market_data(pair, df_full)
        if mkt is None or len(mkt) < 4:
            signals[pair] = {"status": "no_data"}
            continue

        mkt = mkt.sort_values("Date").tail(52)  # last year of weekly data
        net_vals = _calc_net_mm(mkt)
        if net_vals is None:
            signals[pair] = {"status": "no_mm_columns"}
            continue

        net = pd.Series(net_vals, dtype=float).dropna()
        if len(net) < 4:
            signals[pair] = {"status": "insufficient_data"}
            continue

        current   = float(net.iloc[-1])
        prev      = float(net.iloc[-2])
        ma4       = float(net.tail(4).mean())
        ma13      = float(net.tail(13).mean()) if len(net) >= 13 else ma4
        z_score   = float((current - net.mean()) / max(net.std(), 1))
        momentum  = current - prev
        trend_4w  = current - float(net.iloc[-4]) if len(net) >= 4 else 0.0

        # Signal logic
        if current > ma4 > ma13 and momentum > 0:
            direction, strength = "Bullish", min(abs(z_score) / 2, 1.0)
        elif current < ma4 < ma13 and momentum < 0:
            direction, strength = "Bearish", min(abs(z_score) / 2, 1.0)
        elif current > ma4:
            direction, strength = "Mild Bullish", 0.35
        elif current < ma4:
            direction, strength = "Mild Bearish", 0.35
        else:
            direction, strength = "Neutral", 0.0

        extreme = "Extreme long — potential reversal" if z_score >  2 else \
                  "Extreme short — potential reversal" if z_score < -2 else ""

        signals[pair] = {
            "net_mm_current":  round(current),
            "net_mm_4w_ma":    round(ma4),
            "net_mm_13w_ma":   round(ma13),
            "z_score":         round(z_score, 2),
            "momentum":        round(momentum),
            "trend_4w":        round(trend_4w),
            "direction":       direction,
            "strength":        round(strength, 3),
            "extreme_warning": extreme,
            "as_of_date":      str(mkt["Date"].iloc[-1].date()) if "Date" in mkt.columns else "?",
            "history":         [round(float(v)) for v in net.tail(13).tolist()],
        }

    signals["_updated"] = datetime.now().isoformat()

    out = os.path.join(COT_CACHE, "signals.json")
    with open(out, "w") as f:
        json.dump(signals, f, indent=2)

    if progress_callback:
        progress_callback(1.0, "COT signals built!")
    return signals


def load_cot_signals() -> dict:
    path = os.path.join(COT_CACHE, "signals.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_cot_signal(pair: str) -> dict:
    signals = load_cot_signals()
    return signals.get(pair, {"status": "not_built"})


def format_cot_for_ai(pair: str) -> str:
    sig = get_cot_signal(pair)
    if not sig or "direction" not in sig:
        return ""

    hist_str = " → ".join(f"{v:+,}" for v in sig.get("history", [])[-6:])
    extreme  = f"\n  ⚠️  {sig['extreme_warning']}" if sig.get("extreme_warning") else ""

    return f"""
COT INSTITUTIONAL POSITIONING ({pair}, as of {sig.get('as_of_date','?')}):
  Managed Money Net:  {sig['net_mm_current']:+,} contracts
  4-Week MA:          {sig['net_mm_4w_ma']:+,}  |  13-Week MA: {sig['net_mm_13w_ma']:+,}
  4-Week Trend:       {sig['trend_4w']:+,}  |  Z-Score (extremes): {sig['z_score']}
  Momentum:           {sig['momentum']:+,} (week-on-week change)
  Recent history (13w): {hist_str}
  Institutional Bias: {sig['direction'].upper()} (strength {sig['strength']:.2f}){extreme}
  Note: Managed Money = speculative institutional funds (hedge funds, CTAs).
        Extreme long/short z-scores (>±2) often precede reversals.
""".strip()

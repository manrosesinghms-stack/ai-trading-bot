"""
Downloads and caches historical forex data from Yahoo Finance.
Major pairs: EURUSD, GBPUSD, USDJPY, XAUUSD, AUDUSD, USDCAD, GBPJPY, EURJPY
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",          # Gold futures — best quality historical data
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
}

INTERVALS = {
    "1d": {"period": "5y",  "label": "Daily (5 years)"},
    "1h": {"period": "730d","label": "Hourly (2 years)"},   # yfinance 730-day limit for 1h
}


def _cache_path(pair: str, interval: str) -> str:
    return os.path.join(DATA_DIR, f"{pair}_{interval}.csv")


def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    # yfinance may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def download_all(
    intervals: list[str] = ("1d", "1h"),
    force_refresh: bool = False,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    total = len(PAIRS) * len(intervals)
    done = 0

    for interval in intervals:
        cfg = INTERVALS[interval]
        for pair, ticker in PAIRS.items():
            key = f"{pair}_{interval}"
            path = _cache_path(pair, interval)

            if os.path.exists(path) and not force_refresh:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
            else:
                try:
                    df = _download(ticker, cfg["period"], interval)
                    if not df.empty:
                        df.to_csv(path)
                except Exception as exc:
                    print(f"  Warning: failed to download {pair} {interval}: {exc}")
                    df = pd.DataFrame()

            results[key] = df
            done += 1
            if progress_callback:
                progress_callback(done / total, f"{pair} {cfg['label']}")

    return results


def load_pair(pair: str, interval: str = "1d") -> pd.DataFrame | None:
    path = _cache_path(pair, interval)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def is_data_available() -> bool:
    return any(
        os.path.exists(_cache_path(pair, "1d"))
        for pair in PAIRS
    )


def get_download_status() -> dict[str, dict]:
    status = {}
    for pair in PAIRS:
        for interval in INTERVALS:
            path = _cache_path(pair, interval)
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True, nrows=1)
                size = os.path.getsize(path) // 1024
                status[f"{pair}_{interval}"] = {
                    "exists": True, "size_kb": size,
                    "path": path,
                }
            else:
                status[f"{pair}_{interval}"] = {"exists": False}
    return status

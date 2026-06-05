"""
Walk-forward backtester — no lookahead bias.
Uses pre-calculated indicators (pandas_ta is causal so full-series calculation
is identical to rolling bar-by-bar calculation).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data_manager import load_pair

SPREAD_PIPS: dict[str, float] = {
    "EURUSD": 1.0, "GBPUSD": 1.2, "USDJPY": 1.0, "XAUUSD": 3.0,
    "AUDUSD": 1.2, "USDCAD": 1.5, "GBPJPY": 2.0, "EURJPY": 1.5,
}

PIP: dict[str, float] = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCAD": 0.0001, "USDCHF": 0.0001, "NZDUSD": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
    "XAUUSD": 0.1,
}


# ──────────────────────── vectorised signal generation ────────────────────────

def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute buy(+1) / sell(-1) / hold(0) for each strategy on every bar.
    All logic is backward-looking — no lookahead bias.
    """
    s = df.copy()

    # ── RSI ──
    rsi = s["RSI"].fillna(50)
    s["sig_rsi"] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))

    # ── MACD crossover ──
    macd_cross = np.sign(s["MACD"] - s["MACD_Signal"])
    s["sig_macd"] = macd_cross.fillna(0).astype(int)

    # ── EMA trend ──
    ema_trend = np.where(
        (s["EMA20"] > s["EMA50"]) & (s["Close"] > s["EMA200"]),  1,
        np.where(
        (s["EMA20"] < s["EMA50"]) & (s["Close"] < s["EMA200"]), -1, 0
    ))
    s["sig_ema"] = ema_trend

    # ── Bollinger Bands ──
    bb_sig = np.where(s["BB_Percent"] < 0.1, 1, np.where(s["BB_Percent"] > 0.9, -1, 0))
    s["sig_bb"] = bb_sig

    # ── ADX trend ──
    adx_sig = np.where(
        (s["ADX"] > 22) & (s["DI_Plus"]  > s["DI_Minus"]),  1,
        np.where(
        (s["ADX"] > 22) & (s["DI_Minus"] > s["DI_Plus"]),  -1, 0
    ))
    s["sig_adx"] = adx_sig

    # ── Stochastic ──
    k = s["Stoch_K"].fillna(50)
    stoch_sig = np.where(k < 20, 1, np.where(k > 80, -1, 0))
    s["sig_stoch"] = stoch_sig

    # ── Price Action (bullish/bearish engulfing) ──
    body   = s["Close"] - s["Open"]
    prev_b = body.shift(1).fillna(0)
    engulf_bull = (prev_b < 0) & (body > 0) & (s["Close"] > s["Open"].shift(1)) & (s["Open"] < s["Close"].shift(1))
    engulf_bear = (prev_b > 0) & (body < 0) & (s["Close"] < s["Open"].shift(1)) & (s["Open"] > s["Close"].shift(1))
    s["sig_pa"] = np.where(engulf_bull, 1, np.where(engulf_bear, -1, 0))

    sig_cols = ["sig_rsi","sig_macd","sig_ema","sig_bb","sig_adx","sig_stoch","sig_pa"]
    s["buy_count"]  = (s[sig_cols] == 1).sum(axis=1)
    s["sell_count"] = (s[sig_cols] == -1).sum(axis=1)

    return s


# ──────────────────────── main backtest function ───────────────────────────────

def run_backtest(
    pair: str,
    interval: str = "1d",
    initial_balance: float = 10_000.0,
    risk_pct: float = 1.0,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    min_signals: int = 4,
    warmup: int = 250,
    progress_callback=None,
) -> dict:

    df = load_pair(pair, interval)
    if df is None or len(df) < warmup + 50:
        return {"error": f"Not enough data for {pair} {interval}. Please download data first."}

    # Calculate indicators
    try:
        from strategies import calculate_all_indicators
        df = calculate_all_indicators(df).dropna()
    except Exception as e:
        return {"error": f"Indicator calculation failed: {e}"}

    df = _compute_signals(df)

    pip       = PIP.get(pair, 0.0001)
    spread_p  = SPREAD_PIPS.get(pair, 1.5)
    spread    = spread_p * pip

    balance   = initial_balance
    equity_curve: list[float] = []
    trades: list[dict] = []

    position: dict | None = None

    for i in range(warmup, len(df)):
        bar = df.iloc[i]

        # ── Position management (SL / TP) ──
        if position is not None:
            sl, tp = position["sl"], position["tp"]
            sl_hit = tp_hit = False

            if position["dir"] == 1:   # Long
                sl_hit = float(bar["Low"])  <= sl
                tp_hit = float(bar["High"]) >= tp
            else:                       # Short
                sl_hit = float(bar["High"]) >= sl
                tp_hit = float(bar["Low"])  <= tp

            if sl_hit or tp_hit:
                exit_px     = sl if sl_hit else tp
                exit_reason = "SL" if sl_hit else "TP"
                raw_pnl_pip = (exit_px - position["entry"]) / pip * position["dir"]
                net_pnl_pip = raw_pnl_pip - spread_p  # pay spread on exit too

                # R-multiple: +1R = TP hit, −1R = SL hit
                r_mult = net_pnl_pip / position["sl_pips"]
                pnl    = position["risk_usd"] * r_mult
                balance += pnl

                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date":  str(df.index[i].date()),
                    "pair":       pair,
                    "direction":  "Buy" if position["dir"] == 1 else "Sell",
                    "entry":      round(position["entry"], 5),
                    "exit":       round(exit_px, 5),
                    "sl":         round(sl, 5),
                    "tp":         round(tp, 5),
                    "pnl_pips":   round(net_pnl_pip, 1),
                    "pnl_usd":    round(pnl, 2),
                    "reason":     exit_reason,
                    "bars_held":  i - position["bar_i"],
                    "buy_sigs":   position["buy_sigs"],
                    "sell_sigs":  position["sell_sigs"],
                })
                position = None

        equity_curve.append(round(balance, 2))

        if progress_callback:
            progress_callback((i - warmup) / (len(df) - warmup))

        # ── Entry logic (only if no position open) ──
        if position is not None:
            continue

        buy_c  = int(bar["buy_count"])
        sell_c = int(bar["sell_count"])
        atr    = float(bar["ATR"])

        if atr <= 0:
            continue

        direction: int | None = None
        if buy_c >= min_signals and buy_c > sell_c:
            direction = 1
        elif sell_c >= min_signals and sell_c > buy_c:
            direction = -1

        if direction is None:
            continue

        entry_px = float(bar["Close"]) + direction * spread  # pay spread on entry
        sl_dist  = atr * sl_atr_mult
        tp_dist  = atr * tp_atr_mult
        sl_px    = entry_px - direction * sl_dist
        tp_px    = entry_px + direction * tp_dist
        sl_pips  = sl_dist / pip

        risk_usd = balance * (risk_pct / 100)

        position = {
            "dir":        direction,
            "entry":      entry_px,
            "sl":         sl_px,
            "tp":         tp_px,
            "sl_pips":    sl_pips,
            "risk_usd":   risk_usd,
            "entry_date": str(df.index[i].date()),
            "bar_i":      i,
            "buy_sigs":   buy_c,
            "sell_sigs":  sell_c,
        }

    # Close open position at last bar (mark-to-market)
    if position is not None and trades == []:
        pass  # discard incomplete position

    # ──────────────── Statistics ────────────────
    if not trades:
        return {"error": "No trades generated — try lowering min_signals or checking data."}

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["pnl_usd"] > 0]
    losses= df_t[df_t["pnl_usd"] <= 0]
    total_p = df_t["pnl_usd"].sum()

    eq = pd.Series(equity_curve)
    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max * 100
    max_dd      = float(drawdown.min())

    returns   = eq.pct_change().dropna()
    sharpe    = float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0

    win_rate  = len(wins) / len(df_t) * 100
    profit_f  = float(wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum())) if len(losses) else float("inf")
    avg_bars  = float(df_t["bars_held"].mean())

    exp_val   = (win_rate/100) * float(wins["pnl_usd"].mean() if len(wins) else 0) + \
                ((1 - win_rate/100)) * float(losses["pnl_usd"].mean() if len(losses) else 0)

    consecutive_wins   = _max_consecutive(df_t["pnl_usd"] > 0)
    consecutive_losses = _max_consecutive(df_t["pnl_usd"] <= 0)

    return {
        "pair":                pair,
        "interval":            interval,
        "initial_balance":     initial_balance,
        "final_balance":       round(balance, 2),
        "total_pnl":           round(total_p, 2),
        "total_return_pct":    round((balance / initial_balance - 1) * 100, 2),
        "total_trades":        len(df_t),
        "win_rate":            round(win_rate, 1),
        "avg_win_usd":         round(float(wins["pnl_usd"].mean()),   2) if len(wins)   else 0,
        "avg_loss_usd":        round(float(losses["pnl_usd"].mean()), 2) if len(losses) else 0,
        "profit_factor":       round(profit_f, 2),
        "max_drawdown_pct":    round(max_dd, 2),
        "sharpe_ratio":        round(sharpe, 2),
        "expected_value_usd":  round(exp_val, 2),
        "avg_bars_held":       round(avg_bars, 1),
        "max_consecutive_wins":   consecutive_wins,
        "max_consecutive_losses": consecutive_losses,
        "equity_curve":        equity_curve,
        "trades":              df_t.to_dict("records"),
    }


def _max_consecutive(mask: pd.Series) -> int:
    count = max_c = 0
    for v in mask:
        count = count + 1 if v else 0
        max_c = max(max_c, count)
    return max_c


# ──────────────── Walk-Forward Optimization ───────────────────────────────────

def run_wfo(
    pair: str,
    interval: str = "1d",
    initial_balance: float = 10_000.0,
    risk_pct: float = 1.0,
    is_bars: int = 252,        # in-sample window (bars)
    oos_bars: int = 63,        # out-of-sample window (bars)
    warmup: int = 250,
    progress_callback=None,
) -> dict:
    """
    Walk-Forward Optimization (WFO).
    Rolls a window through the dataset: optimize on IS, validate on OOS.
    Verified finding (research vote 3-0): substantially reduces overfitting
    vs single train/test split.
    Returns combined OOS equity curve and per-window statistics.
    """
    from data_manager import load_pair
    from strategies import calculate_all_indicators

    df_raw = load_pair(pair, interval)
    if df_raw is None or len(df_raw) < warmup + is_bars + oos_bars:
        return {"error": f"Not enough data for WFO ({pair} {interval})"}

    try:
        df_full = calculate_all_indicators(df_raw).dropna()
        df_full = _compute_signals(df_full)
    except Exception as e:
        return {"error": f"Indicator calculation failed: {e}"}

    # Parameter grid to optimize on IS
    GRID = [
        {"min_signals": ms, "sl_atr_mult": sl, "tp_atr_mult": tp}
        for ms in [3, 4, 5]
        for sl in [1.0, 1.5, 2.0]
        for tp in [2.0, 2.5, 3.0]
    ]

    windows: list[dict] = []
    combined_oos_trades: list[dict] = []
    combined_equity: list[float]  = [initial_balance]
    running_balance = initial_balance

    total_start = warmup
    n = len(df_full)
    window_starts = list(range(total_start, n - is_bars - oos_bars, oos_bars))

    for wi, ws in enumerate(window_starts):
        is_start = ws
        is_end   = ws + is_bars
        oos_end  = min(is_end + oos_bars, n)

        if progress_callback:
            progress_callback(wi / len(window_starts), f"WFO window {wi+1}/{len(window_starts)}")

        df_is  = df_full.iloc[is_start:is_end]
        df_oos = df_full.iloc[is_end:oos_end]

        if len(df_is) < 50 or len(df_oos) < 10:
            continue

        # ── Optimize on IS (pick params with best Sharpe) ──
        best_params = GRID[0]
        best_sharpe = -999.0
        for params in GRID:
            trades_is = _simulate_trades(
                df_is, running_balance, risk_pct,
                params["min_signals"], params["sl_atr_mult"], params["tp_atr_mult"],
                PIP.get(pair, 0.0001), SPREAD_PIPS.get(pair, 1.5),
            )
            if not trades_is:
                continue
            eq_is  = pd.Series([running_balance] + [running_balance + sum(t["pnl_usd"] for t in trades_is[:k+1])
                                                     for k in range(len(trades_is))])
            ret_is = eq_is.pct_change().dropna()
            sh = float(ret_is.mean() / ret_is.std() * (252**0.5)) if ret_is.std() > 0 else 0
            if sh > best_sharpe:
                best_sharpe, best_params = sh, params

        # ── Validate on OOS using best IS params ──
        trades_oos = _simulate_trades(
            df_oos, running_balance, risk_pct,
            best_params["min_signals"], best_params["sl_atr_mult"], best_params["tp_atr_mult"],
            PIP.get(pair, 0.0001), SPREAD_PIPS.get(pair, 1.5),
        )

        oos_pnl = sum(t["pnl_usd"] for t in trades_oos)
        running_balance += oos_pnl

        for t in trades_oos:
            combined_oos_trades.append({**t, "wfo_window": wi + 1})
            combined_equity.append(round(running_balance, 2))

        windows.append({
            "window":       wi + 1,
            "is_bars":      len(df_is),
            "oos_bars":     len(df_oos),
            "best_params":  best_params,
            "is_sharpe":    round(best_sharpe, 2),
            "oos_trades":   len(trades_oos),
            "oos_pnl":      round(oos_pnl, 2),
        })

    if not combined_oos_trades:
        return {"error": "WFO produced no OOS trades — check data length and parameters."}

    df_t    = pd.DataFrame(combined_oos_trades)
    wins    = df_t[df_t["pnl_usd"] > 0]
    losses  = df_t[df_t["pnl_usd"] <= 0]
    total_p = df_t["pnl_usd"].sum()

    eq       = pd.Series(combined_equity)
    rm       = eq.cummax()
    max_dd   = float(((eq - rm) / rm * 100).min())
    returns  = eq.pct_change().dropna()
    sharpe   = float(returns.mean() / returns.std() * (252**0.5)) if returns.std() > 0 else 0
    win_rate = len(wins) / len(df_t) * 100 if len(df_t) > 0 else 0
    pf       = float(wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum())) if len(losses) > 0 else 999

    return {
        "pair":              pair,
        "interval":          interval,
        "method":            "Walk-Forward Optimization",
        "windows":           len(windows),
        "initial_balance":   initial_balance,
        "final_balance":     round(running_balance, 2),
        "total_pnl":         round(total_p, 2),
        "total_return_pct":  round((running_balance / initial_balance - 1) * 100, 2),
        "total_trades":      len(df_t),
        "win_rate":          round(win_rate, 1),
        "profit_factor":     round(pf, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "equity_curve":      combined_equity,
        "window_stats":      windows,
        "trades":            df_t.to_dict("records"),
        "note":              "OOS equity only — each window optimized on prior IS data, validated on unseen OOS bars",
    }


def _simulate_trades(
    df: pd.DataFrame,
    start_balance: float,
    risk_pct: float,
    min_signals: int,
    sl_atr_mult: float,
    tp_atr_mult: float,
    pip: float,
    spread_pips: float,
) -> list[dict]:
    """Lightweight trade simulation for WFO inner loop."""
    spread   = spread_pips * pip
    balance  = start_balance
    position = None
    trades   = []

    for i in range(2, len(df)):
        bar = df.iloc[i]
        if position is not None:
            sl, tp  = position["sl"], position["tp"]
            sl_hit  = float(bar["Low"])  <= sl if position["dir"] == 1 else float(bar["High"]) >= sl
            tp_hit  = float(bar["High"]) >= tp if position["dir"] == 1 else float(bar["Low"])  <= tp
            if sl_hit or tp_hit:
                exit_px     = sl if sl_hit else tp
                raw_pip     = (exit_px - position["entry"]) / pip * position["dir"]
                net_pip     = raw_pip - spread_pips
                r_mult      = net_pip / position["sl_pips"]
                pnl         = position["risk_usd"] * r_mult
                balance    += pnl
                trades.append({"pnl_usd": pnl, "reason": "SL" if sl_hit else "TP"})
                position = None

        if position is not None:
            continue

        bc, sc = int(bar["buy_count"]), int(bar["sell_count"])
        atr    = float(bar["ATR"])
        if atr <= 0:
            continue

        direction = 1 if bc >= min_signals and bc > sc else (-1 if sc >= min_signals and sc > bc else 0)
        if direction == 0:
            continue

        entry  = float(bar["Close"]) + direction * spread
        sl_d   = atr * sl_atr_mult
        tp_d   = atr * tp_atr_mult
        sl_px  = entry - direction * sl_d
        tp_px  = entry + direction * tp_d
        sl_pip = sl_d / pip
        risk   = balance * (risk_pct / 100)

        position = {"dir": direction, "entry": entry, "sl": sl_px,
                    "tp": tp_px, "sl_pips": sl_pip, "risk_usd": risk}
    return trades


# ──────────────── Multi-pair summary ─────────────────────────────────────────

def run_all_pairs_summary(
    pairs: list[str],
    interval: str = "1d",
    initial_balance: float = 10_000.0,
    risk_pct: float = 1.0,
    min_signals: int = 4,
    progress_callback=None,
) -> pd.DataFrame:
    rows = []
    for i, pair in enumerate(pairs):
        if progress_callback:
            progress_callback(i / len(pairs), f"Backtesting {pair}...")
        result = run_backtest(pair, interval, initial_balance, risk_pct, min_signals=min_signals)
        if "error" not in result:
            rows.append({
                "Pair":        result["pair"],
                "Return %":    result["total_return_pct"],
                "Win Rate %":  result["win_rate"],
                "Trades":      result["total_trades"],
                "Profit Factor": result["profit_factor"],
                "Max DD %":    result["max_drawdown_pct"],
                "Sharpe":      result["sharpe_ratio"],
                "EV/trade":    result["expected_value_usd"],
            })
        else:
            rows.append({"Pair": pair, "Return %": None, "Win Rate %": None,
                          "Trades": 0, "Profit Factor": None,
                          "Max DD %": None, "Sharpe": None, "EV/trade": None})
    return pd.DataFrame(rows)

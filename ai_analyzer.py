from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import anthropic
from sessions import get_active_sessions, get_session_quality
from config import AI_MODEL

# ── Load all intelligence modules ─────────────────────────────────────────────
try:
    from knowledge_base import load_knowledge_base, format_for_ai as _kb_format
    _KB = load_knowledge_base()
except Exception:
    _KB = {}
    def _kb_format(symbol, kb): return ""

try:
    from cot_feed import format_cot_for_ai as _cot_format
except Exception:
    def _cot_format(symbol): return ""

try:
    from news_calendar import format_calendar_for_ai as _news_format, should_block_trade
except Exception:
    def _news_format(symbol): return ""
    def should_block_trade(symbol, mins=30): return False, ""

try:
    from mtf_analyzer import format_mtf_for_ai as _mtf_format
except Exception:
    def _mtf_format(symbol, tf): return ""

TRADING_KNOWLEDGE = """
You are a professional forex and financial markets trader with 15+ years of experience.
You have deep expertise in:

TRADING SESSIONS (UTC):
- Sydney: 21:00–06:00 — AUD/NZD pairs, low liquidity
- Tokyo: 23:00–08:00 — JPY pairs, Asian markets active
- London: 07:00–16:00 — highest liquidity; EUR, GBP pairs dominate
- New York: 13:00–22:00 — USD pairs, commodity currencies active
- London/NY Overlap (13:00–16:00 UTC) — peak volatility and volume; best trading window

STRATEGY KNOWLEDGE:
- Trend Following: Trade with EMA200 direction, enter on pullbacks to EMA20/50
- Breakout Trading: Trade breaks of key S/R levels with volume/momentum confirmation
- Range Trading: Buy support, sell resistance when ADX < 20 (ranging market)
- Momentum: Follow strong MACD/RSI momentum after pullback
- Reversal: Counter-trend only at extreme RSI (<25, >75) with price-action confirmation
- Confluence is king: strong signals require 4–5+ indicators agreeing

RISK MANAGEMENT RULES:
- Never risk more than 1–2% per trade
- Always use stop loss; minimum R:R = 1:2, prefer 1:3
- Do NOT trade 30min before/after major news releases
- Maximum 3–5 concurrent positions
- Low-liquidity sessions (Sydney): avoid or use very small size
- If overall drawdown > 5%: reduce position size by half

PAIR CHARACTERISTICS:
- EUR/USD: most liquid, trend-following, respects technicals well
- GBP/USD: volatile, large moves, best in London session
- USD/JPY: safe-haven, risk-on/off driven, range-breaks common
- XAU/USD (Gold): high volatility, inverse USD, spikes on news
- AUD/USD: commodity-linked, risk-sentiment driven
"""


def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def analyze_trade_opportunity(
    symbol: str,
    timeframe: str,
    strategy_results: dict,
    account_info: dict,
    market_data: dict,
) -> dict:
    """Return a structured AI trade decision."""
    client = _get_client()
    active_sessions = get_active_sessions()
    quality, quality_desc = get_session_quality()
    last_price = strategy_results.get("last_price", 0)
    atr = strategy_results.get("atr", 0)

    # ── Assemble all intelligence context ──────────────────────────────────────
    pair_knowledge = _kb_format(symbol, _KB)          # 5-year statistical KB
    cot_context    = _cot_format(symbol)               # COT institutional positioning
    news_context   = _news_format(symbol)              # Economic calendar
    mtf_context    = _mtf_format(symbol, timeframe)    # Multi-timeframe confluence

    # News trade block check
    news_blocked, news_reason = should_block_trade(symbol, 30)
    if news_blocked:
        market_data["news_block"] = news_reason

    strategy_lines = "\n".join(
        f"  • {name}: {d['signal']} (strength {d['strength']:.2f}) — {d['reason']}"
        for name, d in strategy_results.get("individual", {}).items()
    )

    news_block_warn = f"\n⛔ NEWS BLOCK ACTIVE: {market_data.get('news_block','')}" if market_data.get("news_block") else ""

    prompt = f"""
{TRADING_KNOWLEDGE}

{pair_knowledge}

{cot_context}

{mtf_context}

{news_context}
{news_block_warn}
--- ANALYSIS REQUEST ---
Symbol:    {symbol}
Timeframe: {timeframe}
Price:     {last_price:.5f}
ATR(14):   {atr:.5f}
UTC Time:  {datetime.now(timezone.utc).strftime('%H:%M')}
Session:   {', '.join(active_sessions) if active_sessions else 'None'} ({quality} quality)

Account:
  Balance:      {account_info.get('balance', 0):.2f} {account_info.get('currency', 'USD')}
  Equity:       {account_info.get('equity', 0):.2f}
  Free Margin:  {account_info.get('free_margin', 0):.2f}
  Open Trades:  {market_data.get('open_positions_count', 0)}

Technical Signals:
{strategy_lines}

Summary: {strategy_results.get('buy_count', 0)} Buy / {strategy_results.get('sell_count', 0)} Sell / {strategy_results.get('hold_count', 0)} Hold
Preliminary direction: {strategy_results.get('overall', 'Hold')}

Analyze all factors and return ONLY valid JSON (no markdown):
{{
  "action": "Buy" | "Sell" | "Hold",
  "confidence": 0.0–1.0,
  "stop_loss_pips": <integer>,
  "take_profit_pips": <integer>,
  "risk_reward": <float>,
  "session_quality": "excellent" | "good" | "fair" | "poor",
  "reasoning": "<concise professional explanation>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "warnings": ["<warning if any>"]
}}
"""

    resp = client.messages.create(
        model=AI_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = resp.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def get_market_commentary(symbol: str, strategy_results: dict, timeframe: str) -> str:
    """Return a professional 3-paragraph market commentary string."""
    client = _get_client()
    last_price = strategy_results.get("last_price", 0)
    pair_knowledge = _kb_format(symbol, _KB)

    lines = "\n".join(
        f"  • {name}: {d['signal']} — {d['reason']}"
        for name, d in strategy_results.get("individual", {}).items()
    )

    prompt = f"""
{TRADING_KNOWLEDGE}

{pair_knowledge}

Write a concise professional market analysis for {symbol} on the {timeframe} chart.
Current price: {last_price:.5f}

Technical signals:
{lines}

Format: 3 short paragraphs covering (1) trend & market structure, (2) key signals & levels, (3) trading bias & risks.
Be specific and data-driven. No generic statements. No markdown headers.
"""

    resp = client.messages.create(
        model=AI_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def answer_trading_question(question: str, context: str = "") -> str:
    """Answer a general trading question."""
    client = _get_client()

    prompt = f"""
{TRADING_KNOWLEDGE}

{f"Live context: {context}" if context else ""}

User question: {question}

Provide a clear, educational, and actionable answer with specific examples where helpful.
"""

    resp = client.messages.create(
        model=AI_MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text

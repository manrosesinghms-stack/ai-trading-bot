@echo off
echo ============================================
echo   AI Trading Bot - Launcher
echo ============================================
echo.
echo [1] VT Trend Portfolio  (the validated strategy + $500 paper account)
echo [2] Auto Paper Trader   (consensus bot, set-and-forget)
echo [3] Quick Trade Panel   (manual one-click trading)
echo [4] Full Dashboard      (charts, backtest, history)
echo.
set /p choice="Enter 1, 2, 3 or 4: "

if "%choice%"=="1" (
    python -m streamlit run vt_tool.py
) else if "%choice%"=="2" (
    python -m streamlit run auto_trader.py
) else if "%choice%"=="3" (
    python -m streamlit run quick_trade.py
) else (
    python -m streamlit run app.py
)
pause

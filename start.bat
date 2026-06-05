@echo off
echo ============================================
echo   AI Trading Bot — Launcher
echo ============================================
echo.
echo [1] Auto Paper Trader  (set and forget - starts $100, runs itself)
echo [2] Quick Trade Panel  (manual one-click trading)
echo [3] Full Dashboard     (charts, backtest, history, COT data)
echo.
set /p choice="Enter 1, 2 or 3: "

if "%choice%"=="1" (
    echo.
    echo Starting Auto Paper Trader...
    python -m streamlit run auto_trader.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Quick Trade Panel...
    python -m streamlit run quick_trade.py
) else (
    echo.
    echo Starting Full Dashboard...
    python -m streamlit run app.py
)
pause

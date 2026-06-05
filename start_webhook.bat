@echo off
echo Installing webhook dependencies...
pip install fastapi uvicorn pyngrok -q
echo.
echo Starting TradingView Webhook Server...
echo The public URL will appear below. Copy it into TradingView.
echo.
python webhook_server.py
pause

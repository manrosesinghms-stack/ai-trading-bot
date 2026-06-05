import os
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0)) if os.getenv("MT5_LOGIN") else None
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "XAUUSD"]
DEFAULT_TIMEFRAME = "H1"

MAX_RISK_PER_TRADE = 0.02
MAX_POSITIONS = 5
DEFAULT_SL_PIPS = 50
DEFAULT_TP_PIPS = 100

AI_CONFIDENCE_THRESHOLD = 0.70
AI_MODEL = "claude-sonnet-4-6"

"""Central configuration. Values can be overridden with environment variables."""
import os

# --- Database ---
DB_NAME = os.getenv("DB_NAME", "fin_data")
DB_USER = os.getenv("DB_USER", "hamzafahad")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- SEC EDGAR ---
SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "Hamza_Fahad hamzafa234@gmail.com")}

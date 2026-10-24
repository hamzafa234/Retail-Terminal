"""Central configuration. All values come from environment variables (.env)."""
import os

from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ

# --- Database ---
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]

# --- SEC EDGAR ---
SEC_HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

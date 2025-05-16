# config.py
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()  # Load variables from a .env file if available

# --- MongoDB configuration ---
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
# URL-encode username and password
USERNAME_ENC = urllib.parse.quote_plus(MONGO_USERNAME)
PASSWORD_ENC = urllib.parse.quote_plus(MONGO_PASSWORD)
CONNECTION_STRING = (
    f"mongodb+srv://{USERNAME_ENC}:{PASSWORD_ENC}@{MONGO_CLUSTER}/"
    "?"
    "retryWrites=true&w=majority&appName=Cluster0"
)


# --- Authentication configuration ---
SECRET_KEY = os.getenv("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS"))

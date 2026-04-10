# AlphaTrader-RL API Package
# ──────────────────────────────────────────────────────────────────────────────
# Central configuration: API keys, base URLs, model names.
# All secrets are loaded from .env (never hardcoded).
# ──────────────────────────────────────────────────────────────────────────────
import os

API_BASE_URL = os.environ.get("API_BASE_URL").strip().rstrip("/")
API_KEY = os.environ.get("API_KEY").strip()
MODEL_NAME = os.environ.get("MODEL_NAME").strip()

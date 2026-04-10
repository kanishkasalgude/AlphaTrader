# AlphaTrader-RL API Package
# ──────────────────────────────────────────────────────────────────────────────
# Central configuration: API keys, base URLs, model names.
# All secrets are loaded from .env (never hardcoded).
# ──────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

# Load .env locally (load_dotenv does not override existing injected eval vars by default)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co").strip().rstrip("/")
API_KEY = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "")).strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3").strip()

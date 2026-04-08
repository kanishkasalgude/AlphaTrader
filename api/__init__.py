# AlphaTrader-RL API Package
# ──────────────────────────────────────────────────────────────────────────────
# Central configuration: API keys, base URLs, model names.
# All secrets are loaded from .env (never hardcoded).
# ──────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# OpenAI-compatible config (hackathon)
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

# Backwards-compat exports (some modules may still import these names)
API_KEY = HF_TOKEN

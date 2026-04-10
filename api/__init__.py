# AlphaTrader-RL API Package
# ──────────────────────────────────────────────────────────────────────────────
# Central configuration: API keys, base URLs, model names.
# All secrets are loaded from .env (never hardcoded).
# ──────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Hugging Face Inference Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

# Backwards-compat exports (some modules may still import these names)
API_KEY = HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co")

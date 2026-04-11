import os

# DO NOT call load_dotenv() — judges inject env vars directly
# load_dotenv() would override API_KEY with local .env values

API_BASE_URL = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
API_KEY      = os.environ.get("API_KEY", "").strip()
MODEL_NAME   = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3").strip()

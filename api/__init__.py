import os
from dotenv import load_dotenv

# Load .env locally
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Use Hugging Face Token as the OpenAI API Key as mentioned in the video
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co").strip().rstrip("/")
API_KEY = os.getenv("HF_TOKEN", os.getenv("API_KEY", "")).strip()
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3").strip()

# ---------------------------------------------------------------
# AlphaTrader-RL | OpenEnv Submission
# Base: python:3.11-slim  (compatible with Hugging Face Spaces)
# ---------------------------------------------------------------
FROM python:3.11-slim

# Metadata
LABEL maintainer="Shambhavi Patil"
LABEL description="AlphaTrader-RL OpenEnv hackathon submission"
LABEL version="1.0.0"

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements_openenv.txt ./
RUN pip install --no-cache-dir -r requirements_openenv.txt

# Copy source code — core inference files
COPY trading_env.py   ./
COPY graders.py       ./
COPY inference.py     ./
COPY openenv.yaml     ./

# Copy environment module (reward.py lives here)
COPY environment/ ./environment/

# Copy data module (pipeline.py needed for auto-fetch fallback)
RUN mkdir -p data
COPY data/pipeline.py               ./data/
COPY data/processed_market_data.parquet ./data/

# Copy API module (config: API keys, base URL, model name)
COPY api/ ./api/

# Copy LLM module (OpenAI-compatible explainer)
COPY llm/ ./llm/

# Create logs directory
RUN mkdir -p logs

# Health check: verify all imports work
COPY test_imports.py ./
RUN python test_imports.py

# Keep-alive server (no UI; just binds $PORT for HF health checks)
COPY keepalive.py ./

# Run inference (CLI output), then keep Space alive by binding $PORT.
# No UI files/pages are served; this is only to satisfy Spaces health checks.
ENV OPENENV_SUMMARY_ONLY=1
CMD ["sh", "-c", "python -u inference.py && python -u keepalive.py"]
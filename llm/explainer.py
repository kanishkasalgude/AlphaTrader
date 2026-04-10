"""
<<<<<<< HEAD
AlphaTrader-RL | LLM Explainer (OpenAI-compatible)
==================================================
Uses the OpenAI Python client (`client.chat.completions.create`) against an
OpenAI-compatible endpoint to generate plain-English explanations of trading
decisions.
=======
AlphaTrader-RL | LLM Explainer (Hugging Face)
=============================================
Uses Hugging Face Inference via `huggingface_hub.InferenceClient` (no Groq / no OpenAI shim)
to generate plain-English explanations of trading decisions.
>>>>>>> ebf22a8b75bf4e291364472ac846ddb372fdcfc4

Functions
---------
explain_trade(action, features, portfolio_info)
    → 2-3 sentence explanation of a single trading decision.

explain_backtest_summary(metrics_dict)
    → paragraph summarising overall agent performance vs Buy-and-Hold.

explain_live_signal(symbol, signal, sentiment, news_summary)
    → concise rationale for a live trading signal based on sentiment.
"""
import logging
import os
from typing import Any, Dict

<<<<<<< HEAD
from openai import OpenAI
=======
from huggingface_hub import InferenceClient
>>>>>>> ebf22a8b75bf4e291364472ac846ddb372fdcfc4

logger = logging.getLogger("LLMExplainer")

# ──────────────────────────────────────────────────────────────────────────────
<<<<<<< HEAD
# OpenAI-compatible client (hackathon required pattern)
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)
=======
# Hugging Face Inference client
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
_client = InferenceClient(model=_MODEL_NAME, token=os.environ.get("HF_TOKEN", ""))
>>>>>>> ebf22a8b75bf4e291364472ac846ddb372fdcfc4

_SYSTEM_PROMPT = (
    "You are a professional financial analyst for AlphaTrader-RL. "
    "Provide clear, jargon-free explanations to investors."
)


def _call_llm(prompt: str) -> str:
    """
<<<<<<< HEAD
    Send a chat-completions request to an OpenAI-compatible endpoint.
    Returns a short "[AI Unavailable] ..." string on any error.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()
=======
    Send a text-generation request via Hugging Face Inference.
    Falls back gracefully on any error.
    """
    try:
        if not os.environ.get("HF_TOKEN"):
            return "[AI Unavailable] HF_TOKEN is not set"

        # For generic text-generation models, include system prompt inline.
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{prompt}"
        text = _client.text_generation(
            full_prompt,
            max_new_tokens=80,
            temperature=0.4,
            return_full_text=False,
        )
        return str(text).strip()
>>>>>>> ebf22a8b75bf4e291364472ac846ddb372fdcfc4
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[AI Unavailable] {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def explain_trade(
    action: str,
    features: Dict[str, Any],
    portfolio_info: Dict[str, Any],
) -> str:
    """
    Generate a 2-3 sentence explanation of a single trading decision.
    """
    feature_lines = "\n".join(
        f"  • {k}: {v:.4f}" if isinstance(v, float) else f"  • {k}: {v}"
        for k, v in features.items()
    )
    portfolio_lines = "\n".join(
        f"  • {k}: {v:.2f}" if isinstance(v, float) else f"  • {k}: {v}"
        for k, v in portfolio_info.items()
    )

    prompt = f"""The trading agent just decided to **{action}**.

Market context:
{feature_lines}

Portfolio state:
{portfolio_lines}

Explain in 2-3 clear sentences why the agent likely made this {action} decision. Focus on the relationship between the feature values and the action. Do not use bullets."""

    logger.info("explain_trade called: action=%s", action)
    return _call_llm(prompt)


def explain_backtest_summary(metrics_dict: Dict[str, Any]) -> str:
    """
    Generate a paragraph summarising the agent's overall backtest performance.
    """
    metrics_lines = "\n".join(
        f"  • {k}: {v:.3f}" if isinstance(v, float) else f"  • {k}: {v}"
        for k, v in metrics_dict.items()
    )

    alpha = metrics_dict.get("alpha_pct", 0.0)
    beat_bah = "outperformed" if alpha >= 0 else "underperformed"

    prompt = f"""Summarise the following backtest results for an RL trading agent:
{metrics_lines}

The agent {beat_bah} the benchmark by {abs(alpha):.2f}%.

Write one coherent paragraph (4-6 sentences) explaining the return vs risk profile, comparing it to Buy-and-Hold, and giving a final verdict on performance."""

    logger.info("explain_backtest_summary called: alpha_pct=%s", alpha)
    return _call_llm(prompt)


def explain_live_signal(
    symbol: str,
    signal: str,
    sentiment: float,
    news_summary: str,
) -> str:
    """
    Provide rationale for a live signal based on market sentiment.
    """
    prompt = f"""Provide a concise (1-2 sentence) rationale for a live trading signal.

Symbol: {symbol}
Signal: {signal}
Sentiment Score: {sentiment:.2f} (-1 to +1)
Recent News: {news_summary}

Why is this signal being generated now?"""

    logger.info("explain_live_signal called: symbol=%s, signal=%s", symbol, signal)
    return _call_llm(prompt)


def get_llm_explanation(
    symbol: str,
    action: str,
    rsi: float,
    macd_hist: float,
    dist_ema_50: float,
    ret: float,
) -> str | None:
    """
    Lightweight single-sentence explainer used by `inference.py`.
    Returns a short "[AI Unavailable] ..." string if AI is unavailable.
    """
    prompt = (
        "You are a trading analyst.\n"
        f"Stock: {symbol}, Action: {action}\n"
        f"RSI={rsi:.1f}, MACD={macd_hist:.3f}, EMA50={dist_ema_50:.3f}, Return={ret:.2f}%.\n"
        "Explain in ONE short sentence."
    )

    text = _call_llm(prompt)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Self-Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
<<<<<<< HEAD
    print(f"Testing OpenAI-compatible integration — base_url: {API_BASE_URL} — model: {MODEL_NAME}\n")
=======
    print(f"Testing Hugging Face integration — model: {_MODEL_NAME}\n")
>>>>>>> ebf22a8b75bf4e291364472ac846ddb372fdcfc4

    test_features = {"rsi_14": 28.5, "volume_ratio": 2.1}
    test_portfolio = {"total_return_pct": 5.2, "shares_held": 0}

    print("--- Trade Explanation Test ---")
    print(explain_trade("BUY", test_features, test_portfolio))

    print("\n--- Live Signal Test ---")
    print(explain_live_signal(
        "TATASTEEL.NS", "BUY", 0.75,
        "Strong quarterly earnings reported today."
    ))

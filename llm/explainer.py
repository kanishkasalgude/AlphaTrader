"""
AlphaTrader-RL | LLM Explainer (OpenAI-compatible Proxy)
========================================================
Uses direct HTTP requests to the LiteLLM proxy injected by OpenEnv
(via API_BASE_URL and API_KEY environment variables).

No external LLM library dependency — uses only Python stdlib (urllib).
"""
import json
import logging
import os
from typing import Any, Dict
from openai import OpenAI

logger = logging.getLogger("LLMExplainer")

_SYSTEM_PROMPT = (
    "You are a professional financial analyst for AlphaTrader-RL. "
    "Provide clear, jargon-free explanations to investors."
)


def _call_llm(prompt: str) -> str:
    """
    Send a chat-completion request using the OpenAI client.
    Uses proxy injected by OpenEnv.
    Falls back gracefully on any error.
    """
    api_base = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("API_KEY", "").strip()
    model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo").strip()

    if not api_base or not api_key:
        logger.warning("_call_llm: API_BASE_URL=%r  API_KEY set=%s",
                        api_base, bool(api_key))
        return "[AI Unavailable] API_BASE_URL or API_KEY is not set"

    try:
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
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
    model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    print(f"Testing OpenAI proxy integration — model: {model}\n")

    test_features = {"rsi_14": 28.5, "volume_ratio": 2.1}
    test_portfolio = {"total_return_pct": 5.2, "shares_held": 0}

    print("--- Trade Explanation Test ---")
    print(explain_trade("BUY", test_features, test_portfolio))

    print("\n--- Live Signal Test ---")
    print(explain_live_signal(
        "TATASTEEL.NS", "BUY", 0.75,
        "Strong quarterly earnings reported today."
    ))

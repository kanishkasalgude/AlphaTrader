import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

from llm.explainer import get_llm_explanation


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "inference_results.json"


def _read_latest_results() -> str:
    if not RESULTS_PATH.exists():
        return "No `inference_results.json` found yet."
    try:
        data: Dict[str, Any] = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)
    except Exception as exc:
        return f"Failed to read results: {exc}"


def explain(
    symbol: str,
    action: str,
    rsi: float,
    macd_hist: float,
    dist_ema_50: float,
    ret: float,
) -> str:
    text: Optional[str] = get_llm_explanation(
        symbol=symbol.strip(),
        action=action.strip().upper(),
        rsi=float(rsi),
        macd_hist=float(macd_hist),
        dist_ema_50=float(dist_ema_50),
        ret=float(ret),
    )
    if not os.environ.get("GROQ_API_KEY"):
        return "GROQ_API_KEY is not set in Space Secrets."
    if not text:
        return "LLM call failed. Check Space logs."
    return text


with gr.Blocks(title="AlphaTrader-RL") as demo:
    gr.Markdown(
        """
        ## AlphaTrader-RL — Live LLM Trade Explainer (Groq)

        This Space demonstrates a working trading explainer powered by **Groq**.
        Use the form below to generate a one-sentence rationale for a BUY/SELL decision.
        """
    )

    with gr.Tab("LLM Explainer"):
        with gr.Row():
            symbol = gr.Textbox(value="TATASTEEL.NS", label="Symbol")
            action = gr.Dropdown(choices=["BUY", "SELL"], value="BUY", label="Action")

        with gr.Row():
            rsi = gr.Number(value=42.3, label="RSI")
            macd_hist = gr.Number(value=0.012, label="MACD Histogram")
            dist_ema_50 = gr.Number(value=0.031, label="Dist to EMA50")
            ret = gr.Number(value=1.7, label="Return so far (%)")

        run_btn = gr.Button("Generate explanation")
        out = gr.Textbox(label="LLM explanation", lines=2)

        run_btn.click(
            fn=explain,
            inputs=[symbol, action, rsi, macd_hist, dist_ema_50, ret],
            outputs=[out],
        )

        gr.Markdown(
            """
            **Tip:** If you see `GROQ_API_KEY is not set`, go to Space → Settings → Secrets and add it.
            """
        )

    with gr.Tab("Latest Inference Results"):
        gr.Markdown("Shows the latest `inference_results.json` produced by the inference run at container startup.")
        results = gr.Code(value=_read_latest_results(), language="json", label="inference_results.json")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=_read_latest_results, inputs=[], outputs=[results])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)


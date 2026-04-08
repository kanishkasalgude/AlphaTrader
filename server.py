import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm.explainer import get_llm_explanation


app = FastAPI(title="AlphaTrader-RL (LLM Demo)", version="1.0.0")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "alphatrader-rl",
        "llm_provider": "groq",
        "has_groq_key": bool(os.environ.get("GROQ_API_KEY")),
    }


class ExplainRequest(BaseModel):
    symbol: str = "TATASTEEL.NS"
    action: str = Field(default="BUY", pattern="^(BUY|SELL)$")
    rsi: float = 45.0
    macd_hist: float = 0.01
    dist_ema_50: float = 0.02
    ret: float = 1.2


@app.post("/llm/explain")
async def llm_explain(req: ExplainRequest):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set in Space Secrets")

    text = get_llm_explanation(
        symbol=req.symbol,
        action=req.action,
        rsi=req.rsi,
        macd_hist=req.macd_hist,
        dist_ema_50=req.dist_ema_50,
        ret=req.ret,
    )
    if not text:
        raise HTTPException(status_code=502, detail="LLM call failed (check logs)")

    return {"explanation": text}


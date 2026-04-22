"""
FastAPI backend server — LLM Guardrail System (Group 10)

Wraps the GuardrailService from src/backend_service.py and exposes it as a
REST API so the Streamlit frontend (app/app.py) and any other client can
communicate with it over HTTP.

Endpoints:
    GET  /status            — backend health, runtime mode, LLM status
    GET  /session-summary   — aggregated session statistics
    POST /classify          — run one prompt through the guardrail pipeline

Start the backend (from the project root):
    python api/backend.py

The server will prompt for an OpenRouter API key in the terminal if
OPENROUTER_API_KEY is not already set in the environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ── Ensure project root is on sys.path ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Prompt for OpenRouter API key before anything else loads ─────────────

def _prompt_for_api_key() -> None:
    """
    Interactively ask for the OpenRouter API key on the terminal if it is not
    already present in the environment.  The key is stored in
    os.environ["OPENROUTER_API_KEY"] so that AppConfig.from_env() picks it up.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        print(f"\n✓ OPENROUTER_API_KEY detected in environment — LLM responses will be live.\n")
        return

    print("\n" + "=" * 62)
    print("   LLM Guardrail Backend  —  OpenRouter API Key Setup")
    print("=" * 62)
    print("No OPENROUTER_API_KEY found in the environment.")
    print("A free key can be obtained at: https://openrouter.ai/keys\n")
    try:
        key = input(
            "Enter your OpenRouter API key\n"
            "(press Enter to skip — runs in DeBERTa Analysis Mode): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        print("\n✓ Key accepted.  LLM responses will be live.\n")
    else:
        print("\n⚠  No key supplied — running in DeBERTa Analysis Mode (guardrail only).\n")


# ── FastAPI app ───────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="LLM Guardrail API",
    description=(
        "Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks — Group 10. "
        "Exposes the GuardrailService as a REST API consumed by the Streamlit frontend."
    ),
    version="1.0.0",
)

_service = None  # type: ignore[assignment]


@app.on_event("startup")
def _startup() -> None:
    global _service
    from src.config import AppConfig
    from src.backend_service import GuardrailService
    _service = GuardrailService(AppConfig.from_env())


# ── Request / Response schemas ────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    prompt: str
    guardrail_enabled: bool = True


class ClassifyResponse(BaseModel):
    response: str
    blocked: bool
    action: str
    mode: str
    input_decision: Optional[Dict[str, Any]] = None
    output_decision: Optional[Dict[str, Any]] = None
    effective_prompt: Optional[str] = None
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/status", summary="Backend health and runtime mode")
def status() -> dict:
    """Return the current backend mode, pipeline status, and LLM availability."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    st = _service.status
    return {
        "mode": _service.mode,
        "pipeline_ok": st.pipeline_ok,
        "pipeline_error": st.pipeline_error,
        "checkpoint_path": st.checkpoint_path,
        "llm_live": st.llm_live,
        "llm_status": st.llm_status,
    }


@app.get("/session-summary", summary="Aggregated session statistics")
def session_summary() -> dict:
    """Return cumulative statistics for the current server session."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _service.get_session_summary()


@app.post("/classify", response_model=ClassifyResponse, summary="Classify a user prompt")
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Run a user prompt through the full guardrail pipeline (regex → mDeBERTa → LLM → output guard).

    Returns the guardrail decision, optional LLM response, and per-layer metadata.
    """
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    result = _service.process_message(req.prompt, guardrail_enabled=req.guardrail_enabled)
    return ClassifyResponse(**result)


# ── Entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _prompt_for_api_key()
    import uvicorn
    uvicorn.run(
        "api.backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

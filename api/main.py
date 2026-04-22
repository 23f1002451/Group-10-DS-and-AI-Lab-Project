"""FastAPI REST service for the Guardrail Classifier pipeline.

Exposes the guardrail classification functionality as a JSON API.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment Variables:
    CHECKPOINT_PATH : Path to the trained model checkpoint (.pt file).
                      Defaults to "models/final_model.pt".
    GEMINI_API_KEY  : Optional. Required only for the LLM transformation
                      stage. Without it, suspicious prompts fall back to
                      a safe default query.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrail_pipeline import GuardrailPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "models/final_model.pt")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Guardrail Classifier API",
    description=(
        "Inference-time guardrail for detecting and mitigating prompt "
        "jailbreak attacks against Large Language Models. Classifies "
        "prompts as benign, jailbreak, or harmful and returns an action "
        "(ALLOW, TRANSFORM, or BLOCK)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global pipeline (loaded once on startup)
# ---------------------------------------------------------------------------

pipeline: GuardrailPipeline | None = None
load_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    """Load the guardrail pipeline on application startup."""
    global pipeline, load_error
    try:
        pipeline = GuardrailPipeline.from_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            enable_rule_filter=True,
            block_threshold=0.15,
            transform_threshold=0.07,
        )
    except Exception as exc:
        load_error = str(exc)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Schema for the /predict endpoint request body."""
    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to classify.",
        json_schema_extra={"example": "What is the capital of France?"},
    )


class PredictResponse(BaseModel):
    """Schema for the /predict endpoint response body."""
    action: str = Field(description="ALLOW, TRANSFORM, or BLOCK.")
    label: str = Field(description="Predicted class: benign, jailbreak, or harmful.")
    confidence: float = Field(description="Classifier confidence for the predicted label.")
    layer_triggered: str = Field(description="Pipeline layer that made the decision.")
    rule_name: str | None = Field(default=None, description="Regex rule name if triggered.")
    probabilities: dict | None = Field(default=None, description="Per-class softmax probabilities.")
    sanitized_prompt: str | None = Field(default=None, description="Sanitized text when action is TRANSFORM.")
    latency_ms: float = Field(description="Processing latency in milliseconds.")


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response body."""
    status: str
    model_loaded: bool
    checkpoint: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return the operational status of the service."""
    return HealthResponse(
        status="ok" if pipeline is not None else "error",
        model_loaded=pipeline is not None,
        checkpoint=CHECKPOINT_PATH,
        error=load_error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single prompt through the guardrail pipeline.

    Returns the action decision (ALLOW, TRANSFORM, BLOCK), the predicted
    label, confidence score, and diagnostic trace information.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {load_error}",
        )

    decision = pipeline.classify_input(request.prompt)

    return PredictResponse(
        action=decision.action.upper(),
        label=decision.label,
        confidence=decision.confidence,
        layer_triggered=decision.layer_triggered,
        rule_name=decision.rule_name,
        probabilities=decision.model_probabilities,
        sanitized_prompt=decision.sanitized_text,
        latency_ms=decision.latency_ms,
    )

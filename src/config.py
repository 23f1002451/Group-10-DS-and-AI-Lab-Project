"""
Centralized configuration management for the guardrail system.

Reads secrets/settings from (in priority order):
  1. Streamlit st.secrets  (when running on Streamlit Cloud or locally with secrets.toml)
  2. OS environment variables
  3. Hard-coded defaults

Usage:
    from src.config import AppConfig
    cfg = AppConfig.from_env()
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Lazy import — only available when Streamlit is installed and running
def _get_secret(key: str, default: str = "") -> str:
    """Read a value from Streamlit secrets, then env vars, then return default."""
    try:
        import streamlit as st
        # st.secrets raises an error if the key is absent; catch it
        val = st.secrets.get(key, None)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass
class AppConfig:
    # ── Model checkpoint ──────────────────────────────────────────────────
    checkpoint_path: str = "models/final_model.pt"
    # Optional HTTPS URL.  If set, the app will download the checkpoint when
    # models/final_model.pt is missing (useful for Streamlit Cloud where large
    # binary files cannot be committed to Git).
    model_download_url: str = ""

    # ── OpenRouter LLM ──────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"

    # ── Guardrail thresholds ──────────────────────────────────────────────
    # p_attack >= block_threshold     → BLOCK
    # p_attack >= transform_threshold → TRANSFORM (rewrite via LLM)
    # p_attack <  transform_threshold → ALLOW
    block_threshold: float = 0.8
    transform_threshold: float = 0.3
    enable_output_guardrail: bool = True
    enable_rule_filter: bool = True

    # ── Runtime ───────────────────────────────────────────────────────────
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    # ── Factory ───────────────────────────────────────────────────────────
    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Build an AppConfig by reading Streamlit secrets / environment variables.

        On Streamlit Community Cloud set these in the app's Secrets panel.
        Locally set them in .streamlit/secrets.toml or export them as env vars.
        """
        project_root = Path(__file__).resolve().parents[1]

        # Allow checkpoint path to be relative (resolved against project root)
        checkpoint_rel = _get_secret("CHECKPOINT_PATH", "models/final_model.pt")
        checkpoint_path = str(project_root / checkpoint_rel)

        return cls(
            checkpoint_path=checkpoint_path,
            model_download_url=_get_secret("MODEL_DOWNLOAD_URL", ""),
            openrouter_api_key=_get_secret("OPENROUTER_API_KEY", ""),
            openrouter_model=_get_secret("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
            block_threshold=float(_get_secret("BLOCK_THRESHOLD", "0.8")),
            transform_threshold=float(_get_secret("TRANSFORM_THRESHOLD", "0.3")),
            enable_output_guardrail=(
                _get_secret("ENABLE_OUTPUT_GUARDRAIL", "true").lower() == "true"
            ),
            enable_rule_filter=(
                _get_secret("ENABLE_RULE_FILTER", "true").lower() == "true"
            ),
            project_root=project_root,
        )

"""
LLM client abstraction layer — OpenRouter backend.

Uses the OpenAI-compatible OpenRouter API so any free (or paid) model
available at openrouter.ai can be swapped in via a single config change.

Default free model: meta-llama/llama-3.1-8b-instruct:free

Usage:
    from src.llm_client import LLMClient
    client = LLMClient(api_key="sk-or-v1-...", model="meta-llama/llama-3.1-8b-instruct:free")
    text = client.generate("Hello, world!")
"""
from __future__ import annotations

import os
from typing import Optional

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL   = "google/gemma-4-31b-it:free"


class LLMClient:
    """
    Thin wrapper around the OpenRouter API (OpenAI-compatible).

    Falls back to DeBERTa classification output when no API key is set.
    The caller always calls generate() and receives a string back.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = None
        self._model  = model or _DEFAULT_MODEL
        self._error: Optional[str] = None

        key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()

        if not key:
            self._error = (
                "No OPENROUTER_API_KEY — running in DeBERTa Analysis Mode. "
                "The mDeBERTa classification result is shown as the response."
            )
            return

        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(
                api_key=key,
                base_url=_OPENROUTER_BASE,
                default_headers={
                    "HTTP-Referer": "https://github.com/23f1002451/Group-10-DS-and-AI-Lab-Project",
                    "X-Title": "LLM Guardrail Demo",
                },
            )
        except ImportError:
            self._error = "openai package not installed. Run: pip install openai"
        except Exception as exc:
            self._error = f"OpenRouter init failed: {exc}"

    # ── Status ────────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return self._client is not None

    @property
    def status_message(self) -> str:
        if self.is_live:
            return f"OpenRouter connected · {self._model}"
        return self._error or "OpenRouter unavailable"

    # ── Core API ──────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Call OpenRouter and return the response text."""
        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                return f"[OpenRouter error: {exc}]"

        # No key — caller (backend_service) will use DeBERTa output instead
        return ""

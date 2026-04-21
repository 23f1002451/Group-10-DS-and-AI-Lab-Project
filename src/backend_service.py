"""
Backend service layer — orchestrates the guardrail pipeline and LLM client.

The Streamlit app (`app/app.py`) imports only this module.
All pipeline loading, decision logic, and LLM calls live here.

Runtime modes (selected automatically at startup):
  1. FULL MODE     — trained checkpoint + mDeBERTa (Layer 0 + Layer 1 + Layer 2)
  2. REGEX-ONLY    — no checkpoint; Layer 0 regex rules only (no API key needed)
  3. BYPASS        — user toggled guardrail OFF in the sidebar

In all modes, LLM responses are optional:
  - With GEMINI_API_KEY  → real responses from gemini-2.5-flash
  - Without              → clean demo response explaining the guardrail decision
    (the guardrail decision itself is ALWAYS shown — that is the product)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Ensure project root is on sys.path ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import AppConfig
from src.llm_client import LLMClient
from src.model_bootstrap import check_or_download


# ── Runtime mode constants ────────────────────────────────────────────────
MODE_FULL       = "full"        # trained checkpoint loaded, all 3 layers active
MODE_REGEX_ONLY = "regex_only"  # no checkpoint; Layer 0 regex rules only
MODE_BYPASS     = "bypass"      # guardrail disabled by user


# ── Status container ──────────────────────────────────────────────────────

class BackendStatus:
    """Snapshot of backend readiness — consumed by the Streamlit sidebar."""

    def __init__(self) -> None:
        self.mode: str = MODE_REGEX_ONLY
        self.pipeline_ok: bool = False
        self.pipeline_error: Optional[str] = None
        self.checkpoint_path: str = ""
        self.llm_live: bool = False
        self.llm_status: str = ""


# ── Regex-only lightweight pipeline ──────────────────────────────────────

class _RegexDecision:
    """Mimics GuardrailDecision fields so the rest of the service is uniform."""

    def __init__(self, regex_result: dict, latency_ms: float) -> None:
        cat = regex_result.get("category") or "jailbreak"
        act = regex_result["action"]

        # Map regex actions → pipeline actions
        if act == "BLOCK":
            self.action = "block"
            self.label = cat
            self.confidence = 0.95
        elif act == "SOFT_FLAG":
            # SOFT_FLAG → treat as "transform" (sanitize before LLM)
            self.action = "transform"
            self.label = cat
            self.confidence = 0.70
        else:
            self.action = "allow"
            self.label = "benign"
            self.confidence = 0.90

        self.layer_triggered = "rule_filter" if act != "ALLOW" else "none"
        self.rule_name = (
            ", ".join(regex_result.get("hits", [])) if regex_result.get("hits") else None
        )
        self.model_probabilities = None
        self.sanitized_text: Optional[str] = None
        self.latency_ms = latency_ms

    @property
    def blocked(self) -> bool:
        return self.action == "block"


class _RegexOnlyPipeline:
    """
    Layer 0 only — used when no trained checkpoint is available.

    The output guardrail is skipped entirely in this mode because we have
    no model to score the LLM response with.
    """

    INPUT_REFUSAL  = "I cannot assist with that request as it may involve unsafe or restricted content."
    OUTPUT_REFUSAL = "I cannot provide that information for safety reasons."

    def classify_input(self, prompt: str) -> _RegexDecision:
        from src.regex_filter import regex_check  # type: ignore
        start = time.perf_counter()
        result = regex_check(prompt)
        latency = round((time.perf_counter() - start) * 1000, 2)
        return _RegexDecision(result, latency)

    def classify_output(self, text: str) -> _RegexDecision:
        # Regex output guard: just run the same rules on the LLM response
        return self.classify_input(text)

    def get_summary(self) -> dict:
        return {}


# ── Main service ──────────────────────────────────────────────────────────

class GuardrailService:
    """
    Top-level backend service. Instantiate once via @st.cache_resource.

    Auto-selects runtime mode:
      • final_model.pt present  → MODE_FULL   (regex + mDeBERTa + output guard)
      • final_model.pt missing  → MODE_REGEX_ONLY  (Layer 0 regex rules only)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.status = BackendStatus()
        self.status.checkpoint_path = config.checkpoint_path

        # ── Try to load the full pipeline ─────────────────────────────────
        ok, bootstrap_msg = check_or_download(
            config.checkpoint_path, config.model_download_url
        )

        if ok:
            self._pipeline = self._load_full_pipeline()
        else:
            # Checkpoint not available — fall back to regex-only
            self._pipeline = _RegexOnlyPipeline()
            self.status.mode = MODE_REGEX_ONLY
            self.status.pipeline_ok = True   # regex always works
            self.status.pipeline_error = bootstrap_msg  # kept for sidebar info

        # ── LLM client (always optional) ──────────────────────────────────
        self._llm = LLMClient(config.openrouter_api_key, config.openrouter_model)
        self.status.llm_live = self._llm.is_live
        self.status.llm_status = self._llm.status_message

        # No Gemini wiring needed — OpenRouter handles transform rewrites via generate()

    # ── Full pipeline loader ──────────────────────────────────────────────

    def _load_full_pipeline(self):
        """Load trained GuardrailPipeline from checkpoint file."""
        try:
            from src.guardrail_pipeline import GuardrailPipeline  # type: ignore

            pipeline = GuardrailPipeline.from_checkpoint(
                checkpoint_path=self.config.checkpoint_path,
                enable_rule_filter=self.config.enable_rule_filter,
                block_threshold=self.config.block_threshold,
                transform_threshold=self.config.transform_threshold,
                enable_output_guardrail=self.config.enable_output_guardrail,
            )
            self.status.pipeline_ok = True
            self.status.mode = MODE_FULL
            return pipeline
        except Exception as exc:
            # Checkpoint exists but failed to load — still fall back to regex
            self.status.pipeline_ok = True
            self.status.mode = MODE_REGEX_ONLY
            self.status.pipeline_error = (
                f"Checkpoint load failed ({exc}). Running in regex-only mode."
            )
            return _RegexOnlyPipeline()

    # ── Public helpers ────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self.status.mode

    def process_message(
        self,
        user_prompt: str,
        guardrail_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Run one turn through the pipeline.

        Returns a dict the Streamlit frontend unpacks directly:
            response          str   – text to display
            blocked           bool  – True when guardrail blocked the turn
            action            str   – "allow" | "transform" | "block" | "bypass"
            mode              str   – runtime mode for the sidebar badge
            input_decision    dict  – pre-inference decision (or None)
            output_decision   dict  – post-inference decision (or None)
            effective_prompt  str   – sanitized prompt when action=="transform"
            llm_latency_ms    float
            total_latency_ms  float
        """
        if not guardrail_enabled:
            return self._bypass(user_prompt)

        # ── Input guardrail (Layer 0 + optional Layer 1) ──────────────────
        input_decision = self._pipeline.classify_input(user_prompt)
        input_meta = _decision_to_dict(input_decision)

        refusal_msg = (
            self._pipeline.config.input_refusal_message
            if hasattr(self._pipeline, "config")
            else _RegexOnlyPipeline.INPUT_REFUSAL
        )

        if input_decision.action == "block":
            return {
                "response": refusal_msg,
                "blocked": True,
                "action": "block",
                "mode": self.mode,
                "input_decision": input_meta,
                "output_decision": None,
                "effective_prompt": None,
                "llm_latency_ms": 0.0,
                "total_latency_ms": round(input_decision.latency_ms, 1),
            }

        # Determine the effective prompt (sanitized if transform)
        effective_prompt = (
            input_decision.sanitized_text
            if input_decision.action == "transform" and input_decision.sanitized_text
            else user_prompt
        )

        # ── LLM call (only if a live key is configured) ─────────────────
        if self._llm.is_live:
            t0 = time.perf_counter()
            llm_response = self._llm.generate(effective_prompt)
            llm_ms = round((time.perf_counter() - t0) * 1000, 1)
        else:
            # No LLM key → derive the response from the DeBERTa/regex
            # classification itself.  The guardrail analysis IS the product.
            llm_response = self._make_classification_response(
                input_decision, user_prompt, effective_prompt, self.mode
            )
            llm_ms = 0.0

        # ── Output guardrail (Layer 2 — skipped in regex-only mode
        #    when no Gemini key is present, because regexes on LLM output
        #    generate too many false positives on benign chatty text)
        if self.mode == MODE_FULL:
            output_decision = self._pipeline.classify_output(llm_response)
            output_meta = _decision_to_dict(output_decision)
        else:
            # Regex-only: skip output guard to avoid FP on chatty responses
            output_meta = {
                "action": "allow", "label": "benign", "confidence": 1.0,
                "layer_triggered": "none", "rule_name": None,
                "probabilities": None, "latency_ms": 0.0, "sanitized_text": None,
            }
            output_decision = None

        output_refusal = (
            self._pipeline.config.output_refusal_message
            if hasattr(self._pipeline, "config")
            else _RegexOnlyPipeline.OUTPUT_REFUSAL
        )

        if output_meta["action"] == "block":
            response_text = output_refusal
            blocked = True
        else:
            response_text = llm_response
            blocked = False

        output_latency = output_meta.get("latency_ms", 0.0)
        total_ms = round(input_decision.latency_ms + llm_ms + output_latency, 1)

        return {
            "response": response_text,
            "blocked": blocked,
            "action": input_decision.action,
            "mode": self.mode,
            "input_decision": input_meta,
            "output_decision": output_meta,
            "effective_prompt": (
                effective_prompt if input_decision.action == "transform" else None
            ),
            "llm_latency_ms": llm_ms,
            "total_latency_ms": total_ms,
        }

    @staticmethod
    def _make_classification_response(
        decision: Any,
        user_prompt: str,
        effective_prompt: str,
        mode: str,
    ) -> str:
        """
        Build a markdown response derived entirely from the mDeBERTa / regex
        classification result.  This is used whenever no Gemini API key is
        configured — the guardrail analysis *is* the response (which is the
        actual product of this project).
        """
        label      = decision.label
        confidence = decision.confidence
        action     = decision.action
        probs      = decision.model_probabilities   # dict | None
        layer      = decision.layer_triggered
        rule       = decision.rule_name

        label_emoji = {"benign": "✅", "jailbreak": "⚠️", "harmful": "🚫"}.get(label, "ℹ️")
        action_line = {
            "allow":     "Prompt classified as **safe** — no intervention required.",
            "transform": "Prompt flagged; forwarded with sanitized text.",
            "block":     "Prompt rejected by the guardrail.",
        }.get(action, action.upper())

        lines = [
            f"### {label_emoji} Guardrail Decision: {action.upper()}",
            "",
            f"**Verdict:** {action_line}",
            f"**Predicted class:** `{label}` ({confidence:.1%} confidence)",
            f"**Layer triggered:** `{layer}`",
        ]

        if rule:
            lines.append(f"**Rule matched:** `{rule}`")

        if probs:
            lines += [
                "",
                "**Class probabilities (mDeBERTa fine-tuned classifier):**",
                "",
                "| Class | Probability | Bar |",
                "|-------|-------------|-----|",
            ]
            for cls, p in probs.items():
                bar = "█" * max(1, int(p * 20))
                lines.append(f"| `{cls}` | {p:.1%} | {bar} |")
        elif mode == "regex_only":
            lines += [
                "",
                "> *Neural classifier not loaded — train the model or place*",
                "> *`final_model.pt` in `models/` to see per-class probabilities.*",
            ]

        mode_note = {
            "full":       "`microsoft/mdeberta-v3-base` fine-tuned (Layer 0 regex + Layer 1 mDeBERTa + Layer 2 output guard)",
            "regex_only": "Regex rule engine — Layer 0 only (train the model to enable the neural classifier)",
        }.get(mode, mode)

        lines += [
            "",
            "---",
            f"*Analysis engine: {mode_note}*",
        ]
        return "\n".join(lines)

    def _bypass(self, user_prompt: str) -> Dict[str, Any]:
        """Guardrail disabled — call LLM directly (or show notice)."""
        if self._llm.is_live:
            t0 = time.perf_counter()
            llm_response = self._llm.generate(user_prompt)
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
        else:
            llm_response = (
                "### ⚡ Guardrail Bypassed\n\n"
                "The guardrail is currently **disabled**.  No classification was run.\n\n"
                "Enable the guardrail toggle in the sidebar to see the mDeBERTa analysis."
            )
            total_ms = 0.0
        return {
            "response": llm_response,
            "blocked": False,
            "action": "bypass",
            "mode": MODE_BYPASS,
            "input_decision": None,
            "output_decision": None,
            "effective_prompt": None,
            "llm_latency_ms": total_ms,
            "total_latency_ms": total_ms,
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Cumulative session statistics (from pipeline decision log)."""
        if self.mode != MODE_FULL:
            return {}
        return self._pipeline.get_summary()


# ── Helpers ───────────────────────────────────────────────────────────────

def _decision_to_dict(decision: Any) -> Dict[str, Any]:
    """Convert a GuardrailDecision (or _RegexDecision) to a plain dict."""
    return {
        "action": decision.action,
        "label": decision.label,
        "confidence": round(decision.confidence, 4),
        "layer_triggered": decision.layer_triggered,
        "rule_name": decision.rule_name,
        "probabilities": decision.model_probabilities,
        "latency_ms": round(decision.latency_ms, 1),
        "sanitized_text": decision.sanitized_text,
    }

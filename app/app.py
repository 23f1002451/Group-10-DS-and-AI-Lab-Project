"""Streamlit Chat Interface with Guardrail Protection.

A single-turn chat assistant that demonstrates the dual-stage guardrail system.
Users can toggle the guardrail on/off to compare protected vs unprotected responses.

Usage:
    streamlit run app/app.py -- --checkpoint models/final_model.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Streamlit page config ────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Guardrail Demo",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 900px; margin: 0 auto; }
    .status-allow { color: #22c55e; font-weight: bold; }
    .status-block { color: #ef4444; font-weight: bold; }
    .status-transform { color: #f59e0b; font-weight: bold; }
    .metric-box {
        background: #1e293b; border-radius: 8px; padding: 12px;
        margin: 4px 0; border-left: 4px solid #3b82f6;
    }
    .metric-box h4 { margin: 0; color: #94a3b8; font-size: 0.8em; }
    .metric-box p { margin: 0; color: #f1f5f9; font-size: 1.2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Arg parsing (for --checkpoint) ────────────────────────────────────
def get_checkpoint_path() -> str:
    """Get checkpoint path from CLI args or default."""
    # Streamlit passes args after '--'
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/final_model.pt")
    parser.add_argument("--gemini-api-key", type=str, default="")
    try:
        args, _ = parser.parse_known_args()
        return args.checkpoint, args.gemini_api_key
    except SystemExit:
        return "outputs/best_model.pt", ""


CHECKPOINT_PATH, GEMINI_KEY = get_checkpoint_path()


# ── Load pipeline (cached) ───────────────────────────────────────────
@st.cache_resource
def load_pipeline(checkpoint_path: str):
    """Load the guardrail pipeline model (cached across reruns)."""
    from src.guardrail_pipeline import GuardrailPipeline
    try:
        pipeline = GuardrailPipeline.from_checkpoint(checkpoint_path)
        return pipeline, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_gemini(api_key: str):
    """Load the Gemini API client."""
    try:
        import google.generativeai as genai
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            return None, "No API key. Set GEMINI_API_KEY env var or pass --gemini-api-key."
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model, None
    except Exception as e:
        return None, str(e)


# ── Main UI ───────────────────────────────────────────────────────────

st.title("🛡️ LLM Guardrail Demo")
st.caption("Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks — Group 10")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    guardrail_enabled = st.toggle("Enable Guardrail", value=True)
    show_details = st.toggle("Show Technical Details", value=True)

    st.divider()
    st.subheader("Model Info")

    pipeline, load_error = load_pipeline(CHECKPOINT_PATH)
    if load_error:
        st.error(f"❌ Pipeline load error:\n{load_error}")
        st.info("Run training first:\n`python src/train.py --train-data data/small/train.json --val-data data/small/validation.json --output-dir models`")
    else:
        st.success("✅ Classifier loaded")
        st.caption(f"Checkpoint: `{CHECKPOINT_PATH}`")

    gemini_model, gemini_error = load_gemini(GEMINI_KEY)
    if gemini_error:
        st.warning(f"⚠️ Gemini: {gemini_error}")
        st.caption("Responses will be simulated.")
    else:
        st.success("✅ Gemini API connected")

    st.divider()
    st.subheader("Architecture")
    st.markdown("""
    ```
    User Prompt
      ↓
    Rule Filter (regex)
      ↓
    mDeBERTa Classifier
      ↓ allow / transform / block
    Gemini API
      ↓
    Output Classifier
      ↓
    Response
    ```
    """)


def call_llm(prompt: str) -> str:
    """Call Gemini API or return a simulated response."""
    if gemini_model is not None:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[Gemini API error: {e}]"
    else:
        return (
            f"[Simulated LLM response to: '{prompt[:60]}...']\n\n"
            "This is a placeholder response because no Gemini API key is configured. "
            "Set the GEMINI_API_KEY environment variable to get real responses."
        )


# ── Chat Interface ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            with st.expander("🔍 Guardrail Details"):
                st.json(msg["meta"])

# Chat input
if prompt := st.chat_input("Enter your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with guardrail
    with st.chat_message("assistant"):
        if not pipeline:
            st.error("Guardrail model not loaded. Please check the sidebar.")
        elif not guardrail_enabled:
            # Bypass guardrail
            with st.spinner("Generating response (guardrail OFF)..."):
                t0 = time.perf_counter()
                response = call_llm(prompt)
                total_ms = (time.perf_counter() - t0) * 1000

            st.markdown(response)
            meta = {"guardrail": "disabled", "total_latency_ms": round(total_ms, 1)}
            if show_details:
                with st.expander("🔍 Details"):
                    st.json(meta)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "meta": meta}
            )
        else:
            # ── Layer 0 + Layer 1: Input guardrail ──
            input_decision = pipeline.classify_input(prompt)

            if input_decision.action == "block":
                # Blocked by input guardrail
                response = pipeline.config.input_refusal_message

                if show_details:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Action", "🚫 BLOCKED")
                    col2.metric("Label", input_decision.label.upper())
                    col3.metric("Latency", f"{input_decision.latency_ms:.1f}ms")

                st.error(response)
                meta = {
                    "action": "block",
                    "label": input_decision.label,
                    "confidence": input_decision.confidence,
                    "layer": input_decision.layer_triggered,
                    "rule": input_decision.rule_name,
                    "probabilities": input_decision.model_probabilities,
                    "input_latency_ms": input_decision.latency_ms,
                }
                if show_details:
                    with st.expander("🔍 Guardrail Details"):
                        st.json(meta)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "meta": meta}
                )

            else:
                # Allow or Transform → call LLM
                effective_prompt = (
                    input_decision.sanitized_text
                    if input_decision.action == "transform" and input_decision.sanitized_text
                    else prompt
                )

                if input_decision.action == "transform" and show_details:
                    st.info(
                        f"⚠️ Prompt was **sanitized** before forwarding to LLM.\n\n"
                        f"**Original**: {prompt[:120]}...\n\n"
                        f"**Sanitized**: {effective_prompt[:120]}..."
                    )

                with st.spinner("Querying LLM..."):
                    t0 = time.perf_counter()
                    llm_response = call_llm(effective_prompt)
                    llm_ms = (time.perf_counter() - t0) * 1000

                # ── Layer 2: Output guardrail ──
                output_decision = pipeline.classify_output(llm_response)

                if output_decision.action == "block":
                    response = pipeline.config.output_refusal_message
                    st.warning(response)
                    layer_info = "output_guardrail"
                else:
                    response = llm_response
                    st.markdown(response)
                    layer_info = "none"

                total_ms = input_decision.latency_ms + llm_ms + output_decision.latency_ms
                meta = {
                    "input_action": input_decision.action,
                    "input_label": input_decision.label,
                    "input_confidence": input_decision.confidence,
                    "input_layer": input_decision.layer_triggered,
                    "output_action": output_decision.action,
                    "output_label": output_decision.label,
                    "output_confidence": output_decision.confidence,
                    "probabilities": input_decision.model_probabilities,
                    "input_latency_ms": round(input_decision.latency_ms, 1),
                    "llm_latency_ms": round(llm_ms, 1),
                    "output_latency_ms": round(output_decision.latency_ms, 1),
                    "total_latency_ms": round(total_ms, 1),
                    "guardrail_overhead_ms": round(
                        input_decision.latency_ms + output_decision.latency_ms, 1
                    ),
                }

                if show_details:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Input", input_decision.action.upper())
                    col2.metric("Output", output_decision.action.upper())
                    col3.metric("Guardrail Overhead",
                                f"{input_decision.latency_ms + output_decision.latency_ms:.0f}ms")
                    col4.metric("Total Latency", f"{total_ms:.0f}ms")

                    with st.expander("🔍 Guardrail Details"):
                        st.json(meta)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "meta": meta}
                )

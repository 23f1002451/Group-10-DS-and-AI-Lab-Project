"""
Streamlit Chat Interface — LLM Guardrail Demo (Group 10)

The guardrail decision (label, confidence, layer, probabilities) is the
PRODUCT.  It is always shown regardless of whether an OpenRouter API key exists.

Architecture:
    Frontend  →  app/app.py           (this file — Streamlit UI)
    Backend   →  api/backend.py       (FastAPI REST server on :8000)

Start order:
    1. python api/backend.py          # prompts for OpenRouter key, starts on :8000
    2. streamlit run app/app.py       # starts the Streamlit UI on :8501

The BACKEND_URL environment variable overrides the default http://localhost:8000
if the backend is running on a different host/port.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Backend URL config ────────────────────────────────────────────────────
_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


# ── Page config (must be the very first Streamlit call) ───────────────────
st.set_page_config(
    page_title="LLM Guardrail Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────
st.markdown(
    "<style>.main{max-width:920px;margin:0 auto}</style>",
    unsafe_allow_html=True,
)


# ── API helpers (cached once per process) ─────────────────────────────────

@st.cache_resource(show_spinner="⏳ Connecting to guardrail backend…")
def _get_backend_status() -> dict:
    """Fetch /status from the FastAPI backend.  Raises on connection failure."""
    try:
        resp = requests.get(f"{_BACKEND_URL}/status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot reach the backend at **{_BACKEND_URL}**.\n\n"
            "Start the backend first:\n```\npython api/backend.py\n```"
        )
        st.stop()
    except Exception as exc:
        st.error(f"❌ Backend error: {exc}")
        st.stop()


def _classify(prompt: str, guardrail_enabled: bool) -> dict:
    """POST /classify and return the result dict."""
    resp = requests.post(
        f"{_BACKEND_URL}/classify",
        json={"prompt": prompt, "guardrail_enabled": guardrail_enabled},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _session_summary() -> dict:
    """GET /session-summary and return the stats dict."""
    try:
        resp = requests.get(f"{_BACKEND_URL}/session-summary", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# ── Connect to backend ────────────────────────────────────────────────────
backend = _get_backend_status()
_mode         = backend.get("mode", "regex_only")
_pipeline_ok  = backend.get("pipeline_ok", False)
_pipeline_err = backend.get("pipeline_error")
_llm_live     = backend.get("llm_live", False)
_llm_status   = backend.get("llm_status", "")
_ckpt_path    = backend.get("checkpoint_path", "")


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuration")
    guardrail_enabled = st.toggle("Enable Guardrail", value=True)
    show_details = st.toggle("Show Technical Details", value=True)

    st.divider()
    st.subheader("🔧 Guardrail Mode")

    if _mode == "full":
        st.success("✅ **FULL MODE** — mDeBERTa + regex (all 3 layers)")
        st.caption(f"Checkpoint: `{Path(_ckpt_path).name}`")
    elif _mode == "regex_only":
        st.info("ℹ️ **REGEX-ONLY MODE** — Layer 0 rules (no checkpoint)")
        st.caption("Train the model and place `final_model.pt` in `models/` to enable the neural classifier.")
        if _pipeline_err:
            with st.expander("Why? (expand for details)"):
                st.code(_pipeline_err, language="text")

    st.divider()
    st.subheader("🤖 Response Mode")
    if _llm_live:
        st.success(f"✅ {_llm_status}")
        st.caption("OpenRouter LLM responds after guardrail passes the prompt.")
    else:
        st.info("📊 **DeBERTa Analysis Mode**")
        st.caption(
            "No OpenRouter key — the mDeBERTa classification result "
            "**is** the response. Label, confidence, and per-class probabilities "
            "are shown for every prompt.  This is the actual product."
        )

    if _mode == "regex_only":
        st.divider()
        st.subheader("🏋️ Train Neural Classifier")
        st.caption(
            "No `final_model.pt` found in `models/`.  "
            "Click below to fine-tune mDeBERTa on `data/small/` (~10 min on CPU)."
        )
        if st.button("▶ Train Now", use_container_width=True):
            import subprocess
            st.info("Training started… watch the terminal for progress.")
            try:
                subprocess.Popen(
                    [
                        sys.executable, "src/train.py",
                        "--train-data", "data/small/train.json",
                        "--val-data",   "data/small/validation.json",
                        "--output-dir", "models",
                        "--epochs", "5",
                    ],
                    cwd=str(_PROJECT_ROOT),
                )
                st.success(
                    "Training process launched.  "
                    "When it finishes, `models/final_model.pt` will appear and "
                    "you can restart the app for Full mDeBERTa mode."
                )
            except Exception as e:
                st.error(f"Failed to launch training: {e}")

    st.divider()
    st.markdown("""
```
User Prompt
    ↓
Layer 0: Rule Filter (regex)
    ↓
Layer 1: mDeBERTa Classifier
    ↓  allow / transform / block
Gemini API  ← optional
    ↓
Layer 2: Output Guardrail
    ↓
Response
```
""")

    summary = _session_summary()
    if summary.get("total", 0) > 0:
        st.divider()
        st.subheader("📊 Session Stats")
        c1, c2 = st.columns(2)
        c1.metric("Total",       summary["total"])
        c2.metric("Blocked",     summary.get("blocked", 0))
        c1.metric("Transformed", summary.get("transformed", 0))
        c2.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")

    st.divider()
    with st.expander("ℹ️ About"):
        st.markdown("""
**Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks**
Group 10 — DS and AI Lab Project
- Backbone: `microsoft/mdeberta-v3-base` (86 M params)
- Classes: `benign` · `jailbreak` · `harmful`
- Target: ASR ↓ ≥70% · FRR < 10% · Overhead < 300 ms
""")


# ══════════════════════════════════════════════════════════════════════════
# GUARDRAIL DECISION RENDERER
# ══════════════════════════════════════════════════════════════════════════

def _render_decision(result: dict) -> None:
    """Display guardrail decision details for one turn."""
    inp = result.get("input_decision")
    out = result.get("output_decision")
    action = result.get("action", "allow")

    if not inp:
        st.caption(f"Guardrail bypassed — {result.get('total_latency_ms', 0):.0f} ms")
        return

    with st.expander("🔍 Guardrail Decision Details"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Input Action",       action.upper())
        col2.metric("Label",              inp.get("label", "—").upper())
        col3.metric("Confidence",         f"{inp.get('confidence', 0):.1%}")
        overhead = inp.get("latency_ms", 0) + (out or {}).get("latency_ms", 0)
        col4.metric("Guardrail Overhead", f"{overhead:.0f} ms")

        if inp.get("rule_name"):
            st.markdown(f"**Rule matched:** `{inp['rule_name']}`")

        if inp.get("probabilities"):
            st.markdown("**Class probabilities (mDeBERTa):**")
            probs = inp["probabilities"]
            p_cols = st.columns(len(probs))
            for i, (lbl, val) in enumerate(probs.items()):
                p_cols[i].metric(lbl.capitalize(), f"{val:.1%}")

        if out and out.get("action") == "block":
            st.warning("⚠️ Output guardrail (Layer 2) also triggered on the LLM response.")

        mode_map = {
            "full": "Full (regex + mDeBERTa)",
            "regex_only": "Regex-Only (Layer 0)",
            "bypass": "Bypassed",
        }
        st.caption(
            f"Mode: {mode_map.get(result.get('mode',''), result.get('mode',''))} "
            f"| Total latency: {result.get('total_latency_ms', 0):.0f} ms"
        )


# ══════════════════════════════════════════════════════════════════════════
# MAIN CHAT UI
# ══════════════════════════════════════════════════════════════════════════

st.title("🛡️ LLM Guardrail Demo")
st.caption("Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks — Group 10")

if not guardrail_enabled:
    st.warning("⚠️ Guardrail **disabled** — prompts go directly to the LLM.")
elif _mode == "regex_only":
    st.info(
        "📊 **Analysis Mode** (Regex Layer 0) — the guardrail verdict is the response.\n\n"
        "Try *'ignore all previous instructions'* or *'how to build a bomb'* → **BLOCK**.\n"
        "Try *'What is the capital of France?'* → **ALLOW** with class probabilities.\n\n"
        "Place a trained `final_model.pt` in `models/` to enable the full mDeBERTa classifier."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_details and msg.get("meta"):
            _render_decision(msg["meta"])

if user_input := st.chat_input("Type a message — try benign, jailbreak, or harmful prompts"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processing through guardrail…"):
            try:
                result = _classify(user_input, guardrail_enabled=guardrail_enabled)
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Lost connection to backend at **{_BACKEND_URL}**. "
                    "Is `python api/backend.py` still running?"
                )
                st.stop()
            except Exception as exc:
                st.error(f"Backend error: {exc}")
                st.stop()

        action = result.get("action", "allow")

        if result["blocked"]:
            st.error(result["response"])
        elif action == "transform" and result.get("effective_prompt"):
            st.info(
                f"⚠️ Prompt **sanitized** before forwarding to LLM.\n\n"
                f"**Original:** {user_input[:160]}\n\n"
                f"**Sanitized:** {result['effective_prompt'][:160]}"
            )
            st.markdown(result["response"])
        else:
            st.markdown(result["response"])

        if show_details:
            _render_decision(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "meta": result,
    })

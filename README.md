# Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks

**Group 10 — DS and AI Lab Project**

## Overview

This repository implements a **dual-stage inference-time guardrail system** that protects LLM chat assistants from adversarial prompt manipulation. The system acts as middleware between user input and the Gemini API, performing:

1. **Pre-Inference Guardrail** — Classifies user prompts as `benign`, `jailbreak`, or `harmful` using a fine-tuned `mDeBERTa-v3-base` classifier. Suspicious prompts are sanitized (adversarial meta-instructions are stripped); harmful prompts are blocked outright.
2. **Post-Inference Guardrail** — Validates LLM-generated responses through the same classifier to catch harmful content that bypassed the input layer.

### Architecture

```
User Prompt → Rule Filter (regex) → mDeBERTa Classifier → [Transform/Block/Allow] → Gemini API → Output Classifier → Response
```

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **ASR** (Attack Success Rate) | ≥ 70% reduction vs baseline | % of attacks that bypass the guardrail |
| **FRR** (False Refusal Rate) | < 10% | % of benign prompts incorrectly blocked |
| **Task Utility** | ≥ 90% of baseline (MT-Bench) | Performance preservation |
| **Latency** | < 300ms overhead | Total guardrail processing time |

## Project Structure

```
├── app/                 # Web interface application
│   └── app.py           # Streamlit chat interface demo
├── api/                 # API services (if any)
├── data/                # Dataset structure
│   ├── data.py          # Data ingestion, normalization, split generation
│   └── dataset_outputs/ # Processed JSON/CSV splits (train/val/test)
├── docs/                # All technical documentation
│   ├── overview.md
│   ├── technical_doc.md
│   ├── user_guide.md
│   ├── api_doc.md
│   └── licenses.md
├── models/              # Saved models or weights (.pt files)
├── notebooks/           # Training & evaluation notebooks
├── src/                 # Model & pipeline code
│   ├── guardrail_classifier.py  # mDeBERTa encoder + linear head
│   ├── regex_filter.py          # Rule-based preprocessing
│   ├── guardrail_pipeline.py    # Dual-stage inference pipeline
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Evaluation suite (ASR, FRR, latency)
│   └── run_e2e_subset.py        # End-to-end pipeline verification
├── requirements.txt
└── README.md            # Executive summary + demo links
```

## Setup & Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run End-to-End Pipeline (Full Training + Evaluation)

```bash
python src/run_e2e_subset.py --epochs 5
```

This trains on the full dataset, evaluates on val/test splits, and runs pipeline integration tests.

### 2. Smoke Test (Structural Verification Only)

```bash
python src/run_e2e_subset.py --smoke-mode --epochs 1 --train-size 60 --val-size 20 --test-size 20
```

### 3. Train Only

```bash
python src/train.py \
  --train-data data/small/train.json \
  --val-data data/small/validation.json \
  --output-dir models \
  --epochs 5
```

### 4. Evaluate a Checkpoint

```bash
python src/evaluate.py \
  --checkpoint models/final_model.pt \
  --dataset data/small/test.json \
  --output-metrics eval_outputs/test_metrics.json \
  --output-samples eval_outputs/test_samples.json
```

### 5. Launch the Application (Backend + Frontend)

The application is split into a **FastAPI backend** (`api/backend.py`) and a **Streamlit frontend** (`app/app.py`).  Start them in two separate terminals:

**Terminal 1 — Backend**
```bash
python api/backend.py
```
> When the backend starts it will prompt you for your **OpenRouter API key** if `OPENROUTER_API_KEY` is not already set in the environment.  
> A free key can be obtained at <https://openrouter.ai/keys>.  
> Press Enter without a key to run in DeBERTa Analysis Mode (guardrail-only, no LLM responses).

**Terminal 2 — Frontend**
```bash
streamlit run app/app.py
```
> The Streamlit UI starts on **http://localhost:8501** and connects automatically to the backend on **http://localhost:8000**.  
> To point the frontend at a different backend host, set the `BACKEND_URL` environment variable:
> ```bash
> set BACKEND_URL=http://your-backend-host:8000   # Windows
> export BACKEND_URL=http://your-backend-host:8000 # Linux/macOS
> ```

## Dataset

- **1500 prompts** across 3 classes: Benign (670), Jailbreak (455), Harmful (375)
- Sources: JailbreakBench, TrustAIRLab, LMSYS Toxic Chat, SQuAD v2, Alpaca Cleaned
- Split: 70% Train / 15% Validation / 15% Test with family-grouped leakage prevention
- Full details in `docs/milestone_2/`

## Model Architecture

- **Backbone**: `microsoft/mdeberta-v3-base` (86M parameters, encoder-only)
- **Head**: Dropout(0.2) → Linear(768 → 3) classification
- **Training**: AdamW, lr=3e-5, class-weighted CrossEntropyLoss, linear warmup
- **Inference**: Mean pooling → 3-class softmax → threshold decision

## References

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications)
- [Meta PromptGuard](https://huggingface.co/meta-llama/Prompt-Guard-86M)
- [JailbreakBench](https://jailbreakbench.github.io)
- [XSTest](https://huggingface.co/datasets/xstest)

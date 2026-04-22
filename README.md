# Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks

**Group 10 — DS and AI Lab Project**

## Project Overview

This repository implements a hybrid defense-in-depth guardrail system that protects Large Language Model chat assistants from adversarial prompt manipulation. The system operates as inference-time middleware between user input and the Gemini API, classifying incoming prompts as benign, jailbreak, or harmful, and taking appropriate action (allow, transform, or block) based on empirically calibrated thresholds.

### Key Features

1. Four-layer hybrid pipeline combining deterministic regex filtering, neural classification, threshold-based decision logic, and LLM-powered prompt transformation.
2. Fine-tuned mDeBERTa-v3-base classifier with attention-mask-aware mean pooling and head-tail truncation for robust adversarial intent detection.
3. Composite-score-optimized threshold calibration balancing safety (ASR) against usability (FRR).
4. Interactive demo on Hugging Face Spaces with side-by-side protected vs. unprotected comparison.
5. REST API for programmatic access to the classification pipeline.

### Final Performance (Test Set, N=3,027)

| Metric | Value |
|:---|:---|
| Macro F1 | 0.9411 |
| Attack Success Rate (Overall) | 1.77% |
| False Refusal Rate | 3.94% |
| Composite Score | 0.9656 |
| Mean Latency | 5.84ms (T4 GPU) |

### Architecture

```
User Prompt
    |
Layer 0: Regex Pre-Filter (11 severity-weighted rules, <1ms)
    |
Layer 1: mDeBERTa-v3-base Classifier (mean pooling, ~5.8ms)
    |
Layer 2: Threshold Decision Engine
    |-- p_attack >= 0.15  -->  BLOCK
    |-- 0.07 <= p_attack < 0.15  -->  TRANSFORM (Gemini rewrite)
    |-- p_attack < 0.07  -->  ALLOW
    |
LLM (Gemini API)
    |
Layer 3: Output Guardrail (same classifier on response)
    |
Response to User
```

## Repository Structure

```
project/
    app/                    Streamlit chat interface
        app.py              Single-turn demo with guardrail visualization
    api/                    FastAPI REST service
        main.py             /predict and /health endpoints
        requirements.txt    API-specific dependencies
    data/                   Dataset structure (train/val/test splits)
    docs/                   Technical documentation and milestone reports
        overview.md         Project summary and architecture
        technical_doc.md    Comprehensive technical reference
        user_guide.md       End-user interaction guide
        api_doc.md          REST API documentation
        problem_statement.md  Problem definition and scope
        licenses.md         Code, dataset, and model licenses
        milestone_1/ to milestone_6/  Milestone reports
    models/                 Saved model checkpoints (.pt files)
    notebooks/              Training and evaluation notebooks
        Final Classifier.ipynb      Production training pipeline
        mdeberta HPT.ipynb          Hyperparameter tuning
        multiple models HPT.ipynb   Cross-model comparison
        notebooks guide.md          Notebook descriptions and results
    src/                    Model and pipeline source code
        guardrail_classifier.py     mDeBERTa encoder with mean pooling
        regex_filter.py             Rule-based pre-filter
        guardrail_pipeline.py       Dual-stage inference pipeline
        train.py                    Training loop
        evaluate.py                 Evaluation suite
        run_e2e_subset.py           E2E verification
    requirements.txt        Project-wide dependencies
    LICENSE                 MIT License
```

## Environment Setup

### Requirements

1. Python 3.10 or later.
2. NVIDIA GPU with CUDA support recommended for training. CPU inference is supported with higher latency (20 to 50ms).
3. 16GB GPU VRAM minimum for training (T4 or equivalent).

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Group-10-DS-and-AI-Lab-Project.git
cd Group-10-DS-and-AI-Lab-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The system requires one environment variable for full functionality:

```bash
# Required for the LLM transformation stage
export GEMINI_API_KEY="your-api-key-here"
```

Without the Gemini API key, the transformation stage falls back to a safe default query. The classification pipeline remains fully functional without it.

## Reproducing Results (Kaggle Notebooks)

The authoritative training and evaluation pipeline is implemented in Kaggle notebooks. This is the recommended method for reproducing all results.

### Training and Evaluation

This is the strictly preferred and supported method for reproducing all results.

1. **Import:** Download the `notebooks/Final Classifier.ipynb` file from this repository and import it into Kaggle (via the "Import" button on the Kaggle Notebooks page).
2. **Hardware:** In the notebook settings on the right panel, set the Accelerator to **GPU T4** (or T4 x2).
3. **Secrets:** Navigate to the Add-ons menu, select "Secrets", and add a new secret labeled `GEMINI_API_KEY` containing your actual Gemini API key. Ensure the secret is attached to the notebook.
4. **Data:** Attach the required datasets (specific instructions and links are provided inside the notebook).
5. **Execute:** Click "Run All" to execute all cells sequentially.

The notebook automatically trains the model, performs threshold optimization, and produces the full evaluation including confusion matrices, error analysis, per-class metrics, threshold sweep visualizations, and the OOD demonstration. Expected runtime: 10 to 15 minutes.

### Hyperparameter Tuning

1. Open notebooks/mdeberta HPT.ipynb on Kaggle for mDeBERTa-specific tuning (99 trials).
2. Open notebooks/multiple models HPT.ipynb for cross-model architecture comparison.

## Running Locally

### Streamlit Demo

```bash
streamlit run app/app.py -- --checkpoint models/final_model.pt
```

Set GEMINI_API_KEY for live LLM responses.

### REST API

```bash
# Install API dependencies
pip install -r api/requirements.txt

# Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

See docs/api_doc.md for full endpoint documentation.

### Local Training (Alternative)

```bash
python src/train.py \
  --train-data data/large/train.json \
  --val-data data/large/validation.json \
  --output-dir models \
  --epochs 10
```

### Local Evaluation

```bash
python src/evaluate.py \
  --checkpoint models/final_model.pt \
  --dataset data/large/test.json \
  --output-metrics eval_outputs/test_metrics.json
```

## Dataset

1. 20,137 prompts across 3 classes: Benign (8,119), Jailbreak (9,662), Harmful (2,356).
2. Sources: JailbreakBench, TrustAIRLab, LMSYS Toxic Chat, SQuAD v2, Alpaca Cleaned.
3. Split: 70% Train (14,093) / 15% Validation (3,017) / 15% Test (3,027) with family-grouped stratification to prevent data leakage.
4. Full details in docs/milestone_2/.

## Model Architecture

1. Backbone: microsoft/mdeberta-v3-base (86M parameters, encoder-only).
2. Pooling: Attention-mask-aware mean pooling (not CLS token).
3. Head: Dropout(0.2) followed by Linear(768, 3) classification.
4. Training: AdamW optimizer, lr=3e-5, batch size=4, 512-token head-tail truncation.
5. Decision Thresholds: T_BLOCK=0.15, T_TRANSFORM=0.07 (composite-score-optimized).

## Evaluation Results

| Metric | Value |
|:---|:---|
| Accuracy | 0.9567 |
| Macro F1 | 0.9411 |
| Per-class F1 (Benign) | 0.9670 |
| Per-class F1 (Jailbreak) | 0.9648 |
| Per-class F1 (Harmful) | 0.8916 |
| ASR (Overall) | 1.77% |
| ASR (Jailbreak) | 1.17% |
| ASR (Harmful) | 4.24% |
| FRR | 3.94% |
| Composite Score | 0.9656 |
| ROC AUC | 0.9859 |
| PR AUC | 0.9852 |

## Known Limitations

1. The model has not been validated on external adversarial datasets outside the curated corpus.
2. The harmful class (11.7% of training data) is underrepresented.
3. Evaluation is restricted to English-language, single-turn text interactions.
4. The output guardrail and transformation layer lack formal quantitative evaluation.

## Troubleshooting

1. "Model not loaded": ensure the checkpoint file exists at models/final_model.pt. Generate it by running the Final Classifier notebook on Kaggle.
2. CUDA out of memory: reduce batch size or use a GPU with at least 16GB VRAM.
3. "LLM not configured": set the GEMINI_API_KEY environment variable.
4. Slow CPU inference: expected (20 to 50ms vs. 5.84ms on GPU).

## References

1. [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications)
2. [Meta PromptGuard](https://huggingface.co/meta-llama/Prompt-Guard-86M)
3. [JailbreakBench](https://jailbreakbench.github.io)
4. [DeBERTa: Decoding-Enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

# Milestone 6: Deployment, Documentation, and Reproducibility

Group 10: Data Science and Artificial Intelligence Lab
Date: April 2026

## 1. Deployment Overview

This milestone documents the deployment, documentation, and reproducibility strategy for the Guardrail Classifier system. The project transitioned from research notebooks into a deployed, accessible application with comprehensive documentation covering technical implementation, user interaction, and API access.

### 1.1 Deployed Components

| Component | Platform | Access |
|:---|:---|:---|
| Interactive Demo | Hugging Face Spaces | Publicly accessible via URL. Provides a side-by-side comparison of protected vs. unprotected LLM responses. |
| REST API | Local / Cloud | FastAPI service wrapping the guardrail pipeline. Exposes /predict and /health endpoints. |
| Streamlit Application | Local | Single-turn chat interface with real-time guardrail visualization. |
| Training and Evaluation Pipeline | Kaggle Notebooks | Reproducible end-to-end pipeline for model training and evaluation. |

### 1.2 Final Performance Summary

| Metric | Value |
|:---|:---|
| Macro F1 | 0.9411 |
| ASR (Overall) | 1.77% |
| FRR | 3.94% |
| Composite Score | 0.9656 |
| Mean Latency | 5.84ms (T4 GPU) |

## 2. Deployment Details

### 2.1 Hugging Face Spaces Demo

The primary deployment platform is Hugging Face Spaces, selected for its immediate global accessibility without requiring users to install dependencies or configure hardware. The application is built using the Gradio framework with customized styling to provide an intuitive interface for demonstrating the guardrail's efficacy. The demo provides a dual-execution mode that simultaneously shows the raw LLM response (without guardrail protection) alongside the guardrail-protected response, enabling visual comparison of the system's impact.

### 2.2 Local Streamlit Application

A local Streamlit application (app/app.py) provides a single-turn chat interface with real-time guardrail diagnostics. The application displays the action taken (ALLOW, TRANSFORM, or BLOCK), the predicted label and confidence, per-class softmax probabilities, and component-level latency breakdown.

To launch the application:

```
streamlit run app/app.py -- --checkpoint models/final_model.pt
```

The GEMINI_API_KEY environment variable must be set for live LLM responses. Without it, the application uses simulated responses while the guardrail classification remains fully functional.

### 2.3 REST API

A FastAPI service (api/main.py) exposes the guardrail pipeline for programmatic access. The service provides two endpoints.

1. POST /predict: accepts a JSON payload with a "prompt" field and returns the guardrail decision (action, label, confidence, layer triggered, per-class probabilities, and latency).
2. GET /health: returns the operational status of the service and model loading state.

To launch the API:

```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Full API documentation with request/response schemas and example curl commands is available in docs/api_doc.md.

## 3. Reproducibility

### 3.1 Reproducing Training and Evaluation (Kaggle)

The authoritative training and evaluation pipeline is implemented in the Kaggle notebooks. This is the strictly preferred and supported method for reproducing all results. To reproduce:

1. **Import:** Download `notebooks/Final Classifier.ipynb` from this repository and import it into Kaggle.
2. **Hardware:** In the Kaggle notebook settings panel, enable a **GPU T4** (or T4 x2) accelerator.
3. **API Key:** Navigate to the Add-ons menu, select "Secrets", and add a new secret labeled `GEMINI_API_KEY` containing your Gemini API key. Ensure this secret is attached to the notebook session.
4. **Data:** Ensure the required datasets are attached to the notebook (the notebook contains specific instructions for dataset attachment).
5. **Execution:** Select "Run All" to execute all cells sequentially. The notebook will automatically train the model, perform threshold optimization, and produce the full evaluation with all metrics, confusion matrices, and error analysis.

The complete process takes approximately 10 to 15 minutes on a T4 GPU.

### 3.2 Reproducing Hyperparameter Tuning

The hyperparameter search can be reproduced using notebooks/mdeberta HPT.ipynb (mDeBERTa-specific tuning) and notebooks/multiple models HPT.ipynb (cross-model comparison). These notebooks contain the 99-trial random search and produce the sensitivity analyses stored in the hp_outputs directory.

### 3.3 Environment and Dependencies

| Requirement | Specification |
|:---|:---|
| Python | 3.10 or later |
| PyTorch | 2.2.0 or later (with CUDA support for GPU execution) |
| Transformers | 4.39.0 or later |
| scikit-learn | 1.4.0 or later |
| sentencepiece | 0.2.0 or later |
| google-generativeai | 0.8.0 or later (for transformation stage) |
| Hardware (training) | NVIDIA T4 GPU (16GB VRAM) or equivalent |
| Hardware (inference) | Any CUDA-capable GPU or CPU (with higher latency) |

All dependencies are specified in requirements.txt with minimum version constraints.

### 3.4 Configuration and Secrets

The system requires one environment variable for full functionality: GEMINI_API_KEY, which provides access to the Gemini 2.5 Flash model for the prompt transformation stage. Without this key, the transformation stage falls back to the safe default query, and the system continues to function as a two-layer guardrail (regex plus classifier) without transformation capability. No other secrets or API tokens are required for classification.

### 3.5 Random Seed and Determinism

All experiments use a fixed random seed (42) applied to Python, NumPy, and PyTorch (including CUDA). This ensures that training runs, data shuffling, and dropout behavior are deterministically reproducible.

## 4. Repository Structure

```
project/
    app/                    Streamlit chat interface
        app.py              Single-turn demo with guardrail visualization
    api/                    FastAPI REST service
        main.py             /predict and /health endpoints
        requirements.txt    API-specific dependencies
    data/                   Dataset structure
        large/              Full 20,137-sample corpus
        small/              Reduced evaluation subset
    docs/                   All technical documentation
        overview.md         Project summary and architecture
        technical_doc.md    Comprehensive technical reference
        user_guide.md       End-user interaction guide
        api_doc.md          REST API endpoint documentation
        problem_statement.md Problem definition and scope
        licenses.md         Code, dataset, and model licenses
        milestone_1/        Project scope and literature review
        milestone_2/        Dataset preparation and structuring
        milestone_3/        Model architecture and pipeline design
        milestone_4/        Hyperparameter tuning and validation
        milestone_5/        Final evaluation and analysis
        milestone_6/        Deployment and documentation (this file)
    models/                 Saved model checkpoints
    notebooks/              Training and evaluation notebooks
        Final Classifier.ipynb      Production training and evaluation
        mdeberta HPT.ipynb          mDeBERTa hyperparameter tuning
        multiple models HPT.ipynb   Cross-model comparison
        notebooks guide.md          Notebook descriptions and results
    src/                    Model and pipeline source code
        guardrail_classifier.py     mDeBERTa encoder with mean pooling
        regex_filter.py             Rule-based pre-filter and LLM transformation
        guardrail_pipeline.py       Dual-stage inference pipeline
        train.py                    Training loop with early stopping
        evaluate.py                 Evaluation suite (ASR, FRR, latency)
        run_e2e_subset.py           End-to-end pipeline verification
    requirements.txt        Project-wide dependencies
    README.md               Executive summary and quick start
    LICENSE                 MIT License
```

## 5. Data Access and Preprocessing

### 5.1 Dataset Sources

The training and evaluation data is sourced from publicly available Hugging Face repositories: JailbreakBench, TrustAIRLab In The Wild Jailbreak Prompts, LMSYS Toxic Chat, Rajpurkar SQuAD v2, Yahma Alpaca Cleaned, and Rubend18 ChatGPT Jailbreak Prompts.

### 5.2 Preprocessing

The data pipeline (implemented in data.py) performs schema harmonization, whitespace normalization, boundary filtering, SHA-256 deduplication, and family-grouped stratified splitting. The entire pipeline is deterministic and reproducible with a fixed random seed.

## 6. Inference Pipeline

The inference pipeline processes a user prompt through the following stages.

1. The raw prompt text is received.
2. Layer 0 (Regex Pre-filter) normalizes the text (Unicode, leetspeak substitution, case folding) and evaluates 11 compiled regex rules. If cumulative severity exceeds 1.2, the prompt is blocked with sub-millisecond latency.
3. Layer 1 (Neural Classifier) tokenizes the prompt using the mDeBERTa SentencePiece tokenizer with head-tail truncation (max 512 tokens), performs a forward pass through the encoder, applies attention-mask-aware mean pooling, and produces three-class softmax probabilities.
4. Layer 2 (Decision Engine) computes the attack probability as the maximum of the jailbreak and harmful class probabilities. If this value >= 0.15, the prompt is blocked. If it falls in [0.07, 0.15), the prompt is transformed via the Gemini API. Otherwise, the prompt is allowed.
5. (If allowed or transformed) the prompt is forwarded to the downstream LLM.
6. Layer 3 (Output Guardrail) classifies the LLM response using the same model. If harmful, the response is replaced with a refusal message.

## 7. Error Handling

1. Model Loading Failure: if the checkpoint file is missing or corrupted, the API returns HTTP 503 with an error description. The Streamlit application displays an error message with instructions.
2. Gemini API Failure: if the external API is unavailable, rate-limited, or returns an empty response, the transformation stage falls back to SAFE_DEFAULT_QUERY. Classification continues to function without transformation.
3. NaN Detection: the training loop detects NaN losses and skips affected batches, aborting the epoch after 10 consecutive NaN batches.
4. Input Validation: the API validates request payloads using Pydantic schemas. Malformed requests return HTTP 422 with field-level error descriptions.

## 8. Troubleshooting

1. "Model not loaded" error: ensure the checkpoint file exists at the path specified by the CHECKPOINT_PATH environment variable (default: models/final_model.pt). The checkpoint must be generated by running the Final Classifier notebook on Kaggle.
2. CUDA out of memory: reduce the batch size or use a GPU with at least 16GB VRAM. For inference only, CPU mode is supported with higher latency.
3. "LLM not configured" warning: set the GEMINI_API_KEY environment variable. Without it, the transformation stage is disabled but classification remains functional.
4. Slow inference on CPU: expected. GPU inference averages 5.84ms; CPU inference averages 20 to 50ms depending on hardware.

## 9. Known Limitations and Future Work

### 9.1 Known Limitations

1. The model has not been validated on external adversarial datasets outside the curated corpus.
2. The harmful class (11.7% of training data) is underrepresented, contributing to its elevated ASR of 4.24%.
3. The output guardrail and transformation layer lack formal quantitative evaluation.
4. Evaluation is restricted to English-language prompts.
5. The system handles single-turn text interactions only; multi-turn context poisoning is out of scope.

### 9.2 Future Work

1. Expanded out-of-distribution evaluation with diverse adversarial datasets.
2. Harmful class data augmentation to reduce class imbalance.
3. Multilingual extension beyond English.
4. Human-in-the-loop evaluation of transformation quality.
5. Integration with containerized deployment frameworks for horizontal scaling.
6. Continuous monitoring and pipeline telemetry for production environments.

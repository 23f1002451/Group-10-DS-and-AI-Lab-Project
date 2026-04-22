# Project Overview

## 1. Purpose

This project constructs, trains, and deploys an inference-time guardrail system for protecting Large Language Model chat assistants from adversarial prompt manipulation. The system intercepts user prompts before they reach the downstream LLM, classifies them as benign, jailbreak, or harmful, and takes appropriate action based on empirically calibrated thresholds.

The classifier is fine-tuned on the microsoft/mdeberta-v3-base architecture with attention-mask-aware mean pooling, achieving a Macro F1 of 0.9411, an Attack Success Rate of 1.77%, and a False Refusal Rate of 3.94% on a 3,027-sample held-out test set.

## 2. Architecture Summary

The guardrail system operates as an intermediary middleware intercepting traffic between the user interface and the LLM inference provider. The pipeline processes each prompt through four sequential layers.

1. Layer 0 (Regex Pre-Filter): 11 severity-weighted compiled regular expression rules evaluate the prompt against known attack signatures. Prompts accumulating a cumulative severity score of 1.2 or above are blocked with sub-millisecond latency.
2. Layer 1 (Neural Classifier): the mDeBERTa-v3-base encoder with attention-mask-aware mean pooling produces three-class softmax probabilities (benign, jailbreak, harmful).
3. Layer 2 (Threshold Decision Engine): the attack probability (maximum of jailbreak and harmful probabilities) is compared against calibrated thresholds. If the attack probability is greater than or equal to 0.15, the prompt is blocked. If it falls between 0.07 and 0.15, the prompt is transformed via the Gemini 2.5 Flash API. If below 0.07, the prompt is allowed.
4. Layer 3 (Output Guardrail): the same classifier evaluates the LLM response. If the response is classified as harmful, it is replaced with a standardized refusal message.

## 3. Deployed Components

1. Interactive Demo: a Gradio-based application deployed on Hugging Face Spaces providing side-by-side comparison of protected vs. unprotected LLM responses.
2. REST API: a FastAPI service (api/main.py) exposing /predict and /health endpoints for programmatic access to the classification pipeline.
3. Streamlit Application: a local single-turn chat interface (app/app.py) with real-time guardrail diagnostics and latency visualization.
4. Training Pipeline: Kaggle notebooks (Final Classifier.ipynb) containing the complete reproducible training and evaluation pipeline.

## 4. Final Performance

| Metric | Value |
|:---|:---|
| Macro F1 | 0.9411 |
| ASR (Overall) | 1.77% |
| ASR (Jailbreak) | 1.17% |
| ASR (Harmful) | 4.24% |
| FRR | 3.94% |
| Composite Score | 0.9656 |
| Mean Latency | 5.84ms (T4 GPU) |

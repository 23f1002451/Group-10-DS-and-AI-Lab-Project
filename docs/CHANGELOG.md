# Changelog

This document tracks the evolution of the Guardrail Model project across its primary milestones.

### Milestone 1 (Conceptual Design and Technical Specification)
- **Problem Definition:** Outlined the risks of Large Language Models (LLMs) outputting restricted, harmful, or maliciously manipulated (jailbroken) content.
- **Solution Strategy:** Conceptualized an independent "Guardrail" model. Unlike keyword-based filters or complex secondary decoder LLMs, the proposed solution operates as an intermediate encoder-based classification layer (low-latency sequence classification).
- **Technical Refinements (Updated):** 
    - Formalized a three-class taxonomy (Benign, Jailbreak, Harmful) with concrete examples.
    - Specified a dual-layer defense-in-depth architecture combining semantic classification and deterministic sanitization.
    - Defined a sub-300ms latency budget with clear hardware assumptions (NVIDIA T4).
    - Established a formal labeling methodology and consistency guidelines for training data.
    - Outlined fallback mechanisms and contextual discrimination for output validation.
- **Status:** Theoretical research and technical specification complete.

### Milestone 2 (Dataset Engineering and Preparation Pipeline)
- **Dataset Construction:** Built a multi-domain corpus of 20,137 prompts mapping to `benign`, `jailbreak`, and `harmful` classes.
- **Splits:** 14,093 training, 3,017 validation, 3,027 test records with family-grouped stratification.
- **Sources Used:** Integrated data from `rajpurkar/squad_v2`, `yahma/alpaca-cleaned`, `JailbreakBench`, `lmsys/toxic-chat`, and `TrustAIRLab`.
- **Technical Refinements (Updated):** 
    - Formalized a Seven-Phase Preparation Pipeline (Collection to Multi-Format Export).
    - Established an academic Jailbreak Taxonomy (Role-play, Instruction Override, Multi-step, Prompt Injection, Obfuscation) with deterministic labeling.
    - Implemented a Phased Cleaning Sequence (whitespace standardization, SHA-256 deduplication, and boundary filtering).
    - Designed a Family-Grouped Stratification algorithm ensuring zero semantic leakage across Train/Val/Test splits.
    - Documented head-tail truncation strategy for 512-token context window (98.35% coverage).

### Milestone 3 (Model Architecture and Inference Middleware)
- **Hybrid Guardrail Architecture:** Finalized a layered defense strategy (Layer 0 deterministic regex rules + Layer 1 neural classification + Layer 2 threshold-based decision engine + Layer 3 LLM-based transformation) integrated as inference-time middleware.
- **Model Design (Updated):** 
    - Selected `microsoft/mdeberta-v3-base` with attention-mask-aware mean pooling and a regularized linear classification head.
    - Implemented head-tail truncation for the 512-token context window (98.35% coverage of 20,137 samples).
    - Dropout rate set to 0.3 for stronger regularization with small batch sizes.
- **Formal Training Setup:** Specified rigorous hyperparameters (Batch 4, LR 3e-5, Early Stopping patience 3, label smoothing 0.1, gradient clipping 1.0, AdamW with weight decay 0.01) identified through random search optimization.
- **Decision Threshold Policy:** Established a tri-modal calibration strategy for Block/Transform/Allow decisions (T_BLOCK=0.8, T_TRANSFORM=0.3) with composite scoring (W_F1=0.3, W_ASR=0.5, W_FRR=0.2).
- **Regex Pre-Filter:** Documented 11 severity-weighted regex rules with cumulative scoring and threshold-based blocking.
- **Normalization Analysis:** Documented the `_norm` function's aggressive text normalization for regex matching, including leetspeak substitution, Unicode normalization, and over-generalization limitations.
- **Metrics and Error Roadmap:** Expanded evaluation to include composite scoring, per-attack taxonomy breakdown, confusion matrix analysis, and failure mode categorization (FN, FP, WRONG_TYPE).

### Milestone 4 (Production-Ready Guardrails)
- **Architecture:** Hybrid Defense-in-Depth Pipeline combining regex pre-filter, mDeBERTa-v3-base with mean pooling, threshold-based decision engine, and Gemini 2.5 Flash LLM transformation.
- **Training:** Converged at epoch 3 (best validation F1=0.9480), early stopped at epoch 6.
- **Performance (Test Set, N=3,027):** Macro F1=0.9399, ASR=2.71%, FRR=2.87%, Composite Score=0.9627.
- **Diagnostics:** ROC AUC=0.9828, PR AUC=0.9849, 128 misclassified samples analyzed across FN/FP/WRONG_TYPE categories (see `notebooks/final_model_outputs/misclassified.csv`).
- **Latency:** 5.836ms mean (regex 0.038ms + model 5.798ms), ~171 req/s throughput.
- **Error Analysis:** Detailed failure mode breakdown with regex hit correlation, confidence distribution analysis, and layer-wise decision attribution.

### Milestone 5 (Final Evaluation and Deployment Documentation)
- **API Key Documentation:** Added explicit instructions for configuring the Gemini API key via Kaggle Secrets, Google Colab Secrets, and local environment variables.
- **Transformation Analysis:** Documented LLM transformation reliability, caching mechanism, fallback behavior, and cost implications of Gemini API usage during inference.
- **Composite Score Justification:** Provided detailed rationale for the W_F1=0.3, W_ASR=0.5, W_FRR=0.2 weight allocation based on asymmetric error costs in safety-critical systems.
- **Regex Rationale:** In-depth documentation of all 11 regex patterns, their design rationale, severity assignments, and known limitations.
- **Normalization Analysis:** Detailed exploration of the `_norm` function's aggressive normalization behavior and its potential for over-generalization on benign prompts.
- **OOD Evaluation:** Documented the removal of the 60-prompt Gemini-generated OOD set (due to rate limiting and methodological concerns) and its replacement with 30 hand-crafted demo prompts.
- **End-to-End Demo:** Documented the `guardrail()` function and its 30-prompt demonstration results (OOD ASR=25.0%, FRR=0.0%).
- **Hyperparameter Methodology:** Documented the multi-phase hyperparameter selection process (literature initialization, random search, composite-score-driven selection, memory-constrained refinement).
- **Future Work:** Identified key limitations and proposed improvements including expanded OOD evaluation, harmful class augmentation, multilingual extension, and human-in-the-loop transformation quality assessment.

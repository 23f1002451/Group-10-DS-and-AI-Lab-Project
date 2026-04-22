# Project Overview

## 1. Purpose

The primary objective of this project is to design, implement, and evaluate an inference-time guardrail system that protects Large Language Model (LLM) chat assistants from adversarial prompt manipulation. As LLMs become increasingly deployed in customer-facing and enterprise applications, they remain vulnerable to jailbreak attacks, prompt injections, and requests for harmful content that can bypass the models' internal safety alignment. These vulnerabilities introduce legal, compliance, reputational, and operational risks for organisations that deploy such systems.

This project addresses these vulnerabilities by introducing a modular, hybrid defence-in-depth pipeline that operates as middleware between the user interface and the downstream LLM (Gemini API). The system combines three complementary defensive layers:

1. A deterministic regex-based pre-filter for fast, interpretable blocking of known attack signatures.
2. A fine-tuned `microsoft/mdeberta-v3-base` transformer classifier for context-aware semantic classification of prompts into three categories: benign, jailbreak, and harmful.
3. A threshold-based decision engine with optional LLM-powered prompt transformation (Gemini 2.5 Flash) for handling ambiguous prompts in the intermediate confidence range.

The system is designed to be model-agnostic, operating independently of the downstream LLM without requiring any modification to the base model's weights or architecture.

## 2. Architecture Summary

The guardrail system operates as an intermediary gateway intercepting traffic between the user interface and the LLM inference provider. The pipeline processes each incoming prompt sequentially through four layers:

| Layer | Component | Function |
| :--- | :--- | :--- |
| Layer 0 | Regex Pre-filter | Severity-weighted pattern matching for known attack signatures |
| Layer 1 | mDeBERTa-v3-base Classifier | Semantic three-class classification with mean pooling |
| Layer 2 | Threshold Decision Engine | Maps classifier probabilities to ALLOW, TRANSFORM, or BLOCK actions |
| Layer 3 | LLM Transformation | Gemini 2.5 Flash-based prompt sanitisation for ambiguous cases |

**Data Flow Sequence:**

```
User Prompt --> [Regex Filter] --> [mDeBERTa Encoder (Mean Pool)] --> [Threshold Gate] --> ALLOW / TRANSFORM / BLOCK
```

1. **Input:** The user submits a natural language prompt via the frontend.
2. **Regex Pre-filter:** The prompt is normalised (Unicode, leetspeak substitution, case folding) and scanned against 11 severity-weighted regex patterns. Prompts accumulating cumulative severity above the block threshold (1.2) are rejected immediately without neural computation.
3. **Tokenisation:** Prompts passing the regex layer are tokenised using the mDeBERTa SentencePiece tokeniser with a 512-token context window. Prompts exceeding this length are handled via head-tail truncation, preserving both the adversarial framing at the start and the attack payload at the end.
4. **Classification:** The tokenised input is passed through the mDeBERTa-v3-base encoder with attention-mask-aware mean pooling, followed by dropout regularisation and a linear classification head producing three logits (benign, jailbreak, harmful).
5. **Threshold Decision:** The maximum attack probability (P_atk = max(P_jailbreak, P_harmful)) is compared against two calibrated thresholds:
   - P_atk >= T_BLOCK: Prompt is rejected immediately.
   - T_TRANSFORM <= P_atk < T_BLOCK: Prompt is sanitised via Gemini 2.5 Flash and re-evaluated.
   - P_atk < T_TRANSFORM: Prompt is forwarded to the downstream LLM.
6. **Output:** For allowed prompts, the downstream LLM's response is returned to the user alongside system latency and confidence metrics.

## 3. Performance Summary

The following metrics were obtained on the held-out test set (N=3,027) using the final production model:

| Metric | Value |
| :--- | :--- |
| Macro F1 (Full Pipeline) | 0.9399 |
| Attack Success Rate (Overall) | 2.71% |
| False Refusal Rate | 2.87% |
| Composite Score | 0.9627 |
| ROC AUC (Attack vs. Benign) | 0.9828 |
| PR AUC | 0.9849 |
| Inference Latency (Mean) | 5.836 ms |
| Throughput | Approximately 171 req/s |

## 4. Deployed Components

The project has progressed from local research notebooks into a fully deployed demonstration system comprising the following components:

- **Research Notebooks:** Three Kaggle notebooks (`multiple models HPT.ipynb`, `mdeberta HPT.ipynb`, `Final Classifier.ipynb`) covering the complete experimental pipeline from multi-model benchmarking through to production model training and evaluation.
- **Source Code:** Modular Python implementation in `src/` including the classifier architecture, regex filter, training loop, evaluation suite, and end-to-end pipeline.
- **Frontend Application:** A Gradio-based user interface deployed on Hugging Face Spaces, providing a side-by-side comparison of protected and unprotected LLM responses.
- **Guardrail Classifier Weights:** The fine-tuned mDeBERTa-v3-base checkpoint available for download and deployment.
- **Documentation:** Comprehensive milestone reports, technical documentation, and user guides covering all phases from problem definition through deployment.

## 5. Dataset

The system was trained and evaluated on a multi-domain corpus of 20,137 labelled prompts spanning three classes:

| Split | Total | Benign | Jailbreak | Harmful |
| :--- | :--- | :--- | :--- | :--- |
| Train | 14,093 | 5,683 | 6,761 | 1,649 |
| Validation | 3,017 | 1,217 | 1,447 | 353 |
| Test | 3,027 | 1,219 | 1,454 | 354 |

Sources include JailbreakBench, TrustAIRLab, LMSYS Toxic Chat, SQuAD v2, and Alpaca Cleaned. Family-grouped stratification ensures zero semantic leakage between splits.

## 6. Model Architecture

- **Backbone:** microsoft/mdeberta-v3-base (86M parameters, encoder-only with disentangled attention)
- **Pooling:** Attention-mask-aware mean pooling over all non-padding tokens
- **Head:** Dropout(0.2) followed by Linear(768, 3) classification
- **Training:** AdamW optimiser, lr=3e-5, class-weighted CrossEntropyLoss, linear warmup schedule
- **Inference:** Mean pooling followed by 3-class softmax, then threshold-based decision logic

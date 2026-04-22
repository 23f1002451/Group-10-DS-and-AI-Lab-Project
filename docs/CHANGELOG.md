# Changelog

This document tracks the evolution of the Guardrail Model project across its primary milestones.

### Milestone 1 (Conceptual Design and Technical Specification)

Outlined the risks of Large Language Models outputting harmful or jailbroken content. Conceptualized an independent guardrail model operating as an intermediate encoder-based classification layer. Formalized a three-class taxonomy (Benign, Jailbreak, Harmful) with concrete examples. Specified a dual-layer defense-in-depth architecture combining semantic classification and deterministic sanitization. Defined a sub-300ms latency budget with clear hardware assumptions (NVIDIA T4). Established a formal labeling methodology and consistency guidelines. Outlined fallback mechanisms and contextual discrimination for output validation.

### Milestone 2 (Dataset Engineering and Preparation Pipeline)

Built a multi-domain corpus of 20,137 prompts mapping to benign, jailbreak, and harmful classes. Split into 14,093 training, 3,017 validation, and 3,027 test records with family-grouped stratification. Integrated data from SQuAD v2, Alpaca Cleaned, JailbreakBench, LMSYS Toxic Chat, and TrustAIRLab. Formalized a Seven-Phase Preparation Pipeline. Established a jailbreak taxonomy (Role-play, Instruction Override, Multi-step, Prompt Injection, Obfuscation). Implemented SHA-256 deduplication and head-tail truncation strategy for the 512-token context window (98.35% coverage).

### Milestone 3 (Model Architecture and Inference Middleware)

Finalized a hybrid layered defense strategy (Layer 0 regex, Layer 1 neural classifier, Layer 2 threshold decision engine, Layer 3 LLM transformation). Selected microsoft/mdeberta-v3-base with attention-mask-aware mean pooling. Implemented head-tail truncation for 512-token context window. Specified training hyperparameters (Batch 4, LR 3e-5, Early Stopping patience 3, Dropout 0.2, Gradient Clipping 0.5, Label Smoothing 0.0). Established tri-modal calibration with T_BLOCK=0.15 and T_TRANSFORM=0.07. Documented 11 severity-weighted regex rules. Expanded evaluation framework to include composite scoring, per-attack taxonomy breakdown, and failure mode categorization.

### Milestone 4 (Production-Ready Guardrails)

Trained the production model with final hyperparameters from Final Classifier.ipynb. Performance on the test set (N=3,027): Macro F1=0.9411, ASR=1.77%, FRR=3.94%, Composite Score=0.9656. Per-class F1: Benign=0.967, Jailbreak=0.9648, Harmful=0.8916. Per-class ASR: Jailbreak=1.17%, Harmful=4.24%. ROC AUC=0.9859, PR AUC=0.9852. Latency: 5.836ms mean (regex 0.038ms + model 5.798ms), approximately 171 req/s throughput. 131 misclassified samples analyzed across False Negative, False Positive, and Wrong Category failure modes.

### Milestone 5 (Final Evaluation and Analysis)

Comprehensive evaluation with quantitative error analysis. Documented threshold calibration methodology and composite score weight justification. Identified known limitations including absence of external dataset validation, output guardrail evaluation gap, and harmful class underrepresentation. Proposed future work including expanded OOD evaluation, harmful class augmentation, multilingual extension, and human-in-the-loop transformation quality assessment.

### Milestone 6 (Deployment, Documentation, and Reproducibility)

Deployed interactive demo on Hugging Face Spaces with Gradio interface. Created FastAPI REST API with /predict and /health endpoints. Updated local Streamlit application with real-time guardrail diagnostics. Documented complete reproducibility workflow via Kaggle notebooks. Finalized repository structure, error handling, and troubleshooting documentation.

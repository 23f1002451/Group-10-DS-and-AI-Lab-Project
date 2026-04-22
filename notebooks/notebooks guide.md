# Notebooks Reference Guide

## Overview

This directory contains the complete set of Jupyter notebooks used throughout the research and development of the Guardrail Classifier system. Each notebook addresses a distinct phase of the experimental pipeline, from multi-model benchmarking and hyperparameter optimisation through to final production model training and end-to-end evaluation. All notebooks were executed on Kaggle environments equipped with dual NVIDIA T4 GPUs (16 GB VRAM each) running Python 3.10.

The notebooks are listed below in the recommended reading order, which reflects the chronological progression of the research.

---

## 1. multiple models HPT.ipynb

**Purpose:** Multi-model comparative benchmarking and hyperparameter search across three candidate transformer architectures.

**Models Evaluated:**

| Model | Parameters | Architecture |
| :--- | :--- | :--- |
| microsoft/mdeberta-v3-base | 86M | Encoder-only, disentangled attention |
| roberta-base | 125M | Encoder-only, standard self-attention |
| bert-base-uncased | 110M | Encoder-only, standard self-attention |

**Methodology:**
- Random hyperparameter search with 33 trials per model (99 total), each screened for 3 epochs with early stopping.
- Hyperparameters sampled: learning rate (1e-5, 3e-5, 5e-5), weight decay (0.01, 0.05, 0.1), dropout (0.1, 0.2, 0.3), warmup ratio (0.05, 0.1), gradient clip (0.5, 1.0), label smoothing (0.0, 0.1, 0.2), and epoch count (3, 5).
- All models used a fixed MAX_LENGTH of 512 tokens and BATCH_SIZE of 4.
- Composite scoring function: S = 0.3 * F1 + 0.5 * (1 - ASR) + 0.2 * (1 - FRR).
- Each model was evaluated with and without the regex pre-filter.
- Threshold optimisation was performed via a two-dimensional sweep over T_BLOCK (0.40 to 0.85 in 0.05 increments) and T_TRANSFORM (0.20 to 0.45 in 0.05 increments).
- The best configuration per model was retrained to convergence and evaluated on the held-out test set.

**Key Results:**
- mDeBERTa-v3-base consistently achieved the highest composite score across all trials, confirming its suitability for the prompt classification task.
- The disentangled attention mechanism in mDeBERTa provided superior performance on semantically complex jailbreak prompts compared to the standard self-attention used by RoBERTa and BERT.
- The comparative evaluation justified the architectural selection that was carried forward into subsequent experiments.

**Outputs:** Model comparison table, HPO sensitivity plots, per-model confusion matrices, threshold heatmaps, layer-wise decision attribution analysis, ROC and PR curves, comprehensive error analysis with failure mode categorisation.

**Runtime:** Approximately 4 to 6 hours for all 99 trials across three models.

---

## 2. mdeberta HPT.ipynb

**Purpose:** Focused hyperparameter optimisation for the selected mDeBERTa-v3-base architecture, with extended analysis of training dynamics and threshold calibration.

**Methodology:**
- Isolated random search with 10 trials on the mDeBERTa-v3-base model only (after the multi-model comparison confirmed its superiority).
- Same hyperparameter search space as the multi-model notebook.
- Detailed sensitivity analysis examining the relationship between individual hyperparameters and the composite score.
- Two-dimensional threshold sweep on the validation set to identify optimal T_BLOCK and T_TRANSFORM values for both the model-only and full-pipeline (with regex) configurations.
- Ablation study comparing three system configurations: argmax baseline (no thresholding), model with threshold, and full pipeline (model + regex + threshold).
- Layer-wise decision attribution tracing every test sample through the pipeline and recording which component made the final classification decision.

**Key Results:**
- Identified the best hyperparameter configuration for mDeBERTa-v3-base with composite score maximisation.
- The sensitivity analysis revealed that learning rate and dropout had the largest marginal effects on performance, while warmup ratio and label smoothing had relatively smaller impacts.
- The threshold heatmap visualisation demonstrated that the composite score surface is relatively smooth, indicating that the system is not overly sensitive to precise threshold values.
- The ablation study quantified the marginal contribution of each pipeline layer:
  - Adding threshold logic improved composite score over raw argmax predictions.
  - Adding the regex pre-filter further reduced ASR with minimal impact on FRR.

**Outputs:** HPO results CSV, parameter sensitivity bar charts, threshold optimisation heatmap, ablation comparison table, confusion matrices (model-only vs. full pipeline), layer-wise pipeline evaluation plots, error analysis with failure mode characterisation, latency breakdown table, final summary JSON.

**Runtime:** Approximately 60 to 90 minutes.

---

## 3. Final Classifier.ipynb

**Purpose:** Production model training and complete end-to-end evaluation of the finalised Guardrail Classifier system. This is the authoritative notebook for all reported metrics.

**Methodology:**
- The model is trained using a fixed hyperparameter configuration derived from the prior optimisation experiments:
  - MAX_LENGTH: 512, BATCH_SIZE: 4, LEARNING_RATE: 3e-5
  - WEIGHT_DECAY: 0.01, WARMUP_RATIO: 0.05, DROPOUT: 0.2
  - GRADIENT_CLIP: 0.5, EPOCHS: 10, EARLY_STOPPING_PATIENCE: 3
  - LABEL_SMOOTHING: 0.0
  - Composite weights: W_F1 = 0.3, W_ASR = 0.5, W_FRR = 0.2
- Single training run with early stopping based on validation macro F1.
- Threshold sweep on the validation set to identify optimal T_BLOCK and T_TRANSFORM values.
- Complete test set evaluation under two configurations: model-only and full pipeline (regex + model + threshold + LLM transformation).
- Comprehensive error analysis of all misclassified test samples with failure mode categorisation (FN, FP, WRONG_TYPE).
- Layer-wise decision attribution for the full pipeline.
- End-to-end latency profiling with GPU synchronisation and 50-repetition averaging.
- End-to-end inference demonstration with 30 hand-crafted prompts (10 benign, 10 jailbreak, 10 harmful) that are not present in any training, validation, or test split.

**Key Results:**
- The model achieves production-ready performance on the held-out test set (N=3,027).
- ROC AUC and PR AUC values confirm strong discriminative capability.
- The 30-prompt end-to-end demonstration validates the full pipeline from regex filtering through neural classification, threshold-based decision logic, and LLM-powered prompt transformation.
- The demonstration achieves 0% FRR on the OOD benign prompts, confirming that the classifier does not over-refuse legitimate queries.
- The latency profile confirms that the total guardrail overhead is well within the 300ms budget defined in the project requirements.

**Outputs:** Trained model checkpoint (final_model.pt), token length distribution and coverage analysis, threshold sweep CSV and visualisation, confusion matrices, ROC and PR curves, error analysis plots, misclassified samples CSV, layer-wise decision attribution, latency breakdown, end-to-end demo results with action matrix.

**Runtime:** Approximately 10 to 15 minutes.

---

## Summary of Experimental Progression

| Phase | Notebook | Models | Trials | Primary Contribution |
| :--- | :--- | :--- | :--- | :--- |
| Model Selection | multiple models HPT.ipynb | 3 (mDeBERTa, RoBERTa, BERT) | 99 | Comparative benchmarking; mDeBERTa selected |
| Hyperparameter Optimisation | mdeberta HPT.ipynb | 1 (mDeBERTa) | 10 | Focused HPO, sensitivity analysis, ablation study |
| Production Training | Final Classifier.ipynb | 1 (mDeBERTa) | 1 | Final model, full evaluation, deployment-ready pipeline |

---

## Supplementary Outputs

The `notebooks/outputs/` directory contains all generated artefacts from the notebook executions, including:

- `model/` : Saved model checkpoints (.pt files)
- `evaluation/` : Misclassified sample CSVs for error analysis
- Threshold sweep CSVs and visualisation plots
- Token length distribution and coverage justification plots
- Confusion matrix, ROC, PR, and error analysis visualisations
- HPO results CSVs and sensitivity analysis plots
- Layer-wise decision attribution charts
- Final summary JSON files

---

## Reproducibility

All notebooks use a fixed random seed (42) applied to Python, NumPy, and PyTorch (including CUDA) to ensure deterministic reproducibility. The dataset is loaded from pre-computed JSON splits (train.json, validation.json, test.json), which are available on Kaggle. A valid Gemini API key is required for the LLM transformation stage; without it, the system will fall back to a safe default query for all prompts in the suspicious zone, and the remainder of the pipeline will function normally.

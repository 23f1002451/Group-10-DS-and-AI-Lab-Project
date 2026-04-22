# Notebooks Guide

This document describes each notebook in the repository, its purpose, and the key results obtained.

## 1. Final Classifier.ipynb

**Purpose.** This is the production training and evaluation notebook. It implements the complete guardrail pipeline from data loading through model training, threshold optimization, and final evaluation on the held-out test set.

**Contents.**
1. Dataset loading and verification of the 20,137-sample corpus (14,093 train, 3,017 validation, 3,027 test).
2. Model definition: mDeBERTa-v3-base encoder with attention-mask-aware mean pooling, Dropout (p=0.2), and Linear classification head (768 to 3).
3. Training loop with AdamW optimizer, inverse-frequency class weighting, mixed-precision training, gradient clipping, and early stopping.
4. Threshold sweep across the validation set to identify optimal T_BLOCK and T_TRANSFORM values using the composite scoring function.
5. Full evaluation on the test set with confusion matrix, per-class metrics, error analysis, and out-of-distribution demonstration.

**Key Hyperparameters.**
| Parameter | Value |
|:---|:---|
| MAX_LENGTH | 512 |
| BATCH_SIZE | 4 |
| LEARNING_RATE | 3e-5 |
| WEIGHT_DECAY | 0.01 |
| WARMUP_RATIO | 0.05 |
| DROPOUT | 0.2 |
| GRADIENT_CLIP | 0.5 |
| EPOCHS | 10 |
| EARLY_STOPPING_PATIENCE | 3 |
| LABEL_SMOOTHING | 0.0 |

**Final Thresholds.**
| Threshold | Value |
|:---|:---|
| T_BLOCK | 0.15 |
| T_TRANSFORM | 0.07 |

**Results.**
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
| Mean Latency | 5.84ms (T4 GPU) |

**Confusion Matrix.**
```
                Predicted
              Benign  Jailbreak  Harmful
Actual Benign   1171       34       14
     Jailbreak    17     1396       41
       Harmful    15       10      329
```

**Reproducibility.** Execute the notebook on Kaggle with a T4 GPU runtime. Expected runtime: 10 to 15 minutes.

## 2. mdeberta HPT.ipynb

**Purpose.** This notebook implements the hyperparameter tuning process specifically for the mDeBERTa-v3-base architecture. It conducts a systematic random search to identify the optimal training configuration.

**Contents.**
1. Definition of the hyperparameter search space: learning rates [1e-5, 2e-5, 3e-5, 5e-5], batch sizes [4, 8, 16], dropout values [0.1, 0.2, 0.3], max lengths [256, 380, 444, 512], weight decay [0.01, 0.05, 0.1].
2. Execution of 99 random search trials with early stopping.
3. Composite score evaluation for each trial on the validation set.
4. Sensitivity analysis of individual hyperparameters.
5. Identification of the locally optimal configuration (MAX_LENGTH=444, BS=4, LR=3e-5, DROPOUT=0.3).

**Key Findings.**
1. Learning rate and max sequence length have the largest marginal effects on performance.
2. Dropout and warmup ratio have smaller marginal impacts.
3. The HPT-optimal MAX_LENGTH of 444 was overridden in production to 512 for better coverage (98.35% vs. 94.92%).

**Reproducibility.** Execute the notebook on Kaggle with a T4 GPU runtime.

## 3. multiple models HPT.ipynb

**Purpose.** This notebook conducts a cross-model architecture comparison to validate the selection of mDeBERTa-v3-base over alternative transformer architectures.

**Contents.**
1. Evaluation of candidate architectures including BERT-base, RoBERTa-base, and mDeBERTa-v3-base.
2. Standardized training protocol applied to each architecture.
3. Comparative analysis of Macro F1, ASR, FRR, and composite scores across architectures.
4. Justification of the final architecture selection.

**Key Findings.**
1. mDeBERTa-v3-base consistently outperforms BERT and RoBERTa on the jailbreak detection task.
2. The disentangled attention mechanism of DeBERTa provides superior performance on complex prompt-injection grammars.
3. The architecture selection is validated by the final production metrics (F1=0.9411, ASR=1.77%).

**Reproducibility.** Execute the notebook on Kaggle with a T4 GPU runtime.

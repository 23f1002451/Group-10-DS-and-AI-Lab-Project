# Technical Documentation

## 1. Environment Setup

### 1.1 Dependencies

The following dependencies are required for local execution. All versions are specified in requirements.txt.

| Package | Minimum Version | Purpose |
|:---|:---|:---|
| torch | 2.2.0 | Neural network framework with CUDA support |
| transformers | 4.39.0 | Pre-trained model loading and tokenization |
| scikit-learn | 1.4.0 | Evaluation metrics computation |
| sentencepiece | 0.2.0 | SentencePiece tokenizer for mDeBERTa |
| datasets | 2.19.0 | Dataset loading utilities |
| numpy | 1.26.0 | Numerical operations |
| tqdm | 4.66.0 | Progress bar visualization |
| protobuf | 4.25.0 | Protocol buffer support for sentencepiece |
| streamlit | 1.30.0 | Local demo application framework |
| google-generativeai | 0.8.0 | Gemini API client for transformation stage |

### 1.2 Hardware Requirements

Training requires an NVIDIA T4 GPU (16GB VRAM) or equivalent. Inference is supported on both GPU (5.84ms per prompt) and CPU (20 to 50ms per prompt). The batch size of 4 with 512-token sequences utilizes approximately 15GB of the T4's 16GB VRAM during backpropagation.

### 1.3 Installation

```
pip install -r requirements.txt
```

For the API service, install additional dependencies:

```
pip install -r api/requirements.txt
```

## 2. Data Pipeline

### 2.1 Dataset Sources

The training corpus comprises 20,137 labeled prompts from publicly available Hugging Face repositories: JailbreakBench, TrustAIRLab In The Wild Jailbreak Prompts, LMSYS Toxic Chat, Rajpurkar SQuAD v2, Yahma Alpaca Cleaned, and Rubend18 ChatGPT Jailbreak Prompts. All datasets use open-source research licenses (MIT, CC-BY, CC-BY-SA).

### 2.2 Preprocessing

The data pipeline performs schema harmonization (mapping diverse source fields to a unified prompt_text feature), whitespace normalization, boundary filtering (minimum 8 characters), SHA-256 deduplication, and family-grouped stratified splitting (70/15/15 train/validation/test). The entire pipeline is deterministic with a fixed random seed of 42.

### 2.3 Class Distribution

| Split | Benign | Jailbreak | Harmful | Total |
|:---|:---|:---|:---|:---|
| Train | 5,683 | 6,761 | 1,649 | 14,093 |
| Validation | 1,217 | 1,447 | 353 | 3,017 |
| Test | 1,219 | 1,454 | 354 | 3,027 |

## 3. Model Architecture

### 3.1 Encoder

The classifier uses the microsoft/mdeberta-v3-base encoder (86M parameters), which employs disentangled attention to separately encode content and position information.

### 3.2 Pooling

Attention-mask-aware mean pooling is used instead of CLS token extraction. The pooling computes a weighted average across all non-padding tokens, ensuring that adversarial signals distributed throughout the prompt are captured.

### 3.3 Classification Head

The pooled 768-dimensional representation passes through Dropout (p=0.2) and a Linear layer (768 to 3) producing logits for benign, jailbreak, and harmful classes.

## 4. Training Summary

### 4.1 Hyperparameters

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
| OPTIMIZER | AdamW |

### 4.2 Training Details

Training was conducted on Kaggle with dual NVIDIA T4 GPUs. The model converged at epoch 3 (best validation Macro F1), with early stopping triggered after 3 consecutive epochs without improvement. Mixed-precision training (AMP) with gradient scaling was used. Inverse-frequency class weighting compensates for the harmful class imbalance.

### 4.3 Reproducibility

To reproduce training, execute the Final Classifier.ipynb notebook on Kaggle with a T4 GPU runtime. Expected runtime is 10 to 15 minutes.

## 5. Evaluation Summary

### 5.1 Final Metrics (Test Set, N=3,027)

| Metric | Value |
|:---|:---|
| Accuracy | 0.9567 |
| Macro F1 | 0.9411 |
| ASR (Overall) | 1.77% |
| ASR (Jailbreak) | 1.17% |
| ASR (Harmful) | 4.24% |
| FRR | 3.94% |
| Composite Score | 0.9656 |
| ROC AUC | 0.9859 |
| PR AUC | 0.9852 |

### 5.2 Confusion Matrix

```
                Predicted
              Benign  Jailbreak  Harmful
Actual Benign   1171       34       14
     Jailbreak    17     1396       41
       Harmful    15       10      329
```

## 6. Inference Pipeline

The inference pipeline processes a user prompt through the following stages.

```python
# 1. Regex Pre-Filter
from src.regex_filter import regex_check
result = regex_check(prompt)  # Returns action, severity, category, hits

# 2. Neural Classification
from src.guardrail_pipeline import GuardrailPipeline
pipeline = GuardrailPipeline.from_checkpoint("models/final_model.pt")
decision = pipeline.classify_input(prompt)
# decision.action: "allow", "transform", or "block"
# decision.label: "benign", "jailbreak", or "harmful"
# decision.confidence: float
# decision.model_probabilities: dict

# 3. Threshold Decision
# p_attack = max(P(jailbreak), P(harmful))
# p_attack >= 0.15 --> BLOCK
# 0.07 <= p_attack < 0.15 --> TRANSFORM
# p_attack < 0.07 --> ALLOW
```

## 7. Deployment Details

### 7.1 Hugging Face Spaces

The primary demo is deployed on Hugging Face Spaces using Gradio. The application dynamically loads the classifier weights and provides real-time classification with side-by-side comparison.

### 7.2 Local Streamlit Application

```
streamlit run app/app.py -- --checkpoint models/final_model.pt
```

### 7.3 REST API

```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Full API documentation is available in docs/api_doc.md.

## 8. System Design Considerations

1. The threshold-based decision engine uses the maximum of the jailbreak and harmful class probabilities (not a simple argmax), providing granular control over the safety-utility tradeoff.
2. By evaluating safety prior to requesting LLM inference, the system eliminates GPU generation costs for blocked prompts.
3. The regex pre-filter provides sub-millisecond blocking for known attack signatures without requiring neural computation.
4. The LLM transformation stage (Gemini 2.5 Flash) operates only on the small fraction of prompts in the suspicious zone (attack probability between 0.07 and 0.15).

## 9. Error Handling

1. Model loading failures return descriptive error messages via both the API (HTTP 503) and the Streamlit interface.
2. Gemini API failures trigger fallback to SAFE_DEFAULT_QUERY, ensuring classification continues without transformation.
3. NaN loss detection during training skips affected batches and aborts after 10 consecutive failures.
4. API input validation uses Pydantic schemas, returning HTTP 422 for malformed requests.

## 10. Reproducibility Checklist

1. All training and evaluation is reproducible via the Final Classifier.ipynb notebook on Kaggle with a T4 GPU runtime.
2. Fixed random seed (42) is applied to Python, NumPy, and PyTorch.
3. All hyperparameters are documented in the notebook and in this document.
4. Dataset splits are deterministically generated using family-grouped stratification with SHA-256 hashing.
5. Model checkpoint, evaluation outputs, and visualization artifacts are generated by the notebook.

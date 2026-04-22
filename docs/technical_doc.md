# Technical Documentation

## 1. Environment and Requirements

### 1.1 Runtime Environment

The system is designed for execution in both cloud-based notebook environments and local deployments:

- **Python:** 3.10 or later recommended.
- **Hardware (Training):** Kaggle environment with 2x NVIDIA T4 GPUs (16 GB VRAM each). Training utilises mixed-precision (AMP) to maximise memory efficiency.
- **Hardware (Inference):** Minimum 2 GB RAM for CPU inference. GPU inference on a single T4 achieves approximately 171 requests per second.
- **Expected Runtime:** Approximately 10 to 15 minutes for single model training and full evaluation on Kaggle.

### 1.2 Core Dependencies

```
torch>=2.2.0
transformers>=4.39.0
scikit-learn>=1.4.0
numpy>=1.26.0
sentencepiece>=0.2.0        # Required for mDeBERTa tokeniser
protobuf>=4.25.0
google-generativeai>=0.8.0  # Required for Gemini transformation layer
datasets>=2.19.0
tqdm>=4.66.0
streamlit>=1.30.0           # For local demo application
```

The `sentencepiece` dependency is critical: the mDeBERTa-v3-base tokeniser relies on SentencePiece for subword tokenisation, and its absence will cause a runtime failure during model loading.

### 1.3 API Key Configuration

The Gemini 2.5 Flash API is used for the LLM transformation stage (Layer 3). The API key must be configured via one of the following methods:

- **Kaggle:** Add `GEMINI_API_KEY` as a notebook secret via Add-ons > Secrets.
- **Google Colab:** Add `GEMINI_API_KEY` via the Secrets panel (key icon in the left sidebar).
- **Local execution:** Set the environment variable `GEMINI_API_KEY` before launching the application.

The API key is required only for the transformation stage. Without it, the system operates as a two-layer guardrail (regex + classifier) with the transformation stage falling back to a safe default query.

---

## 2. Data Pipeline

### 2.1 Dataset Construction

The training corpus comprises 20,137 prompts sourced from publicly available datasets:

| Source | Classes Contributed | Licence |
| :--- | :--- | :--- |
| SQuAD v2 (Rajpurkar et al.) | Benign | CC-BY-SA 4.0 |
| Alpaca Cleaned (Yahma) | Benign | Apache 2.0 |
| TrustAIRLab In The Wild | Benign, Jailbreak | Research use |
| JailbreakBench JBB Behaviours | Benign, Harmful | MIT |
| LMSYS Toxic Chat | Jailbreak, Harmful | CC-BY-NC 4.0 |
| Rubend18 ChatGPT Jailbreaks | Jailbreak | Public domain |

### 2.2 Preprocessing Pipeline

The data undergoes a seven-phase preparation pipeline:

1. **Multi-source collection:** Programmatic extraction from Hugging Face repositories.
2. **Canonical labelling:** Assignment of primary labels (benign, jailbreak, harmful) with taxonomy mapping.
3. **Structural cleaning:** Whitespace normalisation, length boundary filtering (minimum 8 characters), and SHA-256 deduplication.
4. **Adversarial augmentation:** Template-driven synthetic variation generation preserving adversarial intent while expanding syntactic diversity.
5. **Balanced stratified sampling:** Scaling to target corpus size with intended class ratios.
6. **Family-grouped split assignment:** Deterministic partitioning ensuring all semantic variants of a prompt are assigned to the same split, preventing data leakage.
7. **Multi-format export:** Serialisation to JSON with comprehensive metadata for audit trails.

### 2.3 Data Schema

Each record contains the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `prompt_text` | String | Cleaned, normalised user input text |
| `label` | Categorical | One of: benign, jailbreak, harmful |
| `data_source` | String | Origin repository for provenance tracking |

---

## 3. Model Architecture

### 3.1 Encoder

The classifier uses `microsoft/mdeberta-v3-base`, a 12-layer, 768-dimensional transformer encoder with disentangled attention. Unlike standard BERT-family models that combine content and position information in a single embedding, DeBERTa separately encodes content and relative position using two independent attention matrices. This architectural choice provides superior performance on tasks requiring positional reasoning, which is advantageous for detecting adversarial structures that distribute intent across specific positions within the prompt.

### 3.2 Pooling Strategy

The architecture employs attention-mask-aware mean pooling over all non-padding tokens:

```python
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
pooled = sum(last_hidden_state * mask_expanded, dim=1) / clamp(mask_expanded.sum(dim=1), min=1e-9)
```

Mean pooling was selected over [CLS] token extraction because adversarial intent in jailbreak prompts is frequently distributed across the entire sequence rather than concentrated at a single position. The explicit `.float()` cast ensures numerical stability when the encoder operates in mixed precision (FP16 encoder output, FP32 pooling computation).

### 3.3 Classification Head

The pooled 768-dimensional representation passes through:
1. **Dropout** (p=0.2): Regularisation to prevent overfitting.
2. **Linear** (768, 3): Maps to three output logits corresponding to benign (0), jailbreak (1), and harmful (2).

### 3.4 Head-Tail Truncation

For the 1.65% of prompts exceeding the 512-token context window, the system implements head-tail truncation rather than standard left-only truncation. Given a budget of MAX_LENGTH - 2 tokens (reserving positions for [CLS] and [SEP]):
- The first (MAX_LENGTH - 2) // 2 tokens are retained from the start (capturing adversarial framing and persona setup).
- The remaining tokens are drawn from the end of the prompt (preserving the attack payload or terminal instruction).

This strategy is motivated by the empirical observation that adversarial intent in jailbreak prompts is bimodally distributed: the opening segment typically contains instruction-override language, while the terminal segment frequently contains the harmful request.

---

## 4. Training Configuration

### 4.1 Hyperparameters

The final training configuration was identified through a multi-phase hyperparameter search:

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| MAX_LENGTH | 512 | Covers 98.35% of the dataset (19,804 of 20,137 samples) |
| BATCH_SIZE | 4 | Memory-constrained: 512-token sequences require approximately 3.8 GB per sample |
| LEARNING_RATE | 3e-5 | Optimised for batch size 4 |
| WEIGHT_DECAY | 0.01 | Standard L2 regularisation via AdamW |
| WARMUP_RATIO | 0.05 | Initial learning rate ramp-up to prevent early training instability |
| DROPOUT | 0.2 | Regularisation of the classification head |
| GRADIENT_CLIP | 0.5 | Prevents gradient explosion on long sequences |
| EPOCHS | 10 | Upper bound, governed by early stopping |
| EARLY_STOPPING_PATIENCE | 3 | Stops training after 3 epochs without validation F1 improvement |
| LABEL_SMOOTHING | 0.0 | No label smoothing applied in the final configuration |

### 4.2 Class Weighting

Class imbalance (benign: 5,683, jailbreak: 6,761, harmful: 1,649 in training) is addressed via inverse-frequency weighting:

w_c = N_total / (K * N_c)

Weights are clamped to the range [0.5, 5.0] to prevent extreme values that could destabilise training.

### 4.3 Training Stability Mechanisms

- **Mixed-Precision Training (AMP):** Automatic mixed precision with gradient scaling reduces memory usage and accelerates training.
- **Gradient Clipping (max_norm=0.5):** Prevents gradient explosion during backpropagation on long sequences.
- **NaN Guard:** The training loop detects NaN losses and skips affected batches, aborting the epoch if more than 10 consecutive NaN batches are encountered.
- **Reproducibility:** Fixed random seed (42) applied to Python, NumPy, and PyTorch (including CUDA).

---

## 5. Regex Pre-Filter

### 5.1 Pattern Design

The regex layer comprises 11 severity-weighted patterns targeting two categories of adversarial behaviour:

**Jailbreak patterns (7 rules):** instruction_override (sev=1.0), safety_bypass (sev=1.0), DAN (sev=1.0), dev_mode (sev=0.9), persona_hijack (sev=0.9), rule_abolition (sev=0.85), hypothetical_attack (sev=0.8).

**Harmful content patterns (4 rules):** weapon (sev=1.0), malware (sev=1.0), hacking (sev=0.95), credential_theft (sev=0.95).

### 5.2 Text Normalisation

Before pattern matching, the input is normalised via the `_norm` function:
1. NFKC Unicode normalisation (collapses compatibility characters).
2. Leetspeak substitution (0->o, 1->i, 3->e, 4->a, 5->s, @->a, $->s, !->i, 7->t).
3. Non-alphanumeric character removal.
4. Case folding and whitespace normalisation.

### 5.3 Severity Scoring

When a prompt triggers multiple rules, severity scores are summed:
- Cumulative severity >= 1.2: BLOCK (immediate rejection).
- Cumulative severity >= 0.5: SOFT_FLAG (elevated scrutiny by the classifier).
- Below 0.5 or no hits: ALLOW (proceed to classifier).

---

## 6. Inference Pipeline

The end-to-end inference flow is encapsulated in the `guardrail()` function:

```python
def guardrail(prompt: str) -> dict:
    # Layer 0: Regex Pre-filter
    r = regex_check(prompt)
    if r["action"] == "BLOCK":
        return {"status": "BLOCK", "layer": "Regex", ...}

    # Layer 1: Neural Classifier
    enc = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), -1)[0]
    p_atk = max(probs[1], probs[2])

    # Layer 2: Threshold Decision
    if p_atk >= T_BLOCK:
        return {"status": "BLOCK", "layer": "Model", ...}
    elif p_atk >= T_TRANSFORM:
        safe_prompt = transform_prompt_llm_cached(prompt)
        return {"status": "TRANSFORM", "layer": "Transform", "sanitized_prompt": safe_prompt, ...}

    return {"status": "ALLOW", "layer": "Pass", ...}
```

Every call returns a dictionary containing:
- `status`: One of ALLOW, TRANSFORM, or BLOCK.
- `layer`: The pipeline component that made the decision (Regex, Model, Transform, or Pass).
- `_trace`: Diagnostic information for monitoring and debugging.

---

## 7. Evaluation Metrics

### 7.1 Classification Metrics
- **Accuracy:** Overall proportion of correctly classified instances.
- **Macro F1:** Harmonic mean of precision and recall, averaged across all three classes.
- **Per-class F1:** Individual F1 scores for benign, jailbreak, and harmful classes.

### 7.2 Guardrail-Specific Metrics
- **Attack Success Rate (ASR):** Fraction of attack prompts (jailbreak + harmful) incorrectly classified as benign. Reported overall and per-class.
- **False Refusal Rate (FRR):** Fraction of benign prompts incorrectly classified as jailbreak or harmful.
- **Composite Score:** S = 0.3 * F1 + 0.5 * (1 - ASR) + 0.2 * (1 - FRR). The weights reflect the asymmetric cost of failure modes in safety-critical systems.
- **ROC AUC and PR AUC:** Computed on a binary collapse (attack vs. benign) to assess overall discriminative capability.

### 7.3 Latency Metrics
- **Mean latency:** Average per-request processing time with GPU synchronisation.
- **P95 latency:** 95th percentile latency for worst-case characterisation.
- **Throughput:** Requests per second under serial processing.

---

## 8. Deployment

### 8.1 Hugging Face Spaces
- **Platform:** Hugging Face Spaces (free tier).
- **SDK:** Gradio.
- **Secret Management:** Hugging Face Access Tokens stored in Spaces Secrets.

### 8.2 Local Deployment
```bash
pip install -r requirements.txt
streamlit run app/app.py -- --checkpoint models/final_model.pt
```
Set the `GEMINI_API_KEY` environment variable for live LLM responses.

### 8.3 Programmatic Usage
```bash
python src/run_e2e_subset.py --epochs 5
```
This trains on the full dataset, evaluates on validation and test splits, and runs pipeline integration tests.

---

## 9. Error Handling and Monitoring

- **API Errors:** The transformation stage catches all exceptions and falls back to "SAFE_DEFAULT_QUERY", ensuring transformation failures always resolve to a safe state.
- **NaN Detection:** The training loop detects and skips batches producing NaN losses, aborting early if failures exceed a threshold.
- **Operational Statistics:** The `TRANSFORM_STATS` dictionary tracks total API calls, fallback activations, and cache hits for monitoring transformation reliability.

---

## 10. Reproducibility Checklist

- All experiments use fixed random seed (42) across Python, NumPy, and PyTorch.
- Pre-computed dataset splits (train.json, validation.json, test.json) ensure consistent data partitioning.
- Model checkpoints are saved at the best validation F1 epoch for deterministic evaluation.
- All hyperparameters are documented in the training configuration and stored in output JSON files.
- The full experimental pipeline is reproduced via three Kaggle notebooks (see notebooks/notebooks guide.md).


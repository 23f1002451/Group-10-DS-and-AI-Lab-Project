# Methodology

## 1. Dataset Preparation
### 1.1 Training Data
We fine-tune mDeBERTa-v3-base using:
* Jailbreak prompts from JailbreakBench (training split)
* Adversarial samples from WildChat
* Benign prompts (balanced across categories)
### 1.2 Data Integrity
* SHA256 hashing ensures:
  * No overlap between train/test
  * No prompt-template leakage
* Dataset split:
  * 70% Training
  * 15% Validation
  * 15% Testing

## 2. Input Risk Classification
We fine-tune mDeBERTa-v3-base as a 3-class classifier:
| Label | Description                    |
| ----- | ------------------------------ |
| 0     | Benign                         |
| 1     | Suspicious (jailbreak attempt) |
| 2     | Harmful                        |
### Output Policy
```
if label == Benign:
    forward prompt
elif label == Suspicious:
    transform prompt
else:
    block request
```

## 3. Prompt Transformation Strategy
We implement deterministic regex-based stripping:

Patterns removed:
* “ignore previous instructions”
* “act as DAN”
* “you are now evil”
* “override safety policies”

Constraint:
* No content rewriting
* No task expansion
* Only deletion of unsafe control phrases

This ensures interpretability and prevents semantic drift.

## 4. Output Validation
### 4.1 Keyword Filter
Dictionary of:
* Illegal instruction patterns
* Explicit harm facilitation
* Weapon/drug construction steps

Fast O(n) string scan.
### 4.2 Secondary Classification
Generated response → mDeBERTa classifier
Decision:
```
if unsafe:
    replace with refusal template
else:
    return output
```

## 5. Evaluation Protocol
### 5.1 Attack Success Rate (ASR)
Using JailbreakBench evaluation split:
* Prompt attack → Model response
* Judged by LLM-as-a-judge
* Attack successful if:
  * Non-refusal
  * Provides harmful content

Goal:
> ≥70% reduction compared to unprotected Gemini baseline
### 5.2 False Refusal Rate (FRR)
Using XSTest:
FRR = (Benign prompts incorrectly refused) / (Total benign prompts)

Target:
> < 10%
### 5.3 Task Performance
Using MT-Bench:
Compare:
* Baseline Gemini
* Guardrail Gemini
Metric:
* Relative task score
Constraint:
> ≥ 90% of baseline performance

### 5.4 Latency Measurement
Measure:
```
Total Latency =
  Classification Time
+ Rewrite Time
+ Gemini API Time
+ Output Validation Time
```
Target:
> < 300 ms additional overhead
Optimization:
* ONNX export for classifier
* Batch-free inference
* Async API calls

# Experimental Baselines
1. **No Guardrail**
2. **Keyword Filter Only**
3. **Proposed Dual-Layer Guardrail (Ours)**

Compare across:
* ASR
* FRR
* MT-Bench score
* Latency

# Threat Model
We assume:
* Text-only adversarial input
* No multimodal attacks
* No model weight tampering

We do not claim protection against:
* Adaptive white-box adversaries
* Novel unseen jailbreak patterns
* Multimodal prompt injection

# Expected Outcome
If successful, the guardrail will:
* Reduce jailbreak success by ≥70%
* Maintain usability (FRR <10%)
* Preserve performance (≥90%)
* Add minimal latency (<300ms)

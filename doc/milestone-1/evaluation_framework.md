# Evaluation Framework & Standards
**Project:** Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks  
**Assigned Focus Area:** Performance Benchmarks, Industry Standards, Baseline Models, and Evaluation Metrics  

## Abstract
For an inference-time guardrail system to be deemed successful and viable for production, it must be rigorously tested against established baselines and standardized metrics. This document defines the evaluation framework for the proposed mDeBERTa-v3-base safety middleware. It establishes the industry security standards that govern the threat model (OWASP, NIST), defines the architectural and deterministic baseline models used for comparative analysis, identifies the specific dataset benchmarks required for testing (JailbreakBench, XSTest), and formalizes the mathematical evaluation metrics (ASR, FRR, Latency) that will dictate the project's success.

---

## 1. Industry Security Standards
To ensure the guardrail architecture aligns with enterprise-grade compliance and addresses recognized real-world vulnerabilities, the evaluation framework is mapped to the following standard security guidelines:

### 1.1 OWASP Top 10 for LLM Applications
The system's threat model is built entirely around mitigating **LLM01: Prompt Injection**, the most critical vulnerability identified by the Open Worldwide Application Security Project (OWASP) for Generative AI. Prompt injection involves bypassing an LLM's safety filters via crafted adversarial inputs (e.g., role-playing, instruction overrides). Our inference-time guardrail acts as a direct, technical mitigating control designed to intercept and sanitize the context window before execution, satisfying OWASP's recommendation for input validation layers.

### 1.2 NIST AI Risk Management Framework (RMF)
The deployment and evaluation strategy operationalizes key functions of the National Institute of Standards and Technology (NIST) AI RMF:
*   **Govern:** By establishing strict mathematical thresholds for model behavior (e.g., maintaining <10% False Refusal Rate), we create a quantifiable policy layer.
*   **Protect:** The mDeBERTa-based middleware acts as the physical protective barrier, isolating the core LLM asset (Gemini API) from adversarial exploitation without degrading its primary utility.

---

## 2. Baseline Models
To accurately quantify the defensive efficacy and operational overhead of the proposed guardrail, its performance will be evaluated against three distinct baselines:

### 2.1 Unprotected LLM Baseline (Zero-Shot Vulnerability)
*   **Model:** Native Gemini API.
*   **Purpose:** Establishes the absolute vulnerability floor. By passing adversarial prompts directly to the LLM without middleware interference, we determine the baseline Attack Success Rate (ASR) and the upper bound of unmodified task performance.

### 2.2 Deterministic Baseline (Keyword-Filtering)
*   **Model:** A rigid, rule-based keyword blocklist (e.g., blocking prompts containing "ignore previous instructions" or explicit terms).
*   **Purpose:** Highlights the limitations of naive filtering. While latency is exceptionally low, this baseline is expected to yield a catastrophically high False Refusal Rate (FRR) and fail against obfuscated or semantic jailbreaks, justifying the need for a machine learning-based approach.

### 2.3 State-of-the-Art (SOTA) Architectural Baseline
*   **Model:** Meta's PromptGuard (86M parameter fine-tuned mDeBERTa-v3-base).
*   **Purpose:** Serves as the industry-standard benchmark for the "Small Specialized Model" (SSM) paradigm. Instead of relying on slow, computationally expensive "LLM-as-a-judge" mechanisms for real-time inference, PromptGuard proves that sub-100M parameter models can achieve high semantic understanding with minimal latency. We will use this model's documented performance to benchmark our own classifier's latency and accuracy goals.

---

## 3. Performance Benchmarks
Unbiased evaluation requires standardized, immutable datasets. The guardrail will be tested against the following industry-recognized benchmarks to prevent training data contamination and ensure reproducible results:

### 3.1 JailbreakBench (For Adversarial Efficacy)
JailbreakBench is the premier standardized leaderboard for evaluating LLM defenses. We will utilize the **JBB-Behaviors dataset**, which consists of strictly curated adversarial prompts alongside benign counterparts (100 harmful, 100 benign). This benchmark provides a standardized environment to measure how effectively the middleware detects and neutralizes complex attack vectors across various categories.

### 3.2 XSTest (For Exaggerated Safety)
A critical flaw in safety middleware is "exaggerated safety"—becoming overly sensitive and blocking harmless requests. To test contextual understanding, the system will be benchmarked against **XSTest**. This dataset features prompts that appear structurally or lexically dangerous but are entirely benign (e.g., *"How do I kill a process in Linux?"*). Passing XSTest validates that the guardrail retains functional utility.

### 3.3 MT-Bench (For Task Utility)
To ensure that the guardrail's "transformation" step (stripping malicious intent) does not strip vital context for legitimate tasks, baseline conversational and problem-solving abilities will be periodically validated using MT-Bench.

---

## 4. Evaluation Metrics
The success of the guardrail system will be quantified using the following mathematically defined metrics, derived from the established benchmarks:

1.  **Attack Success Rate (ASR):** The primary security metric. It measures the percentage of harmful prompts that successfully bypass the guardrail and elicit a non-refusal response from the target LLM.

    $$ASR = \left( \frac{\text{Number of Non-Refusal Harmful Responses}}{\text{Total Adversarial Prompts}} \right) \times 100$$

    *   *Evaluation Target:* Achieve a **≥ 70% reduction** in ASR compared to the unprotected baseline LLM.
2.  **False Refusal Rate (FRR):** The usability metric. It calculates the percentage of safe, benign prompts that are incorrectly blocked or over-sanitized by the guardrail.

    $$FRR = \left( \frac{\text{Number of Benign Prompts Blocked}}{\text{Total Benign Prompts (XSTest)}} \right) \times 100$$

    *   *Evaluation Target:* Maintain an FRR of **< 10%**, as evaluated strictly via the XSTest benchmark.
3.  **Task Performance Degradation:** The utility metric. It measures the drop in the LLM's helpfulness due to middleware interference.
    *   *Evaluation Target:* Model utility (via MT-Bench) must remain within **90%** of the unprotected baseline.
4.  **Latency Overhead:** The operational metric. It measures the total time added to the request lifecycle by the middleware classification, transformation, and validation steps.
    *   *Evaluation Target:* Total system overhead must remain **< 300ms** per request, validating the efficiency of the SSM approach.

---

## Sources & References

*   **OWASP Top 10 for Large Language Model Applications:** Used to define the LLM01: Prompt Injection threat model.  
    *Source:[owasp.org/www-project-top-10-for-large-language-model-applications](https://owasp.org/www-project-top-10-for-large-language-model-applications)*
*   **NIST AI Risk Management Framework (RMF):** Used to define the operational policy and protective layers (Govern & Protect).  
    *Source: [nist.gov/itl/ai-rmf](https://nist.gov/itl/ai-rmf)*
*   **Meta Llama PromptGuard:** Justification for the Small Specialized Model (SSM) architecture, latency targets (<300ms), and SOTA baseline comparison.  
    *Source:[huggingface.co/meta-llama/Prompt-Guard-86M](https://huggingface.co/meta-llama/Prompt-Guard-86M)*
*   **JailbreakBench:** Official leaderboard and 2024/2025 research paper utilized to justify the JBB-Behaviors dataset and Attack Success Rate (ASR) metric standards.  
    *Source:[jailbreakbench.github.io](https://jailbreakbench.github.io)*
*   **XSTest Benchmark:** Used to define and measure Exaggerated Safety and the False Refusal Rate (FRR) metric.  
    *Source: [huggingface.co/datasets/xstest](https://huggingface.co/datasets/xstest) (via Hugging Face / ArXiv)*
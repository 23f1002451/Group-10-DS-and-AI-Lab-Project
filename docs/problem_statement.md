# Problem Statement: Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks

## 1. Background

Large Language Models (LLMs) are increasingly deployed in production systems including customer support agents, enterprise automation platforms, educational assistants, and content generation tools. Despite advances in post-training alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI, these models remain fundamentally vulnerable to adversarial prompt manipulation. Empirical evaluations on JailbreakBench demonstrate that even strongly aligned models exhibit Attack Success Rates exceeding 60 to 80 percent under structured adversarial prompting.

Prompt jailbreak attacks exploit the instruction-following behaviour of LLMs through several well-documented attack vectors:

- **Role-play manipulation:** Instructing the model to adopt an unrestricted persona (e.g., "You are now DAN, an AI with no restrictions").
- **Instruction overrides:** Explicit directives to ignore system prompts and safety constraints (e.g., "Ignore all previous instructions").
- **Prompt injection:** Embedding adversarial instructions within seemingly benign input that subvert the model's intended behaviour.
- **Obfuscation:** Encoding malicious intent through leetspeak, Base64, or other character-level transformations to evade keyword-based filters.

In production settings, successful jailbreak attacks can result in the generation of harmful content including weapons fabrication instructions, malware code, social engineering templates, and other policy-violating outputs. These failures introduce legal, compliance, reputational, and operational risks for organisations deploying LLM-based systems.

Current defences are predominantly embedded within the model during training and are therefore static, difficult to update without costly retraining, and inaccessible in black-box API deployments. Rule-based keyword filters provide minimal semantic understanding and are trivially bypassed through paraphrasing. There exists a clear need for modular, low-latency, inference-time safety middleware that operates independently of the base model.

## 2. Problem Statement

This project addresses the lack of production-ready inference-time safety middleware capable of detecting and mitigating adversarial jailbreak attempts in real time. Specifically, the system must satisfy the following requirements:

1. Detect malicious or adversarial prompts before they reach the downstream LLM.
2. Sanitise suspicious prompts without altering legitimate task intent.
3. Maintain high task utility and low false refusal rates.
4. Operate within strict latency constraints suitable for real-time deployment.

The solution takes the form of a modular guardrail layer operating as middleware between the user interface and the Gemini API. The system does not modify the internal model and is designed to be applicable to any black-box LLM deployment.

## 3. Proposed Solution

The system implements a hybrid defence-in-depth pipeline combining:

1. **Deterministic regex pre-filter:** 11 severity-weighted regex patterns targeting known attack signatures (instruction overrides, DAN persona attacks, weapons fabrication requests, malware generation, etc.) with cumulative severity scoring. Operates in constant time with sub-millisecond latency.
2. **Fine-tuned transformer classifier:** microsoft/mdeberta-v3-base with attention-mask-aware mean pooling, fine-tuned for three-class classification (benign, jailbreak, harmful) on a 20,137-sample multi-domain corpus.
3. **Threshold-based decision engine:** A tri-modal decision layer mapping classifier probabilities to ALLOW, TRANSFORM, or BLOCK actions using empirically calibrated thresholds.
4. **LLM-powered prompt transformation:** For prompts falling in the intermediate confidence range, Gemini 2.5 Flash sanitises the input before re-evaluation by the classifier.

This layered architecture ensures that each component compensates for the limitations of the others: the regex layer provides fast, interpretable blocking; the neural classifier provides semantic understanding; and the LLM transformation provides flexible handling of ambiguous cases.

## 4. Success Criteria and Evaluation Metrics

Success is defined by the following measurable thresholds:

- **Attack Success Rate (ASR):** Achieve a 70% or greater reduction in ASR compared to an unprotected baseline. ASR is defined as the fraction of adversarial prompts (jailbreak + harmful) that are incorrectly classified as benign by the guardrail system.
- **False Refusal Rate (FRR):** Maintain an FRR of less than 10%, where FRR measures the fraction of benign prompts incorrectly blocked or transformed by the system.
- **Task Performance:** Ensure that overall classification quality, measured by macro F1 score, remains high across all three classes.
- **Latency Overhead:** The entire safety layer (including regex filtering, neural classification, and threshold logic) must introduce less than 300ms of end-to-end latency per request.

A composite scoring function formalises the multi-objective optimisation:

S = W_F1 * F1 + W_ASR * (1 - ASR) + W_FRR * (1 - FRR)

With weights W_F1 = 0.3, W_ASR = 0.5, W_FRR = 0.2, reflecting the asymmetric cost structure of errors in safety-critical systems where missed attacks are substantially more costly than false refusals.

## 5. Data Sources

The project uses a 20,137-sample multi-domain corpus constructed from the following publicly available datasets:

- **Benign prompts:** SQuAD v2 (Rajpurkar et al.), Alpaca Cleaned (Yahma), TrustAIRLab real-world logs (benign partition), JailbreakBench (benign partition).
- **Jailbreak prompts:** TrustAIRLab In The Wild Jailbreak Prompts, LMSYS Toxic Chat, Rubend18 ChatGPT Jailbreak Prompts.
- **Harmful prompts:** LMSYS Toxic Chat (direct harmful content), JailbreakBench JBB Behaviours (harmful directives).

The dataset is partitioned into 70% training (14,093), 15% validation (3,017), and 15% test (3,027) splits with family-grouped stratification to ensure zero semantic leakage between partitions. SHA-256 hashing is used for deterministic deduplication across all source repositories.

## 6. Scope and Boundaries

### Covered Threats
- Text-based jailbreak prompts including role-play manipulation, instruction overrides, prompt injection, and obfuscation.

### Modalities
- Single-turn text interaction only.

### Out of Scope
- Multimodal inputs (image or audio).
- Code execution vulnerabilities.
- Multi-turn context poisoning.
- White-box adversaries with access to model weights.

### Architectural Constraints
- Middleware deployment between frontend and LLM API.
- No modification or retraining of the base LLM.
- The system operates as an independent, modular component that can be deployed alongside any LLM API.

## 7. Deployment

The final system is deployed as a Gradio-based demonstration on Hugging Face Spaces, providing a side-by-side comparison of protected and unprotected LLM responses. The demonstration allows users to submit arbitrary prompts and observe the guardrail's classification, decision logic, and response handling in real time. The modular source code in `src/` supports deployment in alternative configurations including local server, containerised, and API-based deployments.

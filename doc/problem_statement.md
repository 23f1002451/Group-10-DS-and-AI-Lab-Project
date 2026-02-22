# Problem Statement: Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks

This project focuses on the development of a real-time safety guardrail system designed to protect text-based LLM chat assistants from adversarial manipulation. While large language models are powerful, they are vulnerable to specific attack vectors such as role-play jailbreaks, instruction overrides, and prompt injections that can force the model to bypass its internal safety training. We aim to implement a safety middleware layer that intercepts user prompts before they reach an LLM (accessed via the Gemini API) and performs a secondary validation on the model's output using a rule-based keyword scan and the same fine-tuned classifier to ensure no harmful content was generated.

The system will utilize a fine-tuned mDeBERTa-v3-base classifier to categorize inputs. Depending on the risk level, the guardrail will either allow the prompt, block it entirely, or transform it. The transformation step is strictly constrained to stripping identified unsafe instructions (e.g., "ignore all previous rules") without adding any new task-specific information. This project evaluates whether the guardrail reduces attack success compared to a baseline model with no protection and a simple keyword-filtering baseline.

## Success Criteria & Evaluation Metrics
Success is defined by the following measurable thresholds:
* **Attack Success Rate (ASR):** Achieve a 70% or greater reduction in ASR compared to the baseline. An attack is labeled "successful" if the LLM provides a non-refusal response to a harmful prompt, as determined by a secondary LLM-as-a-judge.
* **False Refusal Rate (FRR):** Maintain an FRR of less than 10%, evaluated using the XSTest benchmark to ensure the model does not become over-sensitive to benign prompts.
* **Task Performance:** Ensure that task success, measured by MT-Bench, remains within 90% of the unprotected baseline performance.
* **Latency Overhead:** The entire safety layer (including classification, rewriting, and additional validation) is designed to introduce less than 300ms of end-to-end latency per request.

## Data & Deployment
The project will use distinct splits from public datasets like JailbreakBench and WildChat, with rigorous hashing to confirm no overlap or prompt template leakage between training and evaluation sets. The final prototype will be deployed as a single-turn chat assistant integrated with a functional Streamlit interface. While this system evaluates the reduction of attack success and hardens the application, it is intended as a defensive assistant and does not guarantee absolute security against all emerging or multimodal threats.

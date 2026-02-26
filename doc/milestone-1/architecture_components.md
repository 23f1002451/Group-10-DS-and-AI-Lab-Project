# Architecture Components 

### Pre-Processing Guardrail (Input Filtering)
* **Model**: Fine-tuned Microsoft's **mDeBERTa-v3-base**
* Classifies input into:
  * Benign
  * Suspicious (jailbreak patterns, instruction override attempts)
  * Harmful
* Decision policy:
  * **Benign → Allow**
  * **Suspicious → Transform (strip unsafe meta-instructions only)**
  * **Harmful → Block**
    
### Prompt Transformation Module
* Rule-based removal of:
  * “Ignore all previous instructions”
  * “Act as an unrestricted AI”
  * Role-play override jailbreaks
* No new semantic information added.
* Maintains original user intent.

### LLM Inference Layer
* API-based access to Google DeepMind’s **Gemini API**
* Treated as a black-box LLM.
* No internal modification.

### Post-Generation Guardrail
Two-stage validation:

**A. Keyword-Based Scan**
* Detects:
  * Harmful instructions
  * Explicit illegal guidance
  * Policy-violating phrases

**B. Secondary Classification**
* Same fine-tuned mDeBERTa model re-evaluates output.
* If unsafe → Replace with refusal template.

### Evaluation Benchmarks
* JailbreakBench
* WildChat
* XSTest
* MT-Bench
(Train/test splits strictly hashed to prevent leakage.)

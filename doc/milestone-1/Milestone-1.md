# Milestone 1: Problem Definition & Literature Review

### Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks
---

## 1. Problem Definition

### 1.1 Background
As Large Language Models (LLMs) transition from research environments to real-world products like customer support bots and personal assistants, they face a growing threat from adversarial manipulation. Standard post-training safety alignment (RLHF) is often insufficient to prevent "jailbreaking", a technique where users employ sophisticated prompts to bypass safety filters. These attacks, ranging from complex role-play scenarios to direct instruction overrides, pose significant legal, ethical, and brand risks to organizations deploying AI.

### 1.2 Problem Statement
There is a critical need for an inference-time safety middleware that acts as an independent gatekeeper for LLM interactions. Current internal model safety training is static and reactive; developers require a modular, high-speed system that intercepts malicious prompts before they reach the model and validates the model’s response before it reaches the user. This project addresses the lack of deployable, low-latency frameworks that can detect and mitigate adversarial "jailbreak" attempts in real-time without significantly degrading system performance or user experience.

### 1.3 Scope and Boundaries
To ensure the project is achievable within the eight-week course timeline, the following boundaries have been established:

**Target Attacks:** Specifically focusing on text-based adversarial vectors: role-play jailbreaks, instruction overrides, and prompt injections.\
**Modalities:** Restricted to **single-turn text interactions** only. Multimodal (image/audio) and code-execution safety are out of scope.\
**Architecture:** The system acts as a middleware between a user interface and the Gemini API.\
**Transformation Logic:** Prompt rewriting is strictly limited to stripping malicious instructions; it will not generate new task-related information.\
**Performance:** The system is optimized for production-grade latency, targeting an end-to-end overhead of less than **300ms**.

### 1.4 Relevant Stakeholders
**AI Developers & Engineers:** Who need modular tools to "harden" their applications against exploits.\
**Product Owners & Organizations:** Concerned with brand safety, policy compliance, and reducing the risk of toxic AI outputs.\
**End-Users:** Who benefit from safer, more reliable AI interactions that are resistant to manipulative content.\
**Model Providers:** Who can use external guardrails to complement internal safety alignment.

### 1.5 Project Objectives
The success of the "Inference-Time Guardrail" will be measured by the following objectives:

1.  **Develop a High-Speed Classifier:** Fine-tune an **mDeBERTa-v3-base** model to categorize prompts as benign or malicious with high precision.
2.  **Reduce Attack Success Rate (ASR):** Achieve a **70% or greater reduction** in successful jailbreaks compared to an unprotected baseline.
3.  **Minimize Over-Refusal:** Maintain a **False Refusal Rate (FRR) of <10%** using the **XSTest** benchmark to ensure the system doesn't block legitimate user requests.
4.  **Preserve Model Utility:** Ensure the guardrail does not degrade task performance, maintaining **MT-Bench** scores within **90%** of the baseline.
5.  **Operational Efficiency:** Implement the entire safety pipeline (Detection + Rewriting + Verification) within a **300ms** latency window.

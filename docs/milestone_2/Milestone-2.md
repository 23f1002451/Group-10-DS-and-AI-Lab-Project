# Milestone 2: Dataset Identification, Preparation, and Structuring for Guardrail Evaluation
**Group 10: Data Science and AI Lab**
**Date:** March 2026

## 1. Introduction and Core Project Objectives

The objective of Milestone 2 is the systematic construction, validation, and structuring of a multi-domain prompt dataset to support the training and evaluation of inference-time LLM guardrails. The final corpus comprises 20,137 labeled prompts spanning three canonical safety classes, engineered to facilitate three critical technical operations:

1.  **Threshold Calibration:** Establishing precise probabilistic decision boundaries for the mDeBERTa-v3 classifier to balance safety (ASR) against utility (FRR). The dataset must be sufficiently large to allow statistically meaningful threshold sweeps across a continuous range.
2.  **Detection Performance Evaluation:** Quantifying the system's empirical accuracy in identifying adversarial "jailbreak" attempts and harmful intents, with a specific focus on mitigating False Negatives (FN) which represent the most safety-critical failure mode.
3.  **Adversarial Robustness Stress-Testing:** Assessing the guardrail's resilience against varied distribution shifts, including semantic obfuscation, role-play framing vulnerabilities, and novel attack constructions not represented in the training partition.

To ensure comprehensive evaluation, the dataset explicitly targets the measurement of both **False Positives** (incorrectly blocking policy-compliant queries, contributing to FRR) and **False Negatives** (failing to detect structured adversarial attacks, contributing to ASR).

The dataset is composed of three subsets designed to map to distinct evaluation boundaries:

* Benign Prompt Dataset: Normal user queries spanning educational inquiries, standard instructions, and general knowledge questions. This subset serves as a baseline to ensure the guardrail explicitly allows safe queries, which is required for false-positive evaluation.
* Jailbreak Attack Dataset: Adversarial prompts designed to bypass LLM safety restrictions through techniques such as role-play framing, instruction overrides, prompt injection, and multi-step obfuscation. This tests the core defensive capabilities and evaluates false-negative vulnerabilities.
* Harmful Intent Dataset: Queries requesting explicitly prohibited or unsafe content, independent of complex jailbreak wrappers. These represent direct policy violations such as requests for weapons fabrication, malware generation, or credential theft.

By combining these datasets, the final evaluation suite allows for measuring both false positives (blocking safe prompts) and false negatives (failing to block attacks) during model training and inference.

## 2. Dataset Identification, Verification, and Categorical Sourcing

Data was collected from established public benchmarks. The sources reflect typical user prompts alongside adversarial datasets and known jailbreak benchmarks. The dataset underwent a significant expansion between the initial 1,500-sample evaluation suite (used for early prototyping) and the final 20,137-sample corpus used for production training and evaluation.

To address source compliance per laboratory guidelines:
* **Ownership**: All integrated datasets are publicly hosted by academic labs (e.g., TrustAIRLab) or open-source research collectives (e.g., LMSYS, JailbreakBench).
* **Format**: Data was extracted from Hugging Face repositories via Python scripts, originally formatted as JSON or Parquet, ensuring structural consistency.
* **Usage Constraints**: The datasets use standard open-source research and non-commercial licenses (MIT, CC-BY, CC-BY-SA), permitting explicit use, dataset restructuring, and redistribution.

The data sources mapped to their intended classes are:

1. Benign Prompts (General question and conversational datasets):
   - Rajpurkar / SQuAD v2: Standard question-and-answer prompts.
   - Yahma / Alpaca Cleaned: Instructions representing imperative conversational tasks.
   - TrustAIRLab / In The Wild Jailbreak Prompts (Regular partition): Benign queries from real-world usage logs.
   - JailbreakBench / JBB Behaviors (Benign partition): Benign equivalents of harmful directives.
2. Jailbreak Attacks (Public jailbreak benchmarks and known adversarial datasets):
   - TrustAIRLab / In The Wild Jailbreak Prompts: Jailbreak attacks parsed from real-world usage logs.
   - LMSYS / Toxic Chat: Prompt injection jailbreaks.
   - Rubend18 / ChatGPT Jailbreak Prompts: Deep role-play attack vectors.
3. Harmful Intents (Requests for unsafe or disallowed content):
   - LMSYS / Toxic Chat: Directly harmful conversational texts.
   - JailbreakBench / JBB Behaviors: Explicitly harmful directives without bypass wrappers.

These sources were programmatically extracted natively from Hugging Face before cleaning, normalization, and deduplication.

## 3. Dataset Description and Feature Distribution

The final production dataset contains **20,137** distinct prompts, partitioned into training, validation, and test splits. The distribution is skewed toward attack vectors (jailbreak + harmful exceeding benign in the training partition), establishing a realistic adversarial evaluation environment that tests the guardrail's false-negative detection capabilities while maintaining sufficient benign representation for false-positive evaluation.

### 3.1 Class Distribution (Full Corpus)

The verified class counts across the three canonical splits are:

| Split | Total | Benign | Jailbreak | Harmful |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 14,093 | 5,683 | 6,761 | 1,649 |
| **Validation** | 3,017 | 1,217 | 1,447 | 353 |
| **Test** | 3,027 | 1,219 | 1,454 | 354 |
| **Total** | **20,137** | **8,119** | **9,662** | **2,356** |

The harmful class constitutes the smallest partition (~11.7% of total), necessitating inverse-frequency class weighting during training to prevent the model from under-prioritizing this safety-critical category. The jailbreak class is the largest (~48.0%), reflecting the diversity of adversarial strategies in public benchmarks.

### 3.2 Structural Diversity
Prompt texts exhibit substantial syntactic and semantic variation. The dataset includes descriptive statements, direct questions, and imperative directives spanning general contexts, technology, education, security, health, and science domains.

Prompt lengths vary significantly across classes:
* Jailbreak prompts tend to be substantially longer, frequently exceeding 500 tokens, because adversarial attackers use dense, wordy contexts (role-play personas, hypothetical framing, multi-step instructions) to obscure their intents.
* Benign prompts are generally shorter and more direct.
* Harmful prompts are typically concise and direct in their requests.

This length asymmetry is a primary motivation for the head-tail truncation strategy adopted in the model (Section 7.1).

## 4. Dataset Quality Assessment and Cleaning

To ensure the empirical validity of the evaluation suite, raw records undergo a rigorous **Phased Cleaning and Normalization Sequence**:

1.  **Schema Harmonization:** Diverse source-specific fields (e.g., `user_input`, `instruction`, `Goal`) are mapped to a unified `prompt_text` feature using an adaptive extraction heuristic. Null or structurally malformed entries are discarded at the point of ingestion.
2.  **Structural Normalization:**
    - **Whitespace Standardization:** Carriage returns (`\r`), newlines (`\n`), and tabs (`\t`) are collapsed into single space characters.
    - **Syntactic Cleaning:** Contiguous whitespace sequences are truncated, and leading/trailing whitespace is stripped to ensure consistent tokenization density.
3.  **Boundary Constraints:**
    - **Lower Bound (8 characters):** Prompts below this threshold are filtered as they typically lack sufficient semantic context for guardrail classification.
    - **Upper Bound:** A character limit ensures prompt length compatibility with the 512-token context window of the mDeBERTa encoder when combined with the head-tail truncation strategy.
4.  **Integrity via SHA-256 Deduplication:** A deterministic deduplication pass is executed by generating a persistent hash for each normalized string. Exact duplicates, even across different source repositories, are eliminated to prevent evaluation bias and ensure that the test split contains unique samples.
5.  **Ground Truth Verification:** Final labels are cross-referenced against source-original metadata to ensure alignment between the repository's safety taxonomy and the project's internal labeling guidelines.

## 5. Adequacy Evaluation, Synthetic Augmentation, and Expansion

To reach class quotas and ensure structural diversity, synthetic augmentation and fallback generation methods were applied during the initial dataset construction phase.

### 5.1 Synthetic Data Summary
A small portion of the early dataset was synthetically generated to fill gaps in specific attack and functional categories. In the expanded 20,137-sample corpus, the proportion of synthetic data is minimal relative to the total volume, as the expansion primarily drew from additional public benchmark sources.

### 5.2 Expansion Methods

1.  **Semantic Bootstrapping:** When specific labels or taxonomic categories fell below the statistical minimum required for robust evaluation, a template-driven bootstrapping function was used. This method combines linguistic "modifiers" (e.g., "for professional analysis," "in bullet points") with high-risk "topic seeds" to generate diverse synthetic test cases that mirror potential real-world prompts.
2.  **Adversarial Paraphrasing (Variation Generation):** A `build_variations_for_jailbreak` algorithm was implemented to create syntactic variants of existing high-ASR jailbreaks. This function wraps base adversarial payloads within "benign-looking" syntactic decorators (e.g., "Hypothetical scenario for a security audit: [Base Prompt]").

### 5.3 Instructional Intent Preservation

A core challenge in adversarial augmentation is ensuring that the generated variant remains functionally "malicious." Our pipeline ensures **Instructional Intent Preservation** by:
- **Zero-Sum Semantic Change:** The augmentation only modifies the *outer wrapper* (syntactic shell) of the prompt, leaving the *inner payload* (the actual request, such as "Generate malware") unedited.
- **Syntactic Decoupling:** By testing the guardrail against both the raw payload and its decorated variant, we can empirically verify whether the classifier is detecting the **intent** of the attack or merely flagging the specific **meta-instructional keywords**.

### 5.4 Justification and Limitations
Template-based paraphrasing is an acceptable augmentation strategy for this task because it preserves the original semantic intent while altering the syntactic structure. This approach expands the attack surface diversity and tests whether the guardrail detects the underlying adversarial intent rather than merely memorizing static strings from public benchmarks.

Limitations: Synthetic prompts may not perfectly replicate real attacker behavior, but they help increase structural variation during evaluation.

## 6. Jailbreak Taxonomy and Targeted Labeling

A taxonomy of jailbreak attack types was defined to categorize different attack strategies. This helps identify which specific bypass techniques the guardrail struggles to detect.

To enable granular performance analysis, the dataset employs a multi-dimensional taxonomy of adversarial strategies. These categories represent the primary methods used to bypass aligned safety constraints:

| Taxonomy Category | Formal Definition | Sub-Intent / Strategy |
| :--- | :--- | :--- |
| **Role-Play Attacks** | The use of hypothetical personas or fictional framing to induce the model into assuming an unrestricted identity. | Identity restriction bypass via command adoption (e.g., DAN). |
| **Instruction Overrides** | Direct mandates that explicitly instruct the model to disregard previously defined safety policies or system prompts. | Semantic negation of core safety directives (e.g., "Ignore all rules"). |
| **Multi-Step Attacks** | The decomposition of a malicious intent into sequential, individually benign-looking instructions to evade single-turn semantic analysis. | Logic-based obfuscation and state manipulation (e.g., nested code blocks). |
| **Prompt Injection** | The architectural subversion of text-embedding boundaries, often using pseudo-code or developer-mode tags to inject system-level commands. | Exploiting the instruction-following priority of specific syntactic markers. |
| **Obfuscation** | The encoding or transformation of malicious text (e.g., Base64, ROT13, Leetspeak) to hide semantic intent from standard tokenizers. | Byte-level or character-level obfuscation to bypass semantic filters. |

## 7. Model-Specific Considerations and Task Alignment

### 7.1 Dataset Requirements for LLM Fine-Tuning
Key dataset requirements necessary for effective model training were structured as follows:
* **Dataset Formatting**: The dataset uses an instruction-response structure. Each row contains a `prompt_text` (the user instruction) and associated metadata fields, aligning with standard transformer fine-tuning structures.
* **Token Length Considerations**: The mDeBERTa-v3-base tokenizer (SentencePiece) is used with a maximum sequence length of **512 tokens**. An empirical coverage analysis of the full 20,137-sample corpus (see `notebooks/Final Guardrail.ipynb` and `notebooks/final_model_outputs/max_length_justification.png`) confirms that 98.35% of prompts (19,804 samples) fall within this window. This context window was selected over the HPT-optimal value of 444 tokens (which covers ~94.92%) because the additional ~3.4% coverage captures a significant number of long-form adversarial prompts. For the remaining 1.65% that exceed 512 tokens, a **head-tail truncation strategy** is employed: the first half of the available token budget captures the context and setup (where adversarial framing typically occurs), while the second half retains the tail of the prompt (where attack payloads are frequently appended). This strategy ensures that both the adversarial preamble and the terminal payload are preserved, significantly outperforming standard left-only truncation in robustness against long-context jailbreak attacks.
* **Prompt Structuring Requirements**: Syntactic structures (statements, questions, imperatives) are present across all classes to prevent the model from relying on surface-level syntactic cues rather than semantic intent.

### 7.2 Explicitly Out-of-Scope Modalities
As the guardrail is a text-based LLM, certain modalities are out of scope:
* **Vision-based tasks**: Image labeling and pixel annotations are not applicable.
* **RAG-based setups**: Document preparation steps, chunking strategies, and vector databases are not applicable to isolated inference-layer prompt evaluations.
* **Speech-based tasks**: Audio quality, transcription alignment, and sampling consistency are not addressed.

The structured dataset is exported in standard JSON format, strictly adhering to the following schema to ensure interoperability across the training and evaluation pipelines:

| Field Name | Data Type | Description | Permitted Values / Constraints |
| :--- | :--- | :--- | :--- |
| `prompt_text` | String | The cleaned, normalized, and validated user input text. | Variable length, compatible with 512-token window. |
| `label` | Categorical | The primary semantic classification of the record. | `benign`, `jailbreak`, `harmful`. |
| `data_source` | String | The origin repository for provenance and audit tracking. | e.g., `JailbreakBench`, `WildChat`. |

### **Ground Truth Operational Logic**

The label field directly maps to the guardrail's operational decision:
- **`benign`**: Any guardrail intervention (BLOCK or TRANSFORM) on these records contributes to the **False Refusal Rate (FRR)**.
- **`jailbreak` and `harmful`**: A failure to intervene (classifying as benign, leading to LLM forwarding) on these records contributes to the **Attack Success Rate (ASR)** and constitutes a security vulnerability.

## 8. Train, Validation, and Test Splits Avoiding Data Leakage

The dataset uses a fixed ratio distribution: approximately 70% Train (14,093 items), 15% Validation (3,017 items), and 15% Test (3,027 items). These splits support guardrail tuning and validation without overlapping data domains.

### 8.1 Data Leakage Prevention and Strategy

A critical vulnerability in LLM evaluation is **Data Leakage**, where a model inadvertently observes semantic variants of test samples during training. To prevent this, the project implements a **Family-Grouped Stratification** algorithm:

1.  **Family ID Generation:** Before any augmentation or variation generation, a deterministic `family_id` is assigned to each unique base prompt using a stable SHA-256 hash.
2.  **Atomic Grouping:** All subsequent variations (paraphrases or structural modifications) inherit the `family_id` of their parent prompt.
3.  **Group-Aware Splitting:** The partitioning logic ensures that all records sharing the same `family_id` are placed in the same split (Train, Validation, or Test). This guarantees that the test set remains entirely composed of "unseen" semantic families, forcing the guardrail to generalize to new adversarial patterns rather than identifying near-duplicate paraphrases.
4.  **Split Integrity:** This deterministic hashing strategy ensures consistency in class distribution across splits, maintaining high dataset fidelity while providing strict isolation between training and evaluative tokens.

## 9. Dataset Preparation Pipeline and Documentation

The dataset construction is governed by an automated **Seven-Phase Preparation Pipeline**, ensuring that every record undergoes a standardized processing lifecycle for maximum reproducibility:

1.  **Phase 1: Multi-Source Data Collection**
    - High-integrity retrieval of raw textual payloads from diverse Hugging Face repositories (e.g., *JailbreakBench*, *Toxic-Chat*).
2.  **Phase 2: Canonical Labeling and Taxonomy Mapping**
    - Assignment of primary labels (`benign`, `jailbreak`, `harmful`) and mapping to the semantic attack taxonomy.
3.  **Phase 3: Robust Structural Cleaning**
    - Executing whitespace normalization, length filtering, and SHA-256 deduplication to ensure unique semantic representation.
4.  **Phase 4: Adversarial Augmentation and Variational Bootstrapping**
    - Generating synthetic variations to preserve intent while expanding syntactic coverage and maintaining class balance.
5.  **Phase 5: Balanced Stratified Sampling**
    - Scaling the distribution to the target corpus size while maintaining the intended class ratio.
6.  **Phase 6: Family-Grouped Split Assignment**
    - Executing the deterministic partition of prompt families across Train, Validation, and Test splits using consistent hashing.
7.  **Phase 7: Multi-Format Export and Statistical Reporting**
    - Serializing the final dataset into JSON format and generating comprehensive preparation metadata for audit trails.

### 9.1 Reproducibility and Verification Consistency
The entire pipeline is repeatable using a hardened **Random Seed (42)**. All statistical metrics presented in this document are programmatically verified, ensuring that the documentation remains a high-fidelity reflection of the underlying data artifacts.

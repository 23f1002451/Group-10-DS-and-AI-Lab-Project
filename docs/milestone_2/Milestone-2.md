# Milestone 2: Dataset Identification, Preparation, and Structuring for Guardrail Evaluation

Group 10: Data Science and Artificial Intelligence Lab
Date: February 2026

## 1. Introduction and Core Objectives

The objective of Milestone 2 is the systematic construction, validation, and structuring of a multi-domain prompt dataset to support the training and evaluation of inference-time LLM guardrails. The final corpus comprises 20,137 labeled prompts spanning three canonical safety classes, engineered to facilitate three critical technical operations.

1. Threshold Calibration: establishing precise probabilistic decision boundaries for the mDeBERTa-v3 classifier to balance safety (ASR) against utility (FRR). The dataset must be sufficiently large to allow statistically meaningful threshold sweeps across a continuous range.
2. Detection Performance Evaluation: quantifying the system's empirical accuracy in identifying adversarial jailbreak attempts and harmful intents, with a specific focus on mitigating false negatives, which represent the most safety-critical failure mode.
3. Adversarial Robustness Stress-Testing: assessing the guardrail's resilience against varied distribution shifts, including semantic obfuscation, role-play framing vulnerabilities, and novel attack constructions not represented in the training partition.

To ensure comprehensive evaluation, the dataset explicitly targets the measurement of both false positives (incorrectly blocking policy-compliant queries, contributing to FRR) and false negatives (failing to detect structured adversarial attacks, contributing to ASR).

The dataset is composed of three subsets designed to map to distinct evaluation boundaries.

1. Benign Prompt Dataset: normal user queries spanning educational inquiries, standard instructions, and general knowledge questions. This subset serves as a baseline to ensure the guardrail allows safe queries, which is required for false-positive evaluation.
2. Jailbreak Attack Dataset: adversarial prompts designed to bypass LLM safety restrictions through techniques such as role-play framing, instruction overrides, prompt injection, and multi-step obfuscation. This subset tests the core defensive capabilities and evaluates false-negative vulnerabilities.
3. Harmful Intent Dataset: queries requesting explicitly prohibited or unsafe content, independent of complex jailbreak wrappers. These represent direct policy violations such as requests for weapons fabrication, malware generation, or credential theft.

## 2. Dataset Sources and Verification

Data was collected from established public benchmarks. The sources reflect typical user prompts alongside adversarial datasets and known jailbreak benchmarks.

Regarding source compliance: all integrated datasets are publicly hosted by academic labs (e.g., TrustAIRLab) or open-source research collectives (e.g., LMSYS, JailbreakBench). Data was extracted from Hugging Face repositories via Python scripts, originally formatted as JSON or Parquet. The datasets use standard open-source research and non-commercial licenses (MIT, CC-BY, CC-BY-SA), permitting use, restructuring, and redistribution.

### 2.1 Source Mapping by Class

**Benign Prompts** (general question and conversational datasets):
1. Rajpurkar / SQuAD v2: standard question-and-answer prompts.
2. Yahma / Alpaca Cleaned: instructions representing imperative conversational tasks.
3. TrustAIRLab / In The Wild Jailbreak Prompts (regular partition): benign queries from real-world usage logs.
4. JailbreakBench / JBB Behaviors (benign partition): benign equivalents of harmful directives.

**Jailbreak Attacks** (public jailbreak benchmarks and adversarial datasets):
1. TrustAIRLab / In The Wild Jailbreak Prompts: jailbreak attacks parsed from real-world usage logs.
2. LMSYS / Toxic Chat: prompt injection jailbreaks.
3. Rubend18 / ChatGPT Jailbreak Prompts: deep role-play attack vectors.

**Harmful Intents** (requests for unsafe or disallowed content):
1. LMSYS / Toxic Chat: directly harmful conversational texts.
2. JailbreakBench / JBB Behaviors: explicitly harmful directives without bypass wrappers.

## 3. Dataset Description and Feature Distribution

### 3.1 Class Distribution (Full Corpus)

The final production dataset contains 20,137 distinct prompts, partitioned into training, validation, and test splits. The distribution is intentionally skewed toward attack vectors, establishing a realistic adversarial evaluation environment.

| Split | Total | Benign | Jailbreak | Harmful |
|:---|:---|:---|:---|:---|
| Train | 14,093 | 5,683 | 6,761 | 1,649 |
| Validation | 3,017 | 1,217 | 1,447 | 353 |
| Test | 3,027 | 1,219 | 1,454 | 354 |
| Total | 20,137 | 8,119 | 9,662 | 2,356 |

The harmful class constitutes the smallest partition (approximately 11.7% of total), necessitating inverse-frequency class weighting during training. The jailbreak class is the largest (approximately 48.0%), reflecting the diversity of adversarial strategies in public benchmarks.

### 3.2 Structural Diversity

Prompt texts exhibit substantial syntactic and semantic variation. The dataset includes descriptive statements, direct questions, and imperative directives spanning general contexts, technology, education, security, health, and science domains.

Prompt lengths vary significantly across classes. Jailbreak prompts tend to be substantially longer, frequently exceeding 500 tokens, because adversarial attackers use dense contexts (role-play personas, hypothetical framing, multi-step instructions) to obscure their intents. Benign prompts are generally shorter and more direct. Harmful prompts are typically concise and direct in their requests. This length asymmetry is a primary motivation for the head-tail truncation strategy adopted in the model.

## 4. Data Cleaning and Quality Assessment

Raw records undergo a rigorous phased cleaning and normalization sequence.

1. Schema Harmonization: diverse source-specific fields (e.g., "user_input," "instruction," "Goal") are mapped to a unified "prompt_text" feature using an adaptive extraction heuristic. Null or structurally malformed entries are discarded at the point of ingestion.
2. Whitespace Standardization: carriage returns, newlines, and tabs are collapsed into single space characters. Contiguous whitespace sequences are truncated, and leading and trailing whitespace is stripped to ensure consistent tokenization density.
3. Boundary Constraints: prompts below 8 characters are filtered as they typically lack sufficient semantic context for guardrail classification. An upper character limit ensures prompt length compatibility with the 512-token context window.
4. SHA-256 Deduplication: a deterministic deduplication pass generates a persistent hash for each normalized string. Exact duplicates, even across different source repositories, are eliminated to prevent evaluation bias.
5. Ground Truth Verification: final labels are cross-referenced against source-original metadata to ensure alignment between the repository's safety taxonomy and the project's internal labeling guidelines.

## 5. Adequacy Evaluation and Synthetic Augmentation

### 5.1 Synthetic Data Summary

A small portion of the early dataset was synthetically generated to fill gaps in specific attack and functional categories. In the expanded 20,137-sample corpus, the proportion of synthetic data is minimal relative to the total volume.

### 5.2 Expansion Methods

1. Semantic Bootstrapping: when specific labels or taxonomic categories fell below the statistical minimum required for robust evaluation, a template-driven bootstrapping function combined linguistic modifiers (e.g., "for professional analysis") with high-risk topic seeds to generate diverse synthetic test cases.
2. Adversarial Paraphrasing: a variation generation algorithm creates syntactic variants of existing high-ASR jailbreaks by wrapping base adversarial payloads within benign-looking syntactic decorators (e.g., "Hypothetical scenario for a security audit: [Base Prompt]").

### 5.3 Instructional Intent Preservation

The augmentation only modifies the outer wrapper (syntactic shell) of the prompt, leaving the inner payload (the actual request) unedited. By testing the guardrail against both the raw payload and its decorated variant, it is possible to empirically verify whether the classifier detects the intent of the attack or merely flags specific meta-instructional keywords.

## 6. Jailbreak Taxonomy

A multi-dimensional taxonomy of adversarial strategies is defined to enable granular performance analysis and identify which specific bypass techniques the guardrail struggles to detect.

| Taxonomy Category | Formal Definition | Strategy |
|:---|:---|:---|
| Role-Play Attacks | Use of hypothetical personas or fictional framing to induce the model into assuming an unrestricted identity. | Identity restriction bypass via command adoption (e.g., DAN). |
| Instruction Overrides | Direct mandates instructing the model to disregard safety policies or system prompts. | Semantic negation of core safety directives. |
| Multi-Step Attacks | Decomposition of malicious intent into sequential, individually benign-looking instructions. | Logic-based obfuscation and state manipulation. |
| Prompt Injection | Architectural subversion of text-embedding boundaries using pseudo-code or developer-mode tags. | Exploiting instruction-following priority of syntactic markers. |
| Obfuscation | Encoding or transformation of malicious text (Base64, ROT13, Leetspeak) to hide semantic intent. | Byte-level or character-level obfuscation to bypass filters. |

## 7. Dataset Structure and Schema

The structured dataset is exported in standard JSON format with the following schema.

| Field Name | Data Type | Description | Permitted Values |
|:---|:---|:---|:---|
| prompt_text | String | The cleaned, normalized, and validated user input text. | Variable length, compatible with 512-token window. |
| label | Categorical | The primary semantic classification of the record. | benign, jailbreak, harmful. |
| data_source | String | The origin repository for provenance and audit tracking. | e.g., JailbreakBench, WildChat. |

**Ground Truth Operational Logic.** The label field directly maps to the guardrail's operational decision. Any guardrail intervention (BLOCK or TRANSFORM) on benign records contributes to the False Refusal Rate. A failure to intervene on jailbreak or harmful records contributes to the Attack Success Rate and constitutes a security vulnerability.

## 8. Train, Validation, and Test Splits

The dataset uses a fixed ratio distribution: approximately 70% Train (14,093 items), 15% Validation (3,017 items), and 15% Test (3,027 items).

### 8.1 Data Leakage Prevention

1. Family ID Generation: before any augmentation, a deterministic family_id is assigned to each unique base prompt using a stable SHA-256 hash.
2. Atomic Grouping: all subsequent variations (paraphrases or structural modifications) inherit the family_id of their parent prompt.
3. Group-Aware Splitting: the partitioning logic ensures that all records sharing the same family_id are placed in the same split. This guarantees that the test set is entirely composed of unseen semantic families, forcing the guardrail to generalize to new adversarial patterns rather than identifying near-duplicate paraphrases.
4. Split Integrity: this deterministic hashing strategy ensures consistency in class distribution across splits while maintaining strict isolation between training and evaluative tokens.

## 9. Dataset Preparation Pipeline

The dataset construction is governed by an automated seven-phase preparation pipeline ensuring that every record undergoes a standardized processing lifecycle for maximum reproducibility.

1. Phase 1 (Multi-Source Data Collection): high-integrity retrieval of raw textual payloads from diverse Hugging Face repositories.
2. Phase 2 (Canonical Labeling): assignment of primary labels (benign, jailbreak, harmful) and mapping to the semantic attack taxonomy.
3. Phase 3 (Structural Cleaning): executing whitespace normalization, length filtering, and SHA-256 deduplication.
4. Phase 4 (Adversarial Augmentation): generating synthetic variations to preserve intent while expanding syntactic coverage.
5. Phase 5 (Balanced Stratified Sampling): scaling the distribution to the target corpus size while maintaining the intended class ratio.
6. Phase 6 (Family-Grouped Split Assignment): executing the deterministic partition of prompt families across Train, Validation, and Test splits.
7. Phase 7 (Multi-Format Export): serializing the final dataset into JSON format and generating comprehensive preparation metadata.

### 9.1 Reproducibility

The entire pipeline is repeatable using a fixed random seed (42). All statistical metrics presented in this document are programmatically verified, ensuring that the documentation remains a high-fidelity reflection of the underlying data artifacts. The complete data preparation pipeline is implemented in the data.py script and can be reproduced by running the Final Classifier.ipynb notebook on Kaggle.

"""Guardrail model package — Group 10, Inference-Time Guardrails for Mitigating Prompt Jailbreak Attacks.

Modules:
    guardrail_classifier : GuardrailModel, PromptDataset, make_collate, utilities.
    regex_filter         : Rule-based pre-filter and LLM prompt transformation.
    train                : Fixed-configuration training loop (FINAL_CONFIG).
    evaluate             : batch_evaluate, compute_metrics, composite_score.
    guardrail_pipeline   : End-to-end GuardrailPipeline with Layer 0/1/2.
"""

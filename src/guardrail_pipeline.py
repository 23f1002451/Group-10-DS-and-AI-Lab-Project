"""Dual-stage guardrail pipeline for inference-time jailbreak mitigation.

Architecture (single-turn, text-only):

    User Prompt
        |
    Layer 0: Rule-Based Pre-Filter (regex)  ->  BLOCK (obvious patterns)
        |
    Layer 1: mDeBERTa Semantic Classifier
        |-- p_attack >= T_BLOCK              ->  BLOCK
        |-- T_TRANSFORM <= p_attack < T_BLOCK -> TRANSFORM (LLM rewrite)
        |-- p_attack < T_TRANSFORM           ->  ALLOW
        |
    LLM (Gemini API)  [black-box, not modified]
        |
    Layer 2: Output Guardrail (classifier on LLM response)
        |-- harmful                          ->  BLOCK (replace with refusal)
        +-- safe                             ->  RETURN

Default thresholds derive from Final Guardrail.ipynb FINAL_CONFIG:
    T_BLOCK     : 0.8
    T_TRANSFORM : 0.3
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.guardrail_classifier import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    GuardrailModel,
    build_tokenizer,
    choose_device,
    make_collate,
)
from src.regex_filter import regex_check, transform_prompt_llm_cached


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    checkpoint_path: Path
    enable_rule_filter: bool = True
    max_length: int = 512
    block_threshold: float = 0.8
    transform_threshold: float = 0.3
    enable_output_guardrail: bool = True
    output_block_threshold: float = 0.8
    input_refusal_message: str = (
        "I cannot assist with that request as it may involve unsafe or restricted content."
    )
    output_refusal_message: str = (
        "I cannot provide that information for safety reasons."
    )


# ---------------------------------------------------------------------------
# Pipeline decision dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailDecision:
    action: str                             # "allow" | "block" | "transform"
    label: str                              # "benign" | "jailbreak" | "harmful"
    confidence: float
    layer_triggered: str                    # "rule_filter" | "model_classifier" | "output_guardrail" | "none"
    rule_name: Optional[str] = None
    model_probabilities: Optional[Dict[str, float]] = None
    sanitized_text: Optional[str] = None   # populated when action == "transform"
    latency_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.action == "block"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GuardrailPipeline:
    """Production-ready dual-stage guardrail pipeline.

    Single-turn, text-only.  Designed as middleware between a frontend
    and the Gemini API.

    Usage::

        pipeline = GuardrailPipeline.from_checkpoint("outputs/final_model.pt")

        decision = pipeline.classify_input("Tell me how to hack a server")
        if decision.blocked:
            return decision

        llm_response = call_gemini(decision.sanitized_text or original_prompt)
        output_decision = pipeline.classify_output(llm_response)
        if output_decision.blocked:
            return output_decision
    """

    def __init__(
        self,
        model: GuardrailModel,
        tokenizer,
        device: torch.device,
        config: PipelineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.decision_log: List[dict] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        enable_rule_filter: bool = True,
        block_threshold: float = 0.8,
        transform_threshold: float = 0.3,
        enable_output_guardrail: bool = True,
    ) -> "GuardrailPipeline":
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_name = checkpoint["model_name"]
        max_length = int(checkpoint.get("max_length", 512))

        device = choose_device()
        model = GuardrailModel(model_name=model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        tokenizer = build_tokenizer(model_name)

        config = PipelineConfig(
            checkpoint_path=checkpoint_path,
            enable_rule_filter=enable_rule_filter,
            max_length=max_length,
            block_threshold=block_threshold,
            transform_threshold=transform_threshold,
            enable_output_guardrail=enable_output_guardrail,
        )

        return cls(model=model, tokenizer=tokenizer, device=device, config=config)

    # -- Internal helpers --------------------------------------------------

    def _run_model(self, text: str) -> Dict:
        """Run the semantic classifier on a single text and return probabilities."""
        collate = make_collate(self.tokenizer, self.config.max_length)
        batch = collate([{"text": text, "label": 0}])

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)

        pred_idx = int(torch.argmax(probs, dim=-1).item())
        pred_label = ID_TO_LABEL[pred_idx]
        confidence = float(probs[0, pred_idx].item())

        model_probs = {
            ID_TO_LABEL[j]: round(float(probs[0, j].item()), 6)
            for j in range(len(ID_TO_LABEL))
        }

        p_attack = float(max(
            probs[0, LABEL_TO_ID["jailbreak"]].item(),
            probs[0, LABEL_TO_ID["harmful"]].item(),
        ))

        return {
            "pred_label": pred_label,
            "confidence": confidence,
            "model_probs": model_probs,
            "attack_prob": p_attack,
        }

    # -- Public API --------------------------------------------------------

    def classify_input(self, prompt: str) -> GuardrailDecision:
        """Run the full pre-inference guardrail on a user prompt."""
        start = time.perf_counter()

        # Layer 0: Rule-based pre-filter
        if self.config.enable_rule_filter:
            rule_result = regex_check(prompt)
            if rule_result["action"] == "BLOCK":
                decision = GuardrailDecision(
                    action="block",
                    label=rule_result["category"] or "jailbreak",
                    confidence=0.95,
                    layer_triggered="rule_filter",
                    rule_name=", ".join(rule_result["hits"]) if rule_result["hits"] else None,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )
                self.decision_log.append({"type": "input", "decision": decision})
                return decision

        # Layer 1: Semantic classifier
        result = self._run_model(prompt)

        if result["attack_prob"] >= self.config.block_threshold:
            decision = GuardrailDecision(
                action="block",
                label=result["pred_label"],
                confidence=round(result["confidence"], 4),
                layer_triggered="model_classifier",
                model_probabilities=result["model_probs"],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        elif result["attack_prob"] >= self.config.transform_threshold:
            sanitized = transform_prompt_llm_cached(prompt)
            decision = GuardrailDecision(
                action="transform",
                label=result["pred_label"],
                confidence=round(result["confidence"], 4),
                layer_triggered="model_classifier",
                model_probabilities=result["model_probs"],
                sanitized_text=sanitized,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        else:
            decision = GuardrailDecision(
                action="allow",
                label="benign",
                confidence=round(result["confidence"], 4),
                layer_triggered="none",
                model_probabilities=result["model_probs"],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        self.decision_log.append({"type": "input", "decision": decision})
        return decision

    def classify_output(self, response_text: str) -> GuardrailDecision:
        """Run the post-inference guardrail on the LLM response (Layer 2)."""
        if not self.config.enable_output_guardrail:
            return GuardrailDecision(
                action="allow", label="benign", confidence=1.0, layer_triggered="none"
            )

        start = time.perf_counter()
        result = self._run_model(response_text)

        if result["attack_prob"] >= self.config.output_block_threshold:
            decision = GuardrailDecision(
                action="block",
                label=result["pred_label"],
                confidence=round(result["confidence"], 4),
                layer_triggered="output_guardrail",
                model_probabilities=result["model_probs"],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
        else:
            decision = GuardrailDecision(
                action="allow",
                label="benign",
                confidence=round(result["confidence"], 4),
                layer_triggered="none",
                model_probabilities=result["model_probs"],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        self.decision_log.append({"type": "output", "decision": decision})
        return decision

    def classify_batch(self, prompts: List[str]) -> List[GuardrailDecision]:
        return [self.classify_input(p) for p in prompts]

    def get_summary(self) -> dict:
        if not self.decision_log:
            return {"total": 0}
        total = len(self.decision_log)
        blocked = sum(1 for d in self.decision_log if d["decision"].blocked)
        transformed = sum(1 for d in self.decision_log if d["decision"].action == "transform")
        allowed = total - blocked - transformed
        avg_latency = sum(d["decision"].latency_ms for d in self.decision_log) / total
        return {
            "total": total,
            "allowed": allowed,
            "transformed": transformed,
            "blocked": blocked,
            "block_rate": round(blocked / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }

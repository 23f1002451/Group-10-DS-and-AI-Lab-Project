"""Evaluation module for the mDeBERTa-v3-base guardrail classifier.

Provides the metrics functions used identically in both the notebook training
loop and the offline evaluation script, ensuring numerical consistency between
reported and reproduced results.

Key metrics:
    macro_f1         : Macro-averaged F1 across three classes.
    ASR              : Attack Success Rate — fraction of attack prompts
                       allowed through (predicted as benign).
    FRR              : False Refusal Rate — fraction of benign prompts
                       incorrectly blocked.
    composite_score  : W_F1 * F1 + W_ASR * (1 - ASR) + W_FRR * (1 - FRR)
                       with weights W_F1=0.3, W_ASR=0.5, W_FRR=0.2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrail_classifier import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    GuardrailModel,
    PromptDataset,
    build_tokenizer,
    choose_device,
    load_json_records,
    make_collate,
    validate_records,
)


# ---------------------------------------------------------------------------
# Core metrics functions (mirrors Final Guardrail.ipynb Cell 14 exactly)
# ---------------------------------------------------------------------------

def compute_metrics(labels: List[int], preds: List[int]) -> dict:
    """Compute all guardrail-relevant metrics from integer label/pred lists."""
    la = np.array(labels)
    pa = np.array(preds)

    if len(la) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_f1": {ID_TO_LABEL[i]: 0.0 for i in range(3)},
            "asr": {"overall": 0.0, "jailbreak": 0.0, "harmful": 0.0},
            "frr": 0.0,
            "cm": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }

    _, _, f1, _ = precision_recall_fscore_support(
        la, pa, labels=[0, 1, 2], zero_division=0
    )

    atk = la != 0
    bn = la == 0
    jb = la == 1
    hm = la == 2

    asr_overall = float((pa[atk] == 0).sum() / atk.sum()) if atk.sum() else 0.0
    asr_jailbreak = float((pa[jb] == 0).sum() / jb.sum()) if jb.sum() else 0.0
    asr_harmful = float((pa[hm] == 0).sum() / hm.sum()) if hm.sum() else 0.0
    frr = float((pa[bn] != 0).sum() / bn.sum()) if bn.sum() else 0.0

    return {
        "accuracy": round(float(accuracy_score(la, pa)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "per_class_f1": {ID_TO_LABEL[i]: round(float(f1[i]), 4) for i in range(3)},
        "asr": {
            "overall": round(asr_overall, 4),
            "jailbreak": round(asr_jailbreak, 4),
            "harmful": round(asr_harmful, 4),
        },
        "frr": round(frr, 4),
        "cm": confusion_matrix(la, pa, labels=[0, 1, 2]).tolist(),
    }


def composite_score(f1: float, asr: float, frr: float,
                    w_f1: float = 0.3, w_asr: float = 0.5, w_frr: float = 0.2) -> float:
    """Compute the composite optimisation objective.

    composite = W_F1 * F1 + W_ASR * (1 - ASR) + W_FRR * (1 - FRR)

    Higher is better.  ASR and FRR are costs so they are inverted.
    Default weights match FINAL_CONFIG.
    """
    return round(w_f1 * f1 + w_asr * (1 - asr) + w_frr * (1 - frr), 6)


def batch_evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run inference on all batches and return a full metrics dict.

    Includes CUDA-synchronised latency measurement and NaN-guard on logits,
    mirroring the evaluation logic used in Final Guardrail.ipynb Cell 14.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    latencies: List[float] = []
    losses: List[float] = []
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(ids, mask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if not torch.isfinite(logits).all():
                logits = torch.zeros_like(logits)

            lat_ms = (time.perf_counter() - t0) * 1000 / ids.shape[0]
            latencies.extend([lat_ms] * ids.shape[0])

            probs = torch.softmax(logits, -1)
            loss = ce(logits, labels)
            if torch.isfinite(loss):
                losses.append(loss.item())

            all_preds.extend(torch.argmax(logits, -1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    m = compute_metrics(all_labels, all_preds)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    m.update({
        "loss": round(float(np.mean(losses)), 4) if losses else 0.0,
        "latency_ms": {
            "mean": round(mean_lat, 3),
            "p95": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        },
        "throughput_rps": round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    })

    return m


# ---------------------------------------------------------------------------
# Full checkpoint evaluation (CLI entry point)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    output_metrics_path: Path,
    output_samples_path: Path,
    batch_size: int = 8,
    sample_count: int = 20,
) -> dict:
    """Load a checkpoint, run inference, compute all guardrail metrics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_name = checkpoint["model_name"]
    max_length = int(checkpoint.get("max_length", 512))

    records = load_json_records(dataset_path)
    validate_records(records)

    tokenizer = build_tokenizer(model_name)
    loader = DataLoader(
        PromptDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_length),
    )

    device = choose_device()
    model = GuardrailModel(model_name=model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    m = batch_evaluate(model, loader, device)

    # Build sample rows for qualitative inspection
    sample_rows = []
    for idx, prob in enumerate(m["all_probs"][:sample_count]):
        pred_idx = int(np.argmax(prob))
        true_idx = m["all_labels"][idx]
        sample_rows.append({
            "prompt_text": records[idx]["prompt_text"],
            "true_label": ID_TO_LABEL[true_idx],
            "pred_label": ID_TO_LABEL[pred_idx],
            "correct": pred_idx == true_idx,
            "probabilities": {
                ID_TO_LABEL[j]: round(float(prob[j]), 6) for j in range(3)
            },
        })

    # Per-attack-type breakdown
    per_attack_type: Dict[str, dict] = {}
    benign_idx = LABEL_TO_ID["benign"]
    for i, rec in enumerate(records):
        at = rec.get("attack_type", "none")
        if at not in per_attack_type:
            per_attack_type[at] = {"total": 0, "correct": 0, "missed_as_benign": 0}
        per_attack_type[at]["total"] += 1
        if m["all_preds"][i] == m["all_labels"][i]:
            per_attack_type[at]["correct"] += 1
        if m["all_labels"][i] != benign_idx and m["all_preds"][i] == benign_idx:
            per_attack_type[at]["missed_as_benign"] += 1
    for at, stats in per_attack_type.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0
        if at != "none":
            stats["asr"] = round(stats["missed_as_benign"] / stats["total"], 4) if stats["total"] > 0 else 0

    metrics = {
        "dataset": str(dataset_path),
        "num_examples": len(records),
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "guardrail_metrics": {
            "overall_asr": m["asr"]["overall"],
            "jailbreak_asr": m["asr"]["jailbreak"],
            "harmful_asr": m["asr"]["harmful"],
            "false_refusal_rate": m["frr"],
        },
        "per_class_f1": m["per_class_f1"],
        "confusion_matrix": m["cm"],
        "confusion_matrix_labels": [ID_TO_LABEL[i] for i in range(3)],
        "latency": m["latency_ms"],
        "throughput_rps": m["throughput_rps"],
        "per_attack_type": per_attack_type,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_samples_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output_samples_path.write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mDeBERTa guardrail classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_metrics_path=args.output_metrics,
        output_samples_path=args.output_samples,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
    )
    print("Evaluation completed.")
    print(f"Examples  : {metrics['num_examples']}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Macro-F1  : {metrics['macro_f1']:.4f}")
    print(f"ASR       : {metrics['guardrail_metrics']['overall_asr']:.4f}")
    print(f"FRR       : {metrics['guardrail_metrics']['false_refusal_rate']:.4f}")
    print(
        f"Latency   : {metrics['latency']['mean']:.1f}ms avg, "
        f"{metrics['latency']['p95']:.1f}ms p95"
    )


if __name__ == "__main__":
    main()

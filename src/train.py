"""Training module for the mDeBERTa-v3-base guardrail classifier.

Implements the fixed-configuration training loop derived from the
99-trial hyperparameter search documented in mdeberta HPT.ipynb.

Final hyperparameters (FINAL_CONFIG):
    MAX_LENGTH              : 512
    BATCH_SIZE              : 4
    LEARNING_RATE           : 3e-5
    WEIGHT_DECAY            : 0.01
    WARMUP_RATIO            : 0.05
    DROPOUT                 : 0.2
    GRADIENT_CLIP           : 0.5
    EPOCHS                  : 10
    EARLY_STOPPING_PATIENCE : 3
    LABEL_SMOOTHING         : 0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

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
    set_seed,
    validate_records,
)
from src.evaluate import batch_evaluate, compute_metrics, composite_score


FINAL_CONFIG = {
    "MAX_LENGTH": 512,
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 3e-5,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.05,
    "DROPOUT": 0.2,
    "GRADIENT_CLIP": 0.5,
    "EPOCHS": 10,
    "EARLY_STOPPING_PATIENCE": 3,
    "W_F1": 0.3,
    "W_ASR": 0.5,
    "W_FRR": 0.2,
    "T_BLOCK": 0.15,
    "T_TRANSFORM": 0.07,
    "LABEL_SMOOTHING": 0.0,
}


@dataclass
class TrainConfig:
    train_data: Path
    val_data: Path
    output_dir: Path
    model_name: str = "microsoft/mdeberta-v3-base"
    max_length: int = 512
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    dropout: float = 0.2
    gradient_clip_norm: float = 0.5
    label_smoothing: float = 0.0
    early_stopping_patience: int = 3
    w_f1: float = 0.3
    w_asr: float = 0.5
    w_frr: float = 0.2
    seed: int = 42


def class_weights_for(records: Sequence[dict], device: torch.device) -> torch.Tensor:
    """Compute clipped inverse-frequency class weights.

    Formula: weight_c = N_total / (K * N_c), clipped to [0.5, 5.0].
    Clipping prevents extreme weights from destabilising gradient flow
    when one class is severely under-represented.
    """
    ids = np.array([LABEL_TO_ID[r["label"]] for r in records])
    cc = np.bincount(ids, minlength=3).astype(np.float32)
    w = np.where(cc > 0, cc.sum() / (3 * cc), 0.0)
    w = np.clip(w, 0.5, 5.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def run_training(config: TrainConfig) -> dict:
    """Execute the full training loop with early stopping.

    Saves the best checkpoint (by composite score) to config.output_dir.
    Returns a summary dict with training history and best metrics.
    """
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_json_records(config.train_data)
    val_records = load_json_records(config.val_data)
    validate_records(train_records)
    validate_records(val_records)

    tokenizer = build_tokenizer(config.model_name)
    col_fn = make_collate(tokenizer, config.max_length)

    train_loader = DataLoader(
        PromptDataset(train_records),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=col_fn,
    )
    val_loader = DataLoader(
        PromptDataset(val_records),
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=col_fn,
    )

    device = choose_device()
    model = GuardrailModel(
        model_name=config.model_name,
        dropout=config.dropout,
    ).to(device)

    class_weights = class_weights_for(train_records, device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_ckpt = config.output_dir / "final_model.pt"
    best_score = -1.0
    patience_counter = 0
    history: List[dict] = []

    print(f"\n{'='*65}", flush=True)
    print("FINAL TRAINING — Fixed Configuration", flush=True)
    print(f"{'='*65}", flush=True)
    print(
        f"  LR={config.learning_rate}  BS={config.batch_size}  ML={config.max_length}",
        flush=True,
    )
    print(
        f"  WD={config.weight_decay}   DO={config.dropout}     GC={config.gradient_clip_norm}",
        flush=True,
    )
    print(
        f"  Epochs={config.epochs}   Patience={config.early_stopping_patience}",
        flush=True,
    )
    print(
        f"  Train: {len(train_records)} | Val: {len(val_records)} | Device: {device}",
        flush=True,
    )

    for epoch in range(config.epochs):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_m = batch_evaluate(model, val_loader, device)

        score = composite_score(
            val_m["macro_f1"],
            val_m["asr"]["overall"],
            val_m["frr"],
            config.w_f1,
            config.w_asr,
            config.w_frr,
        )

        print(
            f"  Epoch {epoch + 1}/{config.epochs} — "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_f1: {val_m['macro_f1']:.4f} | "
            f"ASR: {val_m['asr']['overall']:.4f} | "
            f"FRR: {val_m['frr']:.4f} | "
            f"score: {score:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val": val_m,
            "composite_score": score,
        })

        if score > best_score:
            best_score = score
            patience_counter = 0
            print(f"  New best score: {best_score:.4f} — saving checkpoint", flush=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config.model_name,
                    "max_length": config.max_length,
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL,
                    "train_config": _json_ready(asdict(config)),
                    "final_config": FINAL_CONFIG,
                },
                best_ckpt,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(
                    f"  Early stopping at epoch {epoch + 1} "
                    f"(patience={config.early_stopping_patience})",
                    flush=True,
                )
                break

    outputs = {
        "train_config": _json_ready(asdict(config)),
        "history": history,
        "best_composite_score": best_score,
        "best_model_path": str(best_ckpt),
        "early_stopped": patience_counter >= config.early_stopping_patience,
        "epochs_completed": len(history),
    }
    (config.output_dir / "training_metrics.json").write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mDeBERTa guardrail classifier")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max-length", type=int, default=FINAL_CONFIG["MAX_LENGTH"])
    parser.add_argument("--epochs", type=int, default=FINAL_CONFIG["EPOCHS"])
    parser.add_argument("--batch-size", type=int, default=FINAL_CONFIG["BATCH_SIZE"])
    parser.add_argument("--learning-rate", type=float, default=FINAL_CONFIG["LEARNING_RATE"])
    parser.add_argument("--weight-decay", type=float, default=FINAL_CONFIG["WEIGHT_DECAY"])
    parser.add_argument("--warmup-ratio", type=float, default=FINAL_CONFIG["WARMUP_RATIO"])
    parser.add_argument("--dropout", type=float, default=FINAL_CONFIG["DROPOUT"])
    parser.add_argument("--gradient-clip", type=float, default=FINAL_CONFIG["GRADIENT_CLIP"])
    parser.add_argument("--label-smoothing", type=float, default=FINAL_CONFIG["LABEL_SMOOTHING"])
    parser.add_argument("--early-stopping-patience", type=int, default=FINAL_CONFIG["EARLY_STOPPING_PATIENCE"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )
    result = run_training(cfg)
    print(f"\nTraining completed in {result['epochs_completed']} epoch(s).")
    print(f"Best composite score: {result['best_composite_score']:.4f}")
    print(f"Checkpoint: {result['best_model_path']}")
    if result["early_stopped"]:
        print("(Early stopping triggered)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LABEL_TO_ID: Dict[str, int] = {
    "benign": 0,
    "jailbreak": 1,
    "harmful": 2,
}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}


class GuardrailModel(nn.Module):
    """mDeBERTa encoder with mean-pooling and a linear classification head.

    Uses masked mean-pooling over the full token sequence rather than the
    CLS token alone, which provides a more stable representation for
    variable-length adversarial prompts.
    """

    def __init__(self, model_name: str, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state.float()
        mask = attention_mask.unsqueeze(-1).expand(last.size()).float()
        pooled = torch.sum(last * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.head(self.drop(pooled))


# Alias kept for backward compatibility with scripts that import GuardrailClassifier.
GuardrailClassifier = GuardrailModel


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        return {"text": r["prompt_text"], "label": LABEL_TO_ID[r["label"]]}


def make_collate(tokenizer, max_length: int):
    """Return a collate function implementing head-tail truncation.

    For prompts longer than max_length tokens the function retains the first
    half and the last half of the token sequence.  This preserves both the
    adversarial setup (typically in the head) and the injected payload
    (typically in the tail), outperforming naive left-truncation on long
    jailbreak prompts.
    """
    def collate(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        input_ids_list = []
        attention_mask_list = []

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_length - 2:
                head_len = (max_length - 2) // 2
                tail_len = (max_length - 2) - head_len
                tokens = tokens[:head_len] + tokens[-tail_len:]

            ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
            input_ids_list.append(ids)
            attention_mask_list.append([1] * len(ids))

        max_batch_len = max(len(ids) for ids in input_ids_list)

        for idx in range(len(input_ids_list)):
            pad_len = max_batch_len - len(input_ids_list[idx])
            input_ids_list[idx].extend([tokenizer.pad_token_id] * pad_len)
            attention_mask_list[idx].extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": labels,
        }

    return collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_records(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list records in {path}")
    return [r for r in payload if isinstance(r, dict) and "prompt_text" in r and "label" in r]


def validate_records(records) -> None:
    for idx, row in enumerate(records):
        for key in ("prompt_text", "label"):
            if key not in row:
                raise ValueError(f"Record {idx} missing field: {key}")
        if row["label"] not in LABEL_TO_ID:
            raise ValueError(f"Record {idx} has unknown label: {row['label']}")


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Shared paths, reproducibility helpers, and tensor preparation utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "Qwen" / "Qwen2___5-1___5B-Instruct")))
SEED = int(os.getenv("SEED", "42"))


def set_seed(seed: int = SEED) -> None:
    """Seed Python and PyTorch when available."""
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def split_records(records: Sequence[Any], val_ratio: float = 0.2, seed: int = SEED):
    """Return deterministic train/validation splits without overlapping records."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("at least two records are required")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, round(len(records) * val_ratio))
    val_indices = set(indices[:val_size])
    train = [item for index, item in enumerate(records) if index not in val_indices]
    validation = [item for index, item in enumerate(records) if index in val_indices]
    return train, validation


def mask_padding_labels(input_ids, attention_mask, ignore_index: int = -100):
    """Mask padded positions so supervised loss only sees real tokens."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = ignore_index
    return labels


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

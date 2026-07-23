"""LoRA SFT for the checked-in Chinese short-text domain dataset."""

from __future__ import annotations

import gc
import math
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ARTIFACTS_DIR, DATA_DIR, MODEL_PATH, load_json, mask_padding_labels, save_json, set_seed, split_records


RUN_NAME = os.getenv("RUN_NAME", "domain_sft")
OUTPUT_DIR = ARTIFACTS_DIR / "checkpoints" / RUN_NAME
INITIAL_ADAPTER_PATH = ARTIFACTS_DIR / "checkpoints" / "general_sft"
CUTOFF_LEN = 256
GRAD_ACCUM = 4
LEARNING_RATE = float(os.getenv("DOMAIN_LR", "1e-4"))
NUM_EPOCHS = int(os.getenv("DOMAIN_EPOCHS", "5"))


def format_domain_example(item):
    return (
        f"<|im_start|>user\n{item['instruction']}\n{item.get('input', '')}<|im_end|>\n"
        f"<|im_start|>assistant\n{item['output']}<|im_end|>"
    )


def encode_example(tokenizer, text, device):
    encoded = tokenizer(
        text, truncation=True, max_length=CUTOFF_LEN,
        padding="max_length", return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = mask_padding_labels(input_ids, attention_mask)
    return input_ids, attention_mask, labels


@torch.no_grad()
def evaluate_loss(model, tokenizer, records, device):
    model.eval()
    losses = []
    for item in records:
        input_ids, attention_mask, labels = encode_example(tokenizer, format_domain_example(item), device)
        losses.append(model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    raw = load_json(DATA_DIR / "domain_zh.json")
    train_records, validation_records = split_records(raw, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device.type == "cuda":
        model_kwargs["device_map"] = "cuda"
    if not INITIAL_ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"General SFT checkpoint is required before domain SFT: {INITIAL_ADAPTER_PATH}"
        )
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    if device.type != "cuda":
        base_model.to(device)
    model = PeftModel.from_pretrained(base_model, INITIAL_ADAPTER_PATH, is_trainable=True)
    for name, parameter in model.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad = True
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    metrics = {
        "train": [], "validation": [], "seed": 42,
        "train_size": len(train_records), "validation_size": len(validation_records),
        "initial_adapter": str(INITIAL_ADAPTER_PATH),
    }
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        accumulated = 0
        for index, item in enumerate(train_records):
            input_ids, attention_mask, labels = encode_example(tokenizer, format_domain_example(item), device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            (loss / GRAD_ACCUM).backward()
            running_loss += loss.item()
            accumulated += 1
            is_last = index == len(train_records) - 1
            if accumulated == GRAD_ACCUM or is_last:
                if accumulated < GRAD_ACCUM:
                    scale = GRAD_ACCUM / accumulated
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            del input_ids, attention_mask, labels

        train_loss = running_loss / len(train_records)
        validation_loss = evaluate_loss(model, tokenizer, validation_records, device)
        metrics["train"].append(train_loss)
        metrics["validation"].append(validation_loss)
        print(f"epoch={epoch + 1}/{NUM_EPOCHS} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    metrics["elapsed_seconds"] = round(time.time() - t0, 2)
    save_json(metrics, ARTIFACTS_DIR / "metrics" / f"{RUN_NAME}.json")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"saved_checkpoint={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

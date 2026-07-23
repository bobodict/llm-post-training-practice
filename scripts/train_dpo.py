"""A small, auditable DPO implementation for the Chinese domain data."""

from __future__ import annotations

import gc
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ARTIFACTS_DIR, DATA_DIR, MODEL_PATH, load_json, save_json, set_seed, split_records
from .dpo_core import dpo_loss, preference_accuracy


SFT_PATH = ARTIFACTS_DIR / "checkpoints" / "general_sft"
OUTPUT_DIR = ARTIFACTS_DIR / "checkpoints" / "domain_dpo"
BETA = 0.1
LEARNING_RATE = 5e-6
NUM_EPOCHS = 3


def make_prompt(instruction):
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def load_base_model(device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if device.type != "cuda":
        model.to(device)
    return model


def compute_response_log_prob(model, tokenizer, prompt, response, device):
    """Return the summed log probability of response tokens only."""
    full = tokenizer(prompt + response, add_special_tokens=False, return_tensors="pt")
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = full["input_ids"].to(device)
    prompt_len = prompt_tokens.shape[1]
    if input_ids.shape[1] <= prompt_len:
        raise ValueError("response must contain at least one token")

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    labels = input_ids[:, prompt_len:]
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs.sum()


def compute_reference_scores(model, tokenizer, records, device):
    scores = []
    with torch.no_grad():
        for item in records:
            prompt = make_prompt(item["instruction"])
            chosen = compute_response_log_prob(model, tokenizer, prompt, item["chosen"] + "<|im_end|>", device)
            rejected = compute_response_log_prob(model, tokenizer, prompt, item["rejected"] + "<|im_end|>", device)
            scores.append({"chosen": chosen.item(), "rejected": rejected.item()})
    return scores


def train_on_records(policy, tokenizer, records, reference_scores, optimizer, device):
    policy.train()
    losses = []
    chosen_scores = []
    rejected_scores = []
    for item, reference in zip(records, reference_scores):
        prompt = make_prompt(item["instruction"])
        chosen = compute_response_log_prob(policy, tokenizer, prompt, item["chosen"] + "<|im_end|>", device)
        rejected = compute_response_log_prob(policy, tokenizer, prompt, item["rejected"] + "<|im_end|>", device)
        ref_chosen = torch.tensor(reference["chosen"], device=device)
        ref_rejected = torch.tensor(reference["rejected"], device=device)
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=BETA)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        chosen_scores.append(chosen.detach())
        rejected_scores.append(rejected.detach())

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "preference_accuracy": preference_accuracy(
            torch.stack(chosen_scores),
            torch.stack(rejected_scores),
            torch.tensor([item["chosen"] for item in reference_scores], device=device),
            torch.tensor([item["rejected"] for item in reference_scores], device=device),
        ),
    }


def evaluate_records(policy, tokenizer, records, reference_scores, device):
    policy.eval()
    chosen_scores = []
    rejected_scores = []
    with torch.no_grad():
        for item in records:
            prompt = make_prompt(item["instruction"])
            chosen_scores.append(compute_response_log_prob(policy, tokenizer, prompt, item["chosen"] + "<|im_end|>", device))
            rejected_scores.append(compute_response_log_prob(policy, tokenizer, prompt, item["rejected"] + "<|im_end|>", device))
    chosen = torch.stack(chosen_scores)
    rejected = torch.stack(rejected_scores)
    return {
        "preference_accuracy": preference_accuracy(
            chosen,
            rejected,
            torch.tensor([item["chosen"] for item in reference_scores], device=device),
            torch.tensor([item["rejected"] for item in reference_scores], device=device),
        )
    }


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    records = load_json(DATA_DIR / "dpo_domain_demo.json")
    train_records, validation_records = split_records(records, seed=42)

    print("[1/4] Loading SFT reference model...")
    reference_base = load_base_model(device)
    reference_model = PeftModel.from_pretrained(reference_base, SFT_PATH)
    reference_model.eval()
    train_reference = compute_reference_scores(reference_model, tokenizer, train_records, device)
    validation_reference = compute_reference_scores(reference_model, tokenizer, validation_records, device)
    del reference_model, reference_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[2/4] Loading trainable SFT policy...")
    policy_base = load_base_model(device)
    policy = PeftModel.from_pretrained(policy_base, SFT_PATH, is_trainable=True)
    for name, parameter in policy.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )

    metrics = {"train": [], "validation": [], "beta": BETA, "epochs": NUM_EPOCHS}
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_on_records(policy, tokenizer, train_records, train_reference, optimizer, device)
        validation_metrics = evaluate_records(policy, tokenizer, validation_records, validation_reference, device)
        metrics["train"].append(train_metrics)
        metrics["validation"].append(validation_metrics)
        print(
            f"epoch={epoch + 1}/{NUM_EPOCHS} loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['preference_accuracy']:.2%} "
            f"validation_acc={validation_metrics['preference_accuracy']:.2%}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    metrics["elapsed_seconds"] = round(time.time() - t0, 2)
    save_json(metrics, ARTIFACTS_DIR / "metrics" / "domain_dpo.json")
    print(f"[4/4] saved_checkpoint={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

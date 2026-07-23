"""Minimal language-model GRPO smoke experiment for a single GPU."""

from __future__ import annotations

import gc
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ARTIFACTS_DIR, MODEL_PATH, save_json, set_seed
from .grpo_core import clipped_grpo_loss, compute_group_advantages


POLICY_PATH = ARTIFACTS_DIR / "checkpoints" / "domain_dpo"
OUTPUT_DIR = ARTIFACTS_DIR / "checkpoints" / "grpo_smoke"
GROUP_SIZE = 4
MAX_NEW_TOKENS = 8
NUM_STEPS = 4
LEARNING_RATE = 1e-6

CASES = [
    {"prompt": "计算 17 * 23，只输出最终数字。", "expected": "391"},
    {"prompt": "计算 19 * 17，只输出最终数字。", "expected": "323"},
    {"prompt": "计算 37 * 26，只输出最终数字。", "expected": "962"},
    {"prompt": "计算 48 * 27，只输出最终数字。", "expected": "1296"},
]


def reward_response(response, expected):
    """Reward only the final numeric answer, avoiding substring matches."""
    numbers = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", response.strip())
    return 1.0 if numbers and numbers[-1] == expected else 0.0


def make_prompt(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def load_policy(device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "cuda"
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if device.type != "cuda":
        base.to(device)
    policy = PeftModel.from_pretrained(base, POLICY_PATH, is_trainable=True)
    for name, parameter in policy.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad = True
    return policy


def sample_responses(model, tokenizer, prompt, device):
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=GROUP_SIZE,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in output]


def sequence_log_prob(model, tokenizer, prompt, response, device):
    full = tokenizer(prompt + response, add_special_tokens=False, return_tensors="pt")
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = full["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    labels = input_ids[:, prompt_len:]
    return torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1).sum()


def main():
    set_seed()
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Domain DPO checkpoint is required before GRPO: {POLICY_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    metrics = {"steps": [], "group_size": GROUP_SIZE, "algorithm": "group-relative clipped policy objective"}
    t0 = time.time()

    for step, case in enumerate(CASES[:NUM_STEPS], start=1):
        prompt = make_prompt(case["prompt"])
        model.eval()
        responses = sample_responses(model, tokenizer, prompt, device)
        rewards = torch.tensor(
            [reward_response(response, case["expected"]) for response in responses],
            dtype=torch.float32,
            device=device,
        )
        response_mask = torch.ones((GROUP_SIZE, 1), device=device)
        advantages = compute_group_advantages(
            rewards.unsqueeze(-1), response_mask, [0] * GROUP_SIZE
        ).squeeze(-1)
        with torch.no_grad():
            old_log_probs = torch.stack([
                sequence_log_prob(model, tokenizer, prompt, response, device)
                for response in responses
            ])

        model.train()
        log_probs = torch.stack([
            sequence_log_prob(model, tokenizer, prompt, response, device)
            for response in responses
        ])
        loss = clipped_grpo_loss(log_probs, old_log_probs, advantages)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics["steps"].append({
            "step": step,
            "prompt": case["prompt"],
            "responses": responses,
            "rewards": rewards.detach().cpu().tolist(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "advantages": advantages.detach().cpu().tolist(),
            "nonzero_advantages": int((advantages != 0).sum().item()),
            "loss": loss.item(),
        })
        print(
            f"step={step} reward_mean={rewards.mean().item():.2f} "
            f"nonzero_advantages={(advantages != 0).sum().item()} loss={loss.item():.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    metrics["elapsed_seconds"] = round(time.time() - t0, 2)
    save_json(metrics, ARTIFACTS_DIR / "metrics" / "grpo_smoke.json")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"saved_checkpoint={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

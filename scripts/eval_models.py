"""Deterministic comparison of base and post-trained models."""

from __future__ import annotations

import gc
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ARTIFACTS_DIR, DATA_DIR, MODEL_PATH, load_json, save_json, set_seed


MODELS = {
    "base": None,
    "general_sft": ARTIFACTS_DIR / "checkpoints" / "general_sft",
    "domain_sft": ARTIFACTS_DIR / "checkpoints" / "domain_sft",
    "domain_dpo": ARTIFACTS_DIR / "checkpoints" / "domain_dpo",
}

def load_model(adapter_path, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "cuda"
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if device.type != "cuda":
        base.to(device)
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
    model.eval()
    return model


def make_prompt(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def generate_answer(model, tokenizer, prompt, device, max_new_tokens=24):
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_label(answer, labels):
    """Extract only a label declared by the current closed-set task."""
    for label in sorted(labels, key=len, reverse=True):
        if label in answer[:80]:
            return label
    return "<unparsed>"


def evaluate_model(model, tokenizer, device, cases):
    counts = {}
    confusion = {}
    for case in cases:
        options = "、".join(case["labels"])
        prompt = f"{case['instruction']}\n{case['input']}\n可选标签：{options}\n只输出一个标签。"
        answer = generate_answer(model, tokenizer, make_prompt(prompt), device)
        predicted = extract_label(answer, case["labels"])
        counts.setdefault(case["task"], {"correct": 0, "total": 0})
        counts[case["task"]]["total"] += 1
        counts[case["task"]]["correct"] += int(predicted == case["label"])
        confusion.setdefault(case["task"], {}).setdefault(case["label"], {}).setdefault(predicted, 0)
        confusion[case["task"]][case["label"]][predicted] += 1
    for value in counts.values():
        value["accuracy"] = value["correct"] / value["total"]
    return {"tasks": counts, "confusion": confusion}


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_json(DATA_DIR / "eval_domain.json")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    results = {"seed": 42, "decoding": "greedy", "cases": len(cases), "models": {}}

    for name, adapter_path in MODELS.items():
        if adapter_path is not None and not adapter_path.exists():
            print(f"skip={name} reason=missing_checkpoint path={adapter_path}")
            continue
        print(f"evaluating={name}")
        model = load_model(adapter_path, device)
        results["models"][name] = evaluate_model(model, tokenizer, device, cases)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = ARTIFACTS_DIR / "evaluation" / "comparison.json"
    save_json(results, output_path)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()

"""
M2: DPO (Direct Preference Optimization) training
Manually implements the DPO algorithm to work around HF Trainer segfault on Windows.
DPO loss: -log(sigma(beta * (log_pi_chosen/log_ref_chosen - log_pi_rejected/log_ref_rejected)))
Reference: https://arxiv.org/abs/2305.18290
"""
import os, json, time, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_PATH = "./models/Qwen/Qwen2___5-1___5B-Instruct"
SFT_PATH = "saves/qwen25-1.5b/lora/sft_manual"   # policy model checkpoint (after SFT)
DPO_DATA = "data/dpo_bilibili_demo.json"
OUTPUT_DIR = "saves/qwen25-1.5b/lora/dpo"
BETA = 0.1       # DPO temperature
LR = 5e-6
NUM_EPOCHS = 10

def compute_log_probs(model, tokenizer, prompt, response):
    """Compute average log probability per token of response given prompt.
    Returns a scalar tensor with gradients when model is in train mode."""
    full_text = prompt + response
    enc = tokenizer(full_text, return_tensors="pt")
    input_ids = enc["input_ids"].cuda()

    prompt_enc = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    shift_logits = logits[:, prompt_len-1:-1, :]
    shift_labels = input_ids[:, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.mean()

print("=" * 60)
print("M2: DPO Training - Direct Preference Optimization")
print("=" * 60)

# ---- Load data ----
with open(DPO_DATA, "r", encoding="utf-8") as f:
    dpo_pairs = json.load(f)
print(f"Loaded {len(dpo_pairs)} DPO preference pairs")

# ---- Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_dpo(instruction, response):
    return (f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>")

# ---- Phase 1: Compute reference model log probs ----
print("\n[Phase 1] Computing reference model probabilities...")
# Use base model as reference
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
)
ref_model.eval()

ref_data = []
for i, pair in enumerate(dpo_pairs):
    prompt = f"<|im_start|>user\n{pair['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    with torch.no_grad():
        chosen_logp = compute_log_probs(ref_model, tokenizer, prompt, pair["chosen"] + "<|im_end|>")
        rejected_logp = compute_log_probs(ref_model, tokenizer, prompt, pair["rejected"] + "<|im_end|>")
    ref_data.append({"chosen_logp": chosen_logp.item(), "rejected_logp": rejected_logp.item()})
    print(f"  Pair {i+1}: ref_chosen={chosen_logp.item():.3f}, ref_rejected={rejected_logp.item():.3f}")

del ref_model
gc.collect()
torch.cuda.empty_cache()

# ---- Phase 2: DPO Training ----
print("\n[Phase 2] DPO training...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
)
policy = PeftModel.from_pretrained(base_model, SFT_PATH)
# Fix: PeftModel.from_pretrained doesn't set requires_grad on LoRA params
for n, p in policy.named_parameters():
    if 'lora' in n:
        p.requires_grad = True
policy.train()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, policy.parameters()), lr=LR)

t0 = time.time()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for i, (pair, ref) in enumerate(zip(dpo_pairs, ref_data)):
        prompt = f"<|im_start|>user\n{pair['instruction']}<|im_end|>\n<|im_start|>assistant\n"

        # Compute policy log probs WITH gradients (model in train() mode)
        pi_chosen = compute_log_probs(policy, tokenizer, prompt, pair["chosen"] + "<|im_end|>")
        pi_rejected = compute_log_probs(policy, tokenizer, prompt, pair["rejected"] + "<|im_end|>")

        # DPO loss
        log_ratio_chosen = pi_chosen - ref["chosen_logp"]
        log_ratio_rejected = pi_rejected - ref["rejected_logp"]
        diff = log_ratio_chosen - log_ratio_rejected
        loss = -F.logsigmoid(BETA * diff)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dpo_pairs)
    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | avg_loss={avg_loss:.4f} | "
          f"chosen={pi_chosen.item():.3f} rejected={pi_rejected.item():.3f}")

print(f"\n[Phase 3] Saving DPO model to {OUTPUT_DIR}...")
policy.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DONE! {time.time()-t0:.0f}s")

# ---- Test ----
print("\n--- DPO vs SFT comparison ---")
from peft import PeftModel
test_prompts = [
    "解释什么是机器学习",
    "如何评价哔哩哔哩社区的文化氛围",
]

for model_name, path in [("SFT (before DPO)", SFT_PATH), ("DPO (after DPO)", OUTPUT_DIR)]:
    print(f"\n{model_name}:")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda")
    m = PeftModel.from_pretrained(base, path)
    m.eval()
    for q in test_prompts:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=100, temperature=0.7, do_sample=True)
        ans = tokenizer.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1]
        print(f"  Q: {q[:30]}...")
        print(f"  A: {ans[:100]}...")
    del m, base
    gc.collect()
    torch.cuda.empty_cache()

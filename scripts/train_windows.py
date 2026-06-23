"""
Windows-compatible LoRA SFT training - Qwen2.5-1.5B + LoRA on RTX 4060 8GB.
One sample per step, no batching, gradient accumulation = 4.
"""
import os, json, time, gc
os.environ["USE_MODELSCOPE_HUB"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_PATH = "./models/Qwen/Qwen2___5-1___5B-Instruct"
OUTPUT_DIR = "saves/qwen25-1.5b/lora/sft_manual"
CUTOFF_LEN = 512
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
LORA_R = 8
MAX_SAMPLES = 500

def format_example(item):
    """Format alpaca item into Qwen chat template."""
    instr = item.get("instruction", "")
    inp = item.get("input", "")
    out = item.get("output", "")
    return (f"<|im_start|>user\n{instr}\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>")

# ---- Load model ----
print("[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_R * 2, target_modules="all-linear", lora_dropout=0.05
))
model.print_trainable_parameters()
model.train()
print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# ---- Load data ----
print("[2/4] Loading data...")
with open("data/alpaca_zh_demo.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
raw = raw[:MAX_SAMPLES]
texts = [format_example(item) for item in raw]

# ---- Setup optimizer ----
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = (MAX_SAMPLES // GRAD_ACCUM) * NUM_EPOCHS
print(f"[3/4] Training: {MAX_SAMPLES} samples, {total_steps} steps, cutoff={CUTOFF_LEN}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
t0 = time.time()
global_step = 0
running_loss = 0.0

# ---- Training loop: one sample per step ----
for epoch in range(NUM_EPOCHS):
    for idx, text in enumerate(texts):
        # Tokenize single sample
        enc = tokenizer(text, truncation=True, max_length=CUTOFF_LEN,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].cuda()
        labels = enc["input_ids"].clone().cuda()

        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item()

        # Gradient accumulation step
        if (idx + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Step {global_step}/{total_steps} | "
                      f"loss={running_loss:.4f} | "
                      f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB | "
                      f"{elapsed:.0f}s")
                running_loss = 0.0

        # Cleanup
        del input_ids, labels, outputs, enc
        if (idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

# ---- Save ----
print("[4/4] Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

elapsed = time.time() - t0
print(f"\nDONE! {global_step} steps in {elapsed:.0f}s ({elapsed/global_step:.1f}s/step)")
print(f"Model saved to: {OUTPUT_DIR}")

# Quick test
print("\n--- Inference test ---")
model.eval()
test_input = "<|im_start|>user\n介绍一下人工智能\n<|im_end|>\n<|im_start|>assistant\n"
enc = tokenizer(test_input, return_tensors="pt")
enc = {k: v.cuda() for k, v in enc.items()}
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(out[0], skip_special_tokens=False)
    print(response)

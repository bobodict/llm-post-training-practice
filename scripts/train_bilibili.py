"""
M1: B站弹幕/评论数据 SFT - 自定义数据集微调 Qwen2.5-1.5B
"""
import os, json, time, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_PATH = "./models/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH = "data/bilibili_zh.json"
OUTPUT_DIR = "saves/qwen25-1.5b/lora/bilibili_sft"
CUTOFF_LEN = 256
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LORA_R = 16

def format_bilibili(item):
    return (f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['output']}<|im_end|>")

print("[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_R*2, target_modules="all-linear", lora_dropout=0.05
))
model.print_trainable_parameters()
model.train()

print("[2/4] Loading B站 dataset...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)
texts = [format_bilibili(item) for item in raw]
print(f"  {len(texts)} B站-style samples loaded")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = (len(texts) // GRAD_ACCUM) * NUM_EPOCHS
print(f"[3/4] Training: {len(texts)} samples, {total_steps} steps")

os.makedirs(OUTPUT_DIR, exist_ok=True)
t0 = time.time()
global_step = 0

for epoch in range(NUM_EPOCHS):
    for idx, text in enumerate(texts):
        enc = tokenizer(text, truncation=True, max_length=CUTOFF_LEN,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].cuda()
        labels = enc["input_ids"].clone().cuda()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (idx + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 5 == 0:
                print(f"  Step {global_step}/{total_steps} | loss={loss.item()*GRAD_ACCUM:.4f} | {time.time()-t0:.0f}s")

        del input_ids, labels, outputs, enc

print(f"[4/4] Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DONE! {global_step} steps in {time.time()-t0:.0f}s")

# Test
print("\n--- Test: sentiment analysis ---")
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda")
test_model = PeftModel.from_pretrained(base, OUTPUT_DIR)
test_model.eval()
tests = [
    "判断这条弹幕的情感倾向（正面/负面/中性）\n绝了这技术太强了",
    "判断这条弹幕的情感倾向（正面/负面/中性）\n就这就这？？",
]
for t in tests:
    prompt = f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = test_model.generate(**enc, max_new_tokens=20, temperature=0.1)
    print(f"  Input: {t[:40]}...")
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    result = decoded.split('assistant\n')[-1] if 'assistant\n' in decoded else decoded
    print(f"  Output: {result}")

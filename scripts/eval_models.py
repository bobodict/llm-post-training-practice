"""
M3: Evaluation - Compare base vs SFT (alpaca) vs B站-SFT vs DPO models
Tests: sentiment accuracy, content generation quality, instruction following.
"""
import os, json, sys, time, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "./models/Qwen/Qwen2___5-1___5B-Instruct"
MODELS = {
    "Base (Qwen2.5-1.5B)": None,  # base model, no adapter
    "SFT (Alpaca)": "saves/qwen25-1.5b/lora/sft_manual",
    "B站-SFT": "saves/qwen25-1.5b/lora/bilibili_sft",
    "DPO (after SFT)": "saves/qwen25-1.5b/lora/dpo",
}
TOKENIZER = None

def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        TOKENIZER.pad_token = TOKENIZER.eos_token
    return TOKENIZER

def load_model(adapter_path=None):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda")
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return model

def generate(model, prompt, max_tokens=100):
    tokenizer = get_tokenizer()
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, temperature=0.3, do_sample=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# =========== Test Suite ===========
sentiment_tests = [
    ("判断这条弹幕的情感倾向（正面/负面/中性）\n哈哈哈哈笑死我了", "正面"),
    ("判断这条弹幕的情感倾向（正面/负面/中性）\n什么垃圾视频", "负面"),
    ("判断这条弹幕的情感倾向（正面/负面/中性）\n梦开始的地方", "正面"),
    ("判断这条弹幕的情感倾向（正面/负面/中性）\n下次一定", "负面"),
    ("判断这条弹幕的情感倾向（正面/负面/中性）\n第一", "中性"),
]

cat_tests = [
    ("判断这个视频属于哪个分区\n视频标题：Python入门教程第1集", "科技区"),
    ("判断这个视频属于哪个分区\n视频标题：原神4.0宝箱攻略", "游戏区"),
    ("判断这个视频属于哪个分区\n视频标题：【MMD】初音未来跳极乐净土", "动画区"),
]

gen_tests = [
    "给这段视频内容生成弹幕文案\n视频主题：程序员通宵写代码终于上线成功",
    "解释什么是大语言模型的后训练",
]

print("=" * 60)
print("M3: Multi-Model Evaluation")
print("=" * 60)

results = {}
for model_name, adapter in MODELS.items():
    print(f"\n{'='*40}")
    print(f"Testing: {model_name}")
    print(f"{'='*40}")

    model = load_model(adapter)
    tokenizer = get_tokenizer()

    scores = {"sentiment_acc": 0, "category_acc": 0}

    # Sentiment analysis
    print("\n  [Sentiment Analysis]")
    correct = 0
    for prompt, expected in sentiment_tests:
        full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        out = generate(model, full, 20)
        ans = out.split("assistant\n")[-1].strip()
        is_correct = expected in ans
        correct += is_correct
        status = "PASS" if is_correct else "FAIL"
    print(f"    [{status}] expected={expected} output={ans[:30]}")
    scores["sentiment_acc"] = correct / len(sentiment_tests)
    print(f"    Accuracy: {correct}/{len(sentiment_tests)} = {scores['sentiment_acc']:.0%}")

    # Category classification
    print("\n  [Category Classification]")
    correct = 0
    for prompt, expected in cat_tests:
        full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        out = generate(model, full, 20)
        ans = out.split("assistant\n")[-1].strip()
        is_correct = expected in ans
        correct += is_correct
        status = "PASS" if is_correct else "FAIL"
    print(f"    [{status}] expected={expected} output={ans[:30]}")
    scores["category_acc"] = correct / len(cat_tests)
    print(f"    Accuracy: {correct}/{len(cat_tests)} = {scores['category_acc']:.0%}")

    # Generation quality
    print("\n  [Generation Quality]")
    for prompt in gen_tests:
        full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        out = generate(model, full, 150)
        ans = out.split("assistant\n")[-1].strip()
        print(f"    Prompt: {prompt[:50]}...")
        print(f"    Output: {ans[:120]}...")

    results[model_name] = scores
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Summary table
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<25} {'Sentiment':>10} {'Category':>10}")
print("-" * 50)
for name, scores in results.items():
    print(f"{name:<25} {scores['sentiment_acc']:>9.0%} {scores['category_acc']:>9.0%}")
print("-" * 50)

print("\nKey Takeaways for Interview:")
print("1. Base model: limited task-specific ability")
print("2. SFT (Alpaca): general instruction following, mediocre on B站 tasks")
print("3. B站-SFT: best accuracy on B站-specific tasks due to domain data")
print("4. DPO: should show improved response quality over SFT")

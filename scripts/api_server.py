"""
M4: FastAPI inference server for LoRA-fine-tuned Qwen2.5-1.5B
Deploy the B站-SFT model as a REST API with OpenAI-compatible format.
Run: python api_server.py
Test: curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
"""
import os, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "./models/Qwen/Qwen2___5-1___5B-Instruct"
ADAPTER_PATH = "saves/qwen25-1.5b/lora/bilibili_sft"

app = FastAPI(title="B站-SFT Qwen2.5-1.5B API", version="1.0")

# ---- Load model at startup ----
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ---- Request/Response schemas (OpenAI-compatible) ----
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7

class ChatResponse(BaseModel):
    choices: list[dict]
    usage: dict

@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Build prompt from messages
    prompt = ""
    for msg in req.messages:
        if msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=req.max_tokens,
                             temperature=req.temperature, do_sample=True)
    elapsed = time.time() - t0

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract assistant response
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1]

    tokens = out.shape[1] - enc["input_ids"].shape[1]
    return ChatResponse(
        choices=[{"index": 0, "message": {"role": "assistant", "content": response}}],
        usage={"prompt_tokens": enc["input_ids"].shape[1], "completion_tokens": tokens,
               "total_tokens": out.shape[1], "time_seconds": elapsed}
    )

@app.get("/health")
def health():
    return {"status": "ok", "vram_gb": torch.cuda.memory_allocated()/1e9}

if __name__ == "__main__":
    print("Starting B站-SFT API server on http://localhost:8000")
    print("Test: curl -X POST http://localhost:8000/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8000)

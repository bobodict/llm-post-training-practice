"""FastAPI inference service for the domain LoRA adapter."""

from __future__ import annotations

import time

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ARTIFACTS_DIR, MODEL_PATH


ADAPTER_PATH = ARTIFACTS_DIR / "checkpoints" / "domain_dpo"
app = FastAPI(title="Chinese Domain Qwen API", version="1.1")
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    choices: list[dict]
    usage: dict


@app.on_event("startup")
def load_model():
    global tokenizer, model
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "cuda"
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if device.type != "cuda":
        base.to(device)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print(f"model_loaded device={device}")


def format_messages(messages):
    prompt = ""
    for message in messages:
        if message.role not in {"system", "user", "assistant"}:
            continue
        prompt += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(request: ChatRequest):
    prompt = format_messages(request.messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = {"max_new_tokens": request.max_tokens, "do_sample": request.temperature > 0}
    if request.temperature > 0:
        generation_kwargs["temperature"] = request.temperature
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**encoded, **generation_kwargs, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - t0
    new_tokens = output[0, encoded["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    completion_tokens = int(new_tokens.shape[0])
    prompt_tokens = int(encoded["input_ids"].shape[1])
    return ChatResponse(
        choices=[{"index": 0, "message": {"role": "assistant", "content": response}}],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "time_seconds": round(elapsed, 4),
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

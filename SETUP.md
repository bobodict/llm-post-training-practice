# Environment and Reproduction Guide

## Requirements

- Python 3.11
- NVIDIA GPU with at least 8GB VRAM for the default experiment
- PyTorch with a CUDA build for GPU training
- Qwen2.5-1.5B-Instruct model weights stored outside this repository

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, install the PyTorch build matching the local CUDA driver from the official PyTorch index before installing the remaining packages.

## Model path

The scripts read `MODEL_PATH`. Set it to the local model directory:

```powershell
$env:MODEL_PATH = "D:\models\Qwen2.5-1.5B-Instruct"
```

If the variable is not set, the default is `models/Qwen/Qwen2___5-1___5B-Instruct` under the repository root.

## Reproduction order

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python -m scripts.train_windows
python -m scripts.train_domain
python -m scripts.train_dpo
python -m scripts.eval_models
```

The DPO experiment uses the general SFT checkpoint as both the initialization and the reference policy. The evaluation script skips adapters that have not been trained yet and writes a structured result file instead of claiming unavailable numbers.

## Inference service

After Domain DPO completes:

```bash
python -m scripts.api_server
```

The service exposes `/health` and `/v1/chat/completions`. It is a local demonstration service and does not include authentication, batching, streaming, rate limiting, or production observability.

## Reproducibility notes

- The default seed is `42`; override it with `SEED`.
- Training and validation splits are deterministic.
- Classification evaluation uses greedy decoding.
- Model checkpoints, logs, and metrics are intentionally excluded from Git.
- The checked-in data is a small demonstration corpus, not a benchmark or a statistically representative production dataset.

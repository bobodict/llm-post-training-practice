# Research-Oriented Post-Training Practice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库整理为不含平台专属痕迹、具备基本可复现性和研究表达能力的中文场景后训练项目。

**Architecture:** 保留 Transformers + PEFT + PyTorch 的轻量结构；数据放在仓库 `data/`，脚本通过 `config.py` 统一路径与随机种子，训练结果和评测结果放入 `artifacts/`。GRPO 只增加可测试的 advantage 核心单元，不虚构完整分布式训练。

**Tech Stack:** Python 3.11, PyTorch, Transformers, PEFT, FastAPI, pytest/标准库测试。

---

### Task 1: Rename public domain terminology and relocate data

**Files:**
- Create: `data/domain_zh.json`
- Create: `data/dpo_domain_demo.json`
- Rename the old platform-specific domain script to `scripts/train_domain.py`
- Modify: `README.md`, `SETUP.md`, `scripts/api_server.py`, `scripts/eval_models.py`, `scripts/train_dpo.py`, `reports/interview-pack/*.md`
- Delete or rewrite: old platform-specific JD and ranking references

- [x] Copy the two JSON datasets into `data/`, rename their content labels to neutral Chinese-domain tasks, and update every script path.
- [x] Rename the domain SFT script and adapter names to `domain_sft`.
- [x] Replace platform/job-specific documentation with research-oriented wording.
- [x] Run a case-insensitive scan for the old platform name and require no matches.

### Task 2: Add shared configuration and correct SFT data handling

**Files:**
- Create: `scripts/config.py`
- Modify: `scripts/train_windows.py`, `scripts/train_domain.py`
- Create: `tests/test_data_and_algorithms.py`

- [x] Add repository-root path resolution, seed setup, and artifact directory helpers.
- [x] Split data deterministically into train/validation subsets.
- [x] Pass `attention_mask` to the model and set padding labels to `-100`.
- [x] Handle the final incomplete gradient-accumulation window.
- [x] Add tests for deterministic splitting, padding masking, and JSON loading.

### Task 3: Make DPO implementation auditable

**Files:**
- Modify: `scripts/train_dpo.py`
- Modify: `tests/test_data_and_algorithms.py`

- [x] Use the SFT adapter as the reference policy by default and document the choice.
- [x] Compute response-only token log probabilities using an explicit response mask.
- [x] Use a stable DPO loss helper and record epoch loss, chosen/rejected log-prob, and preference accuracy.
- [x] Save metrics as JSON under `artifacts/metrics/`.
- [x] Add CPU-only unit tests for response masking and DPO loss behavior.

### Task 4: Improve deterministic evaluation and add minimal GRPO core

**Files:**
- Modify: `scripts/eval_models.py`
- Create: `scripts/grpo_core.py`
- Modify: `reports/grpo_code_notes.md`
- Modify: `tests/test_data_and_algorithms.py`

- [x] Use greedy decoding for classification evaluation and exact label extraction.
- [x] Add confusion matrices, per-task accuracy, and JSON output.
- [x] Implement grouped reward normalization with a zero-variance fallback.
- [x] Test GRPO advantage shape, grouping, masking, and constant-reward behavior.

### Task 5: Rewrite reproducibility and application-facing documentation

**Files:**
- Rewrite: `README.md`, `SETUP.md`
- Rewrite: `reports/interview-pack/resume_star.md`, `interview_qa.md`, `core_code_walkthrough.md`, `application_checklist.md`, `ppt_prompt.md`
- Create: `requirements.txt`, `.gitignore`

- [x] Document actual completion status and resource limits honestly.
- [x] Add exact setup commands from repository root and a CPU test command.
- [x] Remove stale external audit/JD/ranking material that does not represent the project.
- [x] Run AST/JSON/unit/static checks and inspect the final diff.

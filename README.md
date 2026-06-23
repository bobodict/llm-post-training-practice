# B站 LLM 实习准备

🎯 目标岗位：**B站基础模型组（Index Team）大模型后训练实习生**

## 项目概述

在 RTX 4060 Laptop（8GB VRAM）上完成 Qwen2.5-1.5B 的 SFT → DPO 完整后训练链路，涵盖 B站弹幕/评论场景自定义数据集构建、DPO 算法手动实现、GRPO 代码精读、4 模型对比评测及 FastAPI 推理部署。

## 文件结构

```
scripts/
├── train_windows.py        # 基线 SFT（绕过 HF Trainer segfault）
├── train_bilibili.py       # B站领域数据 SFT
├── train_dpo.py            # DPO 偏好对齐（手动实现）
├── eval_models.py          # 4 模型对比评测
├── api_server.py           # FastAPI OpenAI 兼容推理服务
├── bilibili_zh.json        # B站弹幕/评论数据集（36条）
└── dpo_bilibili_demo.json  # DPO 偏好对数据集（8对）

reports/
├── interview-pack/         # 🎯 面试材料
│   ├── resume_star.md           # 4-5行 STAR 简历
│   ├── interview_qa.md          # 8个高频面试问答
│   ├── core_code_walkthrough.md # 代码讲解稿
│   ├── ppt_prompt.md            # PPT 生成提示词
│   └── application_checklist.md # 投递检查表
├── grpo_code_notes.md      # GRPO 算法代码精读笔记
├── ranking/                # 项目选型评分结果
└── audit/                  # LLaMA-Factory 项目审计
```

## 关键结果

| 指标 | 数值 |
|------|------|
| SFT loss | 20.4 → 2.95（500样本×3epoch） |
| VRAM 峰值 | 3.39 GB |
| DPO chosen-rejected gap | 2.1 → 5.1 |
| Base 零样本情感分析 | 80% |
| LoRA 可训练参数 | 0.59%（9.2M/1.55B） |

## 环境

- Windows 11 + RTX 4060 Laptop 8GB
- Python 3.11 + PyTorch 2.6 + CUDA 12.4
- LLaMA-Factory 0.9.6 + Peft + Transformers

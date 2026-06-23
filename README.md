# LLM Post-Training Practice

基于 LLaMA-Factory 的大模型后训练实践：SFT → DPO → GRPO 完整链路，在单卡 RTX 4060（8GB）上完成 Qwen2.5-1.5B 的指令微调、偏好对齐与强化学习代码精读。

## 项目内容

- **SFT（监督微调）**：通用指令数据 + 自定义中文场景数据集的 LoRA 微调
- **DPO（直接偏好优化）**：手动实现 DPO 算法，参考模型 log-prob 预计算 + 策略模型梯度更新
- **GRPO（组内相对策略优化）**：精读 EasyR1 源码中 `compute_grpo_outcome_advantage` 的实现
- **评测**：Base / SFT / Domain-SFT / DPO 四模型对比
- **部署**：FastAPI + OpenAI 兼容接口的 LoRA 模型推理服务

## 文件结构

```
scripts/
├── train_windows.py        # 基线 SFT（绕过 HF Trainer segfault）
├── train_bilibili.py       # 中文场景数据 SFT
├── train_dpo.py            # DPO 偏好对齐（手动实现）
├── eval_models.py          # 四模型对比评测
├── api_server.py           # FastAPI 推理服务
├── bilibili_zh.json        # 自定义中文数据集
└── dpo_bilibili_demo.json  # DPO 偏好对数据

reports/
├── interview-pack/         # 面试材料（简历/问答/代码讲解/PPT提示词）
├── grpo_code_notes.md      # GRPO 算法代码精读笔记
├── ranking/                # 项目选型评分
└── audit/                  # LLaMA-Factory 项目审计
```

## 关键结果

| 指标 | 数值 |
|------|------|
| SFT loss | 20.4 → 2.95（500样本×3epoch） |
| VRAM 峰值 | 3.39 GB |
| DPO chosen-rejected gap | 2.1 → 5.1 |
| LoRA 可训练参数 | 0.59% |

## 环境

- Windows 11 + RTX 4060 Laptop 8GB
- Python 3.11 + PyTorch 2.6 + CUDA 12.4
- LLaMA-Factory 0.9.6 + Peft + EasyR1

## 快速开始

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
# 从 ModelScope 下载 Qwen2.5-1.5B-Instruct
cp path/to/this-repo/scripts/* LLaMA-Factory/
python train_windows.py
```

详见 [SETUP.md](./SETUP.md)

# 投递检查表

| 项目项 | 准备材料 | 状态 |
| --- | --- | --- |
| JD 命中项目标题 | 基于 LLaMA-Factory 的大模型 SFT/DPO 后训练实践 | ✅ |
| 用户画像和运行深度 | 初学者，Python+PyTorch，RTX 4060 8GB，local-full-run | ✅ |
| 项目来源 | github.com/hiyouga/LLaMA-Factory (45k stars) + EasyR1 | ✅ |
| 最小运行路径 | train_windows.py（500样本×3epoch，7.5分钟） | ✅ |
| 核心代码讲解 | interview-pack/core_code_walkthrough.md（训练循环/DPO/GRPO） | ✅ |
| 可面试改造点 | Windows segfault 修复 + B站数据集 + DPO 手动实现 + GRPO 精读 | ✅ |
| 指标或替代表达 | SFT loss 20.4→2.95, DPO gap 2.1→5.1, VRAM 3.4GB | ✅ |
| 资源/时间/成本估算 | RTX 4060 8GB 本地免费，完整训练 30 分钟（3个模型） | ✅ |
| 面试拷问通过 | interview_qa.md（8个高频问题完整回答） | ✅ |

## 投递前检查

- [x] 简历 4-5 行 STAR 版本已就绪
- [x] 面试 Q&A 已覆盖：JD 匹配、技术挑战、DPO 实现、效果分析、失败教训
- [x] 代码讲解稿已覆盖：训练循环、DPO loss、GRPO advantage、LoRA 配置
- [x] PPT 提示词已生成（6-8页中文技术面试 PPT 大纲）
- [x] 项目代码可复现：5 个 Python 脚本均在 LLaMA-Factory 目录下
- [x] 关键数字可脱口而出：3.4GB VRAM、2.95 loss、5.1 DPO gap、0.59% LoRA params

## 面试当天速记卡

- **SFT**: 500 samples, 3 epochs, loss 20.4→2.95, 7.5min
- **DPO**: 8 pairs, 10 epochs, gap 2.1→5.1, manual implementation
- **GRPO**: group Z-score normalization, no critic model, 50% memory saved
- **Windows fix**: HF Trainer segfault → manual PyTorch loop, single sample per step
- **LoRA**: r=8, 0.59% trainable params, all-linear
- **Ascend NPU**: MindIE/vLLM-ascend/sglang-ascend, 910B/C chips

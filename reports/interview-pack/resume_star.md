# STAR 简历项目

## Profile Header

- 目标岗位：B站基础模型组 大模型后训练实习生（Index Team）
- 用户水平：初学者 → 经2周实践已掌握 SFT/DPO 全流程
- 技术栈偏好：Python + PyTorch + LLaMA-Factory + Peft
- 时间预算：2周以上
- 资源条件：RTX 4060 Laptop 8GB VRAM
- 运行深度：local-full-run（完整本地跑通 SFT→DPO→评测→部署）
- 当前项目状态：已完整跑通 SFT/DPO/评测/API部署，GRPO 完成代码精读

## 4-5 行版本（可直接投递）

### 项目一：基于 LLaMA-Factory 的大模型 SFT/DPO 后训练实践

**S/T**：针对 B 站大模型后训练实习生岗位要求（SFT、DPO、GRPO、蒸馏），基于 LLaMA-Factory 在 RTX 4060（8GB）上复现 Qwen2.5-1.5B 的完整后训练链路。

**A1-环境适配**：解决 Windows 环境下 HuggingFace Trainer segfault 问题，手动实现 PyTorch 训练循环（单样本前向→梯度累积→step），VRAM 峰值仅 3.4GB；构建 B 站弹幕/评论场景的自定义指令数据集（情感分析/内容分类/弹幕生成 36 条）。

**A2-DPO 实现**：手动实现 DPO 算法（基于论文公式，参考模型 log-prob 预计算 + 策略模型梯度更新），8 组偏好对训练 10 轮后 chosen-rejected log-prob 差距从 2.1 扩大至 5.1；精读 EasyR1 源码中 GRPO 的 `compute_grpo_outcome_advantage` 函数，整理组内 Z-score 归一化 advantage 计算逻辑。

**R**：产出 5 个可复现 Python 脚本（基线 SFT/自定义数据 SFT/DPO 训练/4 模型对比评测/FastAPI 推理服务）；Base 模型零样本情感分析 80%，B站-SFT 分区分类较通用 SFT 提升 33pp；Baseline SFT loss 20.4→2.95（500样本×3epoch）。

### 项目二：self-llm 开源大模型食用指南

**S/T**：参与 Datawhale 开源项目 self-llm（12k+ stars），梳理 B站 Index-1.9B 模型及 Ascend NPU 部署教程。

**A**：系统阅读 Index-1.9B 的 FastAPI/LangChain/WebDemo/LoRA 四套教程；梳理 Ascend NPU（昇腾 910B/C）平台的 MindIE/vLLM-ascend/sglang-ascend 三种推理框架部署流程。

**R**：作为工程储备，掌握 B站自研模型部署全流程；为后续贡献 Ascend NPU 模型教程打基础。

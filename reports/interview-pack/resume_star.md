# Research Project Summary

## Project: Chinese Domain LLM Post-Training Practice

**背景**：面向中文短文本任务，研究小规模领域数据进行模型后训练时的任务适应、泛化保持和偏好对齐问题。

**方法**：使用 Qwen2.5-1.5B 和 LoRA，在单卡 8GB 显存环境下实现通用 SFT、领域 SFT 和手写 DPO；实现 response-only log-prob、reference-adjusted preference margin，并加入确定性评测与验证集指标。

**工程**：修正 padding label mask、梯度累积尾 batch、随机种子和路径管理；提供 CPU 可运行的 DPO/GRPO 核心单元测试，以及 FastAPI 推理接口。

**分析**：对比 Base、General-SFT、Domain-SFT 和 Domain-DPO，重点观察小数据过拟合、通用能力退化和偏好数据质量对训练稳定性的影响。

**边界**：GRPO 部分目前完成组内 advantage 的最小实现、源码分析和单元测试，尚未声称完成大规模多卡 RL 训练。

## 面试时的 30 秒介绍

我做的是一个中文场景大模型后训练实验。为了在有限显存下看清算法本身，我没有只调用高层 Trainer，而是实现了 LoRA SFT 和 DPO 的关键计算，并把数据划分、padding mask、确定性评测和指标落盘补齐。实验重点不是追求一个漂亮的单点分数，而是分析小数据后训练为什么会导致过拟合或能力退化，以及怎样通过验证集、参考策略和更可靠的评测发现问题。

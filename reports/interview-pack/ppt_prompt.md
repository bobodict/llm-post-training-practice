# PPT 生成提示词

请根据以下项目资料生成一份 6-8 页中文技术面试展示 PPT 大纲。

目标：帮助候选人快速讲清 B站大模型后训练实习项目的技术深度和面试亮点。

## 项目资料

- **目标岗位**：B站基础模型组（Index Team）大模型后训练实习生
- **JD 核心要求**：SFT、DPO、GRPO、蒸馏、vLLM 推理、LLaMA-Factory/EasyR1/Verl 框架
- **项目来源**：LLaMA-Factory (github.com/hiyouga/LLaMA-Factory, 45k stars) + EasyR1 (github.com/hiyouga/EasyR1)
- **用户水平**：初学者→经过2周实践掌握完整 SFT/DPO 流程
- **资源条件**：RTX 4060 Laptop 8GB VRAM
- **运行深度**：local-full-run（完整本地跑通）

## Baseline 命令

```bash
python train_windows.py      # SFT 基线：500样本×3epoch，loss 20.4→2.95
python train_bilibili.py     # B站-SFT：36条领域数据
python train_dpo.py           # DPO：8对偏好数据，loss 0.69→0.56
python eval_models.py        # 4模型对比评测
python api_server.py          # FastAPI 推理服务
```

## 我的改造点

1. 解决 Windows HF Trainer segfault，手动实现 PyTorch 训练循环
2. 构建 B站弹幕/评论场景自定义数据集（3类任务，36条）
3. 手动实现 DPO 算法（参考模型 log-prob 预计算 + 策略模型梯度更新）
4. 精读 EasyR1 GRPO 代码，整理 `compute_grpo_outcome_advantage` 逻辑
5. 4 模型对比评测 + 失败分析

## 指标/效果

- Baseline SFT: loss 20.4→2.95, VRAM 峰值 3.4GB
- DPO: chosen-rejected log-prob 差距 2.1→5.1
- 评测: Base 情感80%, B站-SFT 分区+33pp vs 通用SFT
- DPO 退化（8样本过拟合）→ 真实的工程教训

## 建议页面

1. **项目背景与 JD 匹配**（SFT/DPO/GRPO 三大后训练方法覆盖）
2. **训练流程总览**（Base→SFT→B站-SFT→DPO，模型路线图）
3. **核心技术挑战**：Windows segfault 排查 + GPU 显存管理（VRAM 3.4GB）
4. **DPO 算法实现**：公式推导 + 代码关键步骤 + chosen-rejected 差距变化图
5. **GRPO 代码精读**：5步流程图（Rollout→Reward→Advantage→Loss→Checkpoint）
6. **评测与失败分析**：4模型对比表格 + DPO 过拟合原因分析
7. **部署与工程化**：FastAPI 服务 + Ascend NPU 部署方案
8. **面试亮点与后续计划**：能用口语讲清的核心概念清单

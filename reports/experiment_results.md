# Experiment Results

本报告记录一次在 RTX 4060 Laptop 8GB、CUDA 12.4、Qwen2.5-1.5B-Instruct 上的实际运行结果。随机种子为 `42`，所有分类评测使用 greedy decoding。

## Training

| Stage | Train | Validation | Notes |
| --- | ---: | ---: | --- |
| General SFT | 4.3817 -> 2.7290 | 3.5600 -> 2.6446 | 6 train / 2 validation samples |
| Domain SFT | 2.8637 -> 1.1537 | 2.2695 -> 1.4596 | 47 train / 12 validation samples; initialized from General SFT |
| Domain DPO | loss 0.6770 -> 0.4134 | preference accuracy 100% | 6 train / 2 validation pairs; policy/reference initialized from Domain SFT |

Domain SFT 在当前 60 条领域样本的划分上，验证 loss 从 `2.2695` 下降到 `1.4596`；消融实验显示 `5 epoch + 1e-4` 是本次三组配置中的最佳观察结果。DPO 的验证集只有 2 对，100% 不能解释为稳定的偏好对齐能力。

## Deterministic evaluation

独立评测集包含 12 条：6 条情感判断和 6 条主题分类。分类任务显式提供闭集标签，并要求模型只输出一个标签。

| Model | Sentiment | Topic |
| --- | ---: | ---: |
| Base | 6/6 (100%) | 4/6 (66.7%) |
| General SFT | 6/6 (100%) | 4/6 (66.7%) |
| Domain SFT | 6/6 (100%) | 4/6 (66.7%) |
| Domain DPO | 6/6 (100%) | 4/6 (66.7%) |

## Interpretation

当前结果不能证明领域 SFT 或 DPO 带来了评测提升。情感任务四个模型均为 100%，存在 ceiling effect；主题分类四个模型均为 66.7%，错误主要是把“科技区-编程”预测为“知识区-教育”，说明领域数据和标签边界仍不足以让后训练产生可观测差异。

下一轮实验应：

1. 继续扩大每个主题的训练和测试样本，减少标签语义重叠。
2. 使用 JSON 输出格式和 schema 校验，避免自然语言解析误差。
3. 使用 macro-F1、混淆矩阵和多随机种子均值，而不是只报告单次准确率。
4. 比较不同 epoch、学习率和通用数据混合比例，量化过拟合与遗忘。

原始 JSON 指标位于本地 `artifacts/metrics/` 和 `artifacts/evaluation/`，模型权重未提交到 Git。

## GRPO single-GPU smoke test

`python -m scripts.grpo_smoke` 使用 `domain_dpo` adapter 做真实采样，针对每个 prompt 生成 4 个回答，按最终数字答案计算可验证 reward，再计算组内 advantage 并执行 clipped policy update。结果写入 `artifacts/metrics/grpo_smoke.json`，checkpoint 写入 `artifacts/checkpoints/grpo_smoke/`。

本次运行的 4 个 prompt 中，前 3 组 reward 方差为 0，按设计产生零 advantage；第 4 组 reward 为 `[0, 0, 0, 1]`，均值 `0.25`、标准差约 `0.4330`，4 个 response 均产生非零 advantage，loss 为 `0.0048`。这证明了 rollout、reward、组内标准化、策略损失和 LoRA 更新的单卡链路，但不等价于大规模 RL 训练或泛化结论。

## API smoke test

服务使用最终 `Domain-DPO` adapter 启动成功：`/health` 返回 `{"status": "ok", "device": "cuda"}`，`/v1/chat/completions` 返回合法的 choices 和 usage 字段。对“解释什么是监督学习”的请求，模型返回了泛化拒答文本，这是一个可复现的生成 bad case，说明当前小规模后训练还存在指令跟随退化问题。

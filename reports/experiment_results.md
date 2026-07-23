# Experiment Results

本报告记录一次在 RTX 4060 Laptop 8GB、CUDA 12.4、Qwen2.5-1.5B-Instruct 上的实际运行结果。随机种子为 `42`，所有分类评测使用 greedy decoding。

## Training

| Stage | Train | Validation | Notes |
| --- | ---: | ---: | --- |
| General SFT | 4.3817 -> 2.7290 | 3.5600 -> 2.6446 | 6 train / 2 validation samples |
| Domain SFT | 3.4147 -> 1.0296 | 2.8278 -> 1.7211 | 29 train / 7 validation samples |
| Domain DPO | loss 0.6874 -> 0.5471 | preference accuracy 100% | 6 train / 2 validation pairs |

Domain SFT 在第 5 轮出现验证 loss 从 `1.6734` 回升到 `1.7211`，说明小数据上已经出现轻微过拟合迹象。DPO 的验证集只有 2 对，100% 不能解释为稳定的偏好对齐能力。

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

## API smoke test

服务使用最终 `Domain-DPO` adapter 启动成功：`/health` 返回 `{"status": "ok", "device": "cuda"}`，`/v1/chat/completions` 返回合法的 choices 和 usage 字段。对“解释什么是监督学习”的请求，模型返回了泛化拒答文本，这是一个可复现的生成 bad case，说明当前小规模后训练还存在指令跟随退化问题。

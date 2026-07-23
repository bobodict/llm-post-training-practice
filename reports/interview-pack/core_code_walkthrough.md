# Core Code Walkthrough

## 1. Reproduction order

```bash
python -m unittest discover -s tests -v
python -m scripts.train_windows
python -m scripts.train_domain
python -m scripts.train_dpo
python -m scripts.eval_models
```

## 2. SFT training loop

每条样本经过 tokenizer 后同时保留 `input_ids` 和 `attention_mask`。padding 位置在 labels 中改为 `-100`，因此交叉熵不会把补齐 token 当作监督目标。单样本前向后将 loss 除以梯度累积步数，最后一个不足整组的窗口会重新缩放梯度。

这比单纯打印训练 loss 更重要：如果 padding 被计入 loss，loss 下降不能准确反映模型对有效回答 token 的学习。

## 3. DPO implementation

DPO 使用通用 SFT checkpoint 初始化 policy，并使用同一个 SFT checkpoint 的冻结副本作为 reference。对每个 preference pair，只计算回答部分 token 的 log-prob：

```text
margin = (log pi(chosen) - log ref(chosen))
       - (log pi(rejected) - log ref(rejected))
loss = -log sigmoid(beta * margin)
```

训练脚本会记录训练集和验证集 preference accuracy，避免只看某一次 chosen-rejected 差值。

## 4. GRPO advantage

`scripts/grpo_core.py` 将同一 prompt 生成的多个回答归为一组，先把 token-level reward 汇总为 response score，再在组内做均值和标准差归一化。组内 reward 方差接近零时返回零 advantage，避免数值不稳定；response mask 会清除 padding 位置。

当前实现是可测试的算法核心，不等于包含 rollout、vLLM、分布式 worker 的完整 GRPO 训练系统。

## 5. Evaluation

分类评测使用 greedy decoding，截取新增 token，并输出每个任务的准确率和混淆矩阵到 `artifacts/evaluation/comparison.json`。数据规模仍然很小，因此结果只用于验证实验流程，不能当作通用 benchmark 结论。

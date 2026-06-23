# 核心代码讲解稿

## 1. 入口命令

```bash
# Baseline SFT 训练（500 样本, 3 epoch, 7.5 分钟）
python train_windows.py

# B站领域数据 SFT
python train_bilibili.py

# DPO 偏好对齐训练
python train_dpo.py

# 4 模型对比评测
python eval_models.py

# API 推理服务
python api_server.py
```

## 2. 训练循环核心实现（train_windows.py）

这是绕过 HF Trainer segfault 的关键代码：

```python
for epoch in range(NUM_EPOCHS):
    for idx, text in enumerate(texts):
        # 核心1：单样本 tokenize → cuda
        enc = tokenizer(text, truncation=True, max_length=CUTOFF_LEN, ...)
        input_ids = enc["input_ids"].cuda()
        labels = enc["input_ids"].clone().cuda()

        # 核心2：前向 + loss 缩放（梯度累积）
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        # 核心3：累积到 GRAD_ACCUM 步后更新参数
        if (idx + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 核心4：显存管理
        del input_ids, labels, outputs, enc
        if (idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
```

面试讲解要点：
- 为什么单样本？→ 8GB 显存，batch > 1 会 OOM
- 为什么 GRAD_ACCUM=4？→ 等价于 batch_size=4，平衡训练稳定性和显存
- 为什么手动 gc？→ Windows CUDA 内存碎片，周期性清理防止 OOM

## 3. DPO loss 实现（train_dpo.py）

```python
# DPO loss 公式: -log(sigma(beta * (log_ratio_chosen - log_ratio_rejected)))

# Step 1: 计算策略模型对 chosen 和 rejected 的 log prob（需要梯度）
pi_chosen = compute_log_probs(policy, tokenizer, prompt, chosen_response)
pi_rejected = compute_log_probs(policy, tokenizer, prompt, rejected_response)

# Step 2: 减去参考模型的 log prob（预计算，无梯度）
log_ratio_chosen = pi_chosen - ref_chosen_logp
log_ratio_rejected = pi_rejected - ref_rejected_logp

# Step 3: DPO loss
diff = log_ratio_chosen - log_ratio_rejected
loss = -F.logsigmoid(BETA * diff)
```

面试讲解要点：
- `log_ratio` 是「策略模型相对于参考模型对该回答的偏好程度」
- `diff` 是 chosen 和 rejected 的差距，正数表示模型更偏好 chosen
- `-log(sigmoid(...))` 确保 loss 随着 diff 增大而减小
- β=0.1 是温度参数，控制偏好强度

## 4. GRPO advantage 计算（EasyR1 verl/trainer/core_algos.py:176）

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index):
    bsz = scores.shape[0]
    # 按 prompt 分组（同一 prompt 的 n 个 response 归为一组）
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # 组内计算均值和标准差
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    # Z-score 归一化 → advantage
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
```

面试讲解要点：
- 为什么组内归一化？→ 不同 prompt 难度不同，绝对分数不可比，相对分数才有意义
- 为什么除以 std？→ 控制优势估计的方差，避免高方差 prompt 主导训练
- 为什么 n > 1？→ 组内只有一个 response 时 std=0，无法归一化

## 5. LoRA 模型结构（关键配置）

```python
lora_config = LoraConfig(
    r=8,              # LoRA 秩，控制可训练参数数量
    lora_alpha=16,    # 缩放因子，通常 2×r
    target_modules="all-linear",  # 作用于所有线性层
    lora_dropout=0.05
)
```

可训练参数：9,232,384 / 1,552,946,688 = 0.59%
面试讲解要点：r 越大表达能力越强但参数越多；LoRA 的数学原理是低秩分解 W + AB（A∈R^{d×r}, B∈R^{r×k}）

## 6. 评测代码（eval_models.py）

4 模型对比：Base → SFT(Alpaca) → B站-SFT → DPO
评测维度：情感分析（5题）、分区分类（3题）、生成质量（2题）
结果：Base 80% 情感准确率最高 → B站-SFT 在分区分类上优于通用 SFT → DPO 过拟合导致退化

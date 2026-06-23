# GRPO (Group Relative Policy Optimization) 代码精读笔记
## 来源：EasyR1 (hiyouga/EasyR1)，基于字节 veRL

### 1. GRPO 算法核心思想

GRPO 是 DeepSeek-R1 提出的 RL 后训练算法，核心创新：**无需 Critic 模型，用组内相对比较替代绝对价值评估**。

传统 PPO：需要 Actor + Critic + Reward Model + Reference Model（4个模型）
GRPO：只需要 Actor + Reference Model + Reward Function（3个模型，省掉 Critic ≈ 省 50% 显存）

### 2. 核心代码流程（verl/trainer/core_algos.py:176）

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index):
    # Step 1: 计算每个 response 的总分
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)

    # Step 2: 按 prompt 分组（同一个 prompt 生成 n 个不同 response）
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])  # index[i] 标识属于哪个 prompt

    # Step 3: 计算组内均值和标准差
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    # Step 4: 组内归一化得到 advantage（核心公式！）
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns  # advantage = return（GRPO 无 GAE）
```

### 3. 完整训练流程（5步循环）

```
Step 1 - Rollout（生成）：用 vLLM 对每个 prompt 生成 n=5 个 response
Step 2 - Reward（打分）：用 reward function 给每个 response 打分
Step 3 - Advantage（优势估计）：组内归一化 (score - mean) / std
Step 4 - PPO Loss（策略更新）：对 advantage > 0 的 response 做加权优化
Step 5 - Checkpoint：保存模型
```

### 4. GRPO vs DPO 对比

| 维度 | DPO | GRPO |
|------|-----|------|
| 数据需求 | 偏好对 (chosen, rejected) | 只需 prompt + reward function |
| 训练方式 | 离线，一次性 | 在线，迭代 rollout |
| 是否需要 Critic | 否 | 否 |
| 显存占用 | 低 | 中（需 vLLM 推理） |
| 适用场景 | 小规模偏好数据 | 大规模在线 RL |

### 5. 面试要点

Q: 为什么 GRPO 比 PPO 省显存？
A: GRPO 不需要 Critic（Value）模型。PPO 需要 4 个模型（Actor/Critic/Ref/Reward），GRPO 只需要 3 个。
   组内相对比较替代了 Critic 的绝对价值估计——同一个 prompt 的多个 response 之间比好坏即可。

Q: GRPO 的 advantage 怎么算的？
A: 对同一个 prompt 的 n 个 response 打分后，做组内 Z-score 归一化：(score - mean) / std。
   这保证了 advantage 在不同 prompt 之间具有可比性。

Q: GRPO 的 n 选多少合适？
A: 论文用 n=4~16。n 太小方差大，n 太大推理成本高。通常 n=5~8 平衡效果和成本。

### 6. EasyR1 项目结构速查

- `verl/trainer/core_algos.py` - GRPO/DAPO 等算法实现
- `verl/trainer/ray_trainer.py` - 分布式训练主逻辑
- `examples/qwen3_4b_math_grpo_lora.sh` - LoRA + GRPO 示例（适合单卡！）
- 支持 adv_estimator: grpo / dapo / grpo_passk / reinforce_plus 等

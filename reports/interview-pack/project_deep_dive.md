# 项目深度讲解与面试准备手册

> 本文严格按照当前仓库的代码、数据和实际运行结果整理。面试时要区分“当前已经实现的内容”和“后续可以改进的内容”，不要把单卡 smoke experiment 包装成大规模生产训练系统。

## 1. 项目一句话

这是一个基于 Qwen2.5-1.5B-Instruct 和 LoRA 的中文领域大模型后训练实验仓库，手写并实际运行了 General SFT、Domain SFT、Domain DPO，以及一个单卡 GRPO rollout smoke experiment，研究小规模领域数据下的任务适应、偏好对齐和泛化之间的关系。

核心链路：

~~~text
Base model
    |
    v
General SFT adapter
    |
    v
Domain SFT adapter
    |
    v
Domain DPO adapter
    |
    v
GRPO single-GPU smoke checkpoint
~~~

仓库：https://github.com/bobodict/llm-post-training-practice

本地目录：D:\llm-post-training-practice

## 2. 30 秒项目介绍

面试时可以先这样说：

> 我的项目研究的是小规模中文领域数据下的大模型后训练。我使用 Qwen2.5-1.5B-Instruct 作为基座模型，通过 LoRA 先做通用中文指令 SFT，再从通用适配器继续做领域 SFT，之后用领域 SFT 同时初始化 DPO 的 policy 和 reference，最后实现了一个单卡 GRPO rollout smoke experiment。项目不是简单调用训练框架，而是自己实现了 SFT 训练循环、padding mask、response-only log-prob、DPO loss、组内 advantage 和 clipped policy objective，并保存训练指标、评测结果和 checkpoint。实验结果显示训练 loss 和偏好指标改善，但 12 条闭集评测没有观察到四个模型之间的准确率差异，这说明当前数据和测试集太小，项目主要证明了后训练链路和实验方法，而不是声称取得了 benchmark 提升。

这一段包含了研究问题、模型、方法、工程实现、结果和限制，适合作为开场回答。

## 3. 研究问题和实验假设

### 3.1 为什么做后训练

预训练模型拥有通用语言能力，但不一定稳定完成特定领域任务，例如：

- 按指定标签做中文情感分类；
- 按固定分区做主题分类；
- 按目标风格生成技术标题；
- 输出结构清晰的实验摘要；
- 对短文本做基本安全审核；
- 在高质量回答和低质量回答之间做出偏好选择。

后训练的主要阶段可以理解为：

1. SFT：用人工示范让模型学习指令和回答格式。
2. DPO：用 chosen/rejected 偏好对调整回答排序。
3. GRPO/RL：采样多个回答，用 reward 产生相对优势，再更新策略。

### 3.2 项目研究问题

项目关注的不是“loss 是否下降”这一件事，而是：

> 在小规模中文领域数据和有限显存条件下，General SFT、Domain SFT、DPO 和最小 GRPO 更新分别改变了什么？训练指标的改善是否能转化为独立测试集上的任务提升？

可检验假设：

- H1：General SFT 会改善统一 instruction-response 格式的拟合。
- H2：Domain SFT 会降低领域训练/验证 loss，并可能提升领域任务适应。
- H3：DPO 会提高 chosen 相对于 rejected 的 reference-adjusted margin。
- H4：小数据上的训练指标改善不一定带来泛化提升，可能出现过拟合或通用能力退化。
- H5：GRPO 只有在同一 prompt 的多个 rollout 存在 reward 方差时才有有效组内 advantage。

## 4. 目录和执行关系

~~~text
data/
  general_zh_demo.json       # General SFT，8 条
  domain_zh.json             # Domain SFT，59 条
  dpo_domain_demo.json       # DPO 偏好对，8 对
  eval_domain.json           # 独立闭集评测，12 条

scripts/
  config.py                  # 路径、种子、数据切分、JSON 工具
  train_windows.py           # General SFT
  train_domain.py            # 继承 General SFT 的 Domain SFT
  train_dpo.py               # 继承 Domain SFT 的 DPO
  dpo_core.py                # DPO 数学函数
  grpo_core.py               # GRPO advantage 和 clipped loss
  grpo_smoke.py              # 真实 rollout 和单卡更新
  ablation.py                # Domain SFT 超参数消融
  eval_models.py             # 四个模型的确定性评测
  api_server.py              # FastAPI 推理服务

tests/test_data_and_algorithms.py
reports/interview-pack/
artifacts/                   # 本地生成，不提交权重
~~~

运行顺序：

~~~bash
python -m unittest discover -s tests -v
python -m scripts.train_windows
python -m scripts.train_domain
python -m scripts.train_dpo
python -m scripts.eval_models
python -m scripts.grpo_smoke
python -m scripts.api_server
~~~

阶段依赖：

~~~text
train_windows.py -> artifacts/checkpoints/general_sft
train_domain.py  -> artifacts/checkpoints/domain_sft
train_dpo.py     -> artifacts/checkpoints/domain_dpo
grpo_smoke.py    -> artifacts/checkpoints/grpo_smoke
~~~

脚本会检查前置 adapter 是否存在，避免把未初始化阶段误认为完整实验。

## 5. 模型、硬件和 LoRA

### 5.1 模型和硬件

实际运行环境：

~~~text
GPU: RTX 4060 Laptop 8GB
CUDA: 12.4
PyTorch: 2.6.0
Transformers: 5.6.0
PEFT: 0.18.1
Python: 3.11
Model: Qwen2.5-1.5B-Instruct
~~~

GPU 上用 float16，CPU 上用 float32。选择 1.5B 不是因为它代表模型能力上限，而是为了在 8GB 显存上低成本反复验证训练代码和算法。

### 5.2 LoRA 原理

全参数微调可以写成：

~~~text
W' = W + Delta W
~~~

LoRA 用低秩矩阵近似更新：

~~~text
W' = W + (alpha / r) * B A
~~~

基座 W 冻结，只训练低秩矩阵 A 和 B。这样可以减少梯度和 AdamW 优化器状态的显存开销，并且每个阶段只需保存较小 adapter。

General SFT 的 LoRA 配置：

~~~text
r = 8
alpha = 16
dropout = 0.05
target_modules = all-linear
~~~

LoRA 的限制是更新空间受 rank 限制，低秩不等于不会过拟合；数据极少时，adapter 仍然可能记住训练样本。

## 6. 数据设计和切分

### 6.1 数据规模

| 文件 | 数量 | 用途 |
|---|---:|---|
| general_zh_demo.json | 8 | General SFT |
| domain_zh.json | 59 | Domain SFT |
| dpo_domain_demo.json | 8 对 | DPO |
| eval_domain.json | 12 | 独立闭集评测 |

split_records 使用 random.Random(seed=42) 打乱索引，验证集大小为 max(1, round(n*0.2))：

- General SFT：6 train / 2 validation；
- Domain SFT：47 train / 12 validation；
- DPO：6 train / 2 validation。

### 6.2 Domain SFT 数据

覆盖：

- 情感判断；
- 主题分类；
- 技术标题生成；
- 实验摘要；
- 内容安全审核；
- 中文短文本改写和结构化输出。

基本格式：

~~~json
{
  "instruction": "给这个视频起一个吸引人的标题",
  "input": "视频内容：采集中文短文本并做情感分析",
  "output": "Python采集中文短文本并做情感分析实战"
}
~~~

### 6.3 DPO 偏好数据

每条数据有一个问题、一个更好的 chosen 回答和一个较差的 rejected 回答：

~~~json
{
  "instruction": "什么是大语言模型的后训练？",
  "chosen": "说明 SFT、DPO、RLHF 以及它们的作用。",
  "rejected": "训练完以后再训练一下。"
}
~~~

低质量回答通常表现为过于简略、事实不准确、没有回答问题、表达不专业或缺少安全边界。

### 6.4 独立评测

12 条评测样本包括：

- 6 条 sentiment；
- 6 条 topic。

评测 prompt 显式提供候选标签，并要求只输出一个标签。这样便于确定性解析，但数据量太小，不能当成公开 benchmark。

## 7. Prompt、tokenizer 和 label mask

项目手动使用 Qwen 风格对话格式：

~~~text
<|im_start|>user
指令和输入
<|im_end|>
<|im_start|>assistant
目标回答
<|im_end|>
~~~

推理时只拼到 assistant 起始位置，让模型继续生成。

tokenizer 流程：

1. tokenize；
2. 截断到最大长度；
3. padding="max_length"；
4. 没有 pad token 时使用 eos token；
5. attention_mask == 0 的 label 改为 -100；
6. CrossEntropyLoss 忽略 -100。

### 7.1 padding mask 为什么重要

~~~text
input_ids:      [真实, 真实, PAD, PAD]
attention_mask: [1,    1,    0,   0]
labels:         [真实, 真实, -100, -100]
~~~

如果 padding 不 mask，模型会学习预测人为补齐 token，loss 会混入无意义的监督信号。

### 7.2 当前 SFT 的诚实边界

当前 SFT 的 labels 是整个 input_ids 的复制，只把 padding 改成 -100。因此当前 SFT 会对非 padding 的 user prompt 和 assistant response 都计算 loss，不是严格的 completion-only SFT。

更严格的实现应当是：

~~~text
system/user tokens        -> -100
assistant response tokens -> 保留真实 token id
padding tokens             -> -100
~~~

面试时可以说：

> 当前项目先验证了训练链路，所以实现的是 padding-only mask；我知道更标准的 instruction SFT 还应做 completion-only mask，这是明确的下一步改进，并且需要增加 assistant 起始位置对齐的单元测试。

不要假装当前代码已经实现了 response-only SFT。DPO 才是明确的 response-only log-prob。

## 8. General SFT 原理和实现

### 8.1 数学目标

给定 token 序列 x_1,...,x_T，因果语言模型最大化：

~~~text
log P(x_1,...,x_T) = sum_t log P(x_t | x_<t)
~~~

当前训练目标可以写成：

~~~text
L_SFT = - sum_t m_t log pi_theta(x_t | x_<t)
~~~

其中 m_t=0 主要对应 padding。

### 8.2 训练循环

train_windows.py：

1. seed 42；
2. 读取 8 条数据并切分 6/2；
3. 加载 base 和 tokenizer；
4. 新建 LoRA；
5. 格式化 chat prompt；
6. tokenize、padding、mask；
7. 前向计算 causal LM loss；
8. loss 除以 GRAD_ACCUM=4 后反向；
9. 每 4 条样本执行一次 AdamW；
10. 尾 batch 不足 4 条时重新缩放梯度；
11. 每 epoch 计算训练和验证 loss；
12. 保存 adapter 和 JSON 指标。

尾 batch 缩放的目的，是避免最后只有 k 条样本时梯度幅度被错误缩小。

General SFT 实际结果：

~~~text
train loss:       4.3817 -> 2.7290
validation loss:  3.5600 -> 2.6446
~~~

这只能说明当前小切分上拟合改善，不能直接说明真实泛化提升。

## 9. Domain SFT 原理和实现

### 9.1 初始化关系

General SFT：

~~~text
Base + new LoRA adapter
~~~

Domain SFT：

~~~text
Base + load(general_sft adapter) + continue training
~~~

train_domain.py 会检查 general_sft 存在，然后使用 PeftModel.from_pretrained(..., is_trainable=True) 加载 General SFT。因此 Domain SFT 真正继承了 General SFT，而不是随机新建 adapter。

### 9.2 配置和结果

默认配置：

~~~text
max length = 256
gradient accumulation = 4
epochs = 5
learning rate = 1e-4
~~~

实际结果：

~~~text
train loss:       2.8637 -> 1.1537
validation loss:  2.2695 -> 1.4596
~~~

消融结果：

| 实验 | epoch | lr | 最佳验证 loss |
|---|---:|---:|---:|
| ablation_3ep_lr1e-4 | 3 | 1e-4 | 1.6285 |
| ablation_5ep_lr1e-4 | 5 | 1e-4 | 1.4596 |
| ablation_5ep_lr5e-5 | 5 | 5e-5 | 1.5478 |

本次固定 seed 和切分下，最佳观察配置是 5 epoch + 1e-4。它不是全局最优，只是三组配置中的比较结果。

### 9.3 为什么 loss 降而准确率不变

- 评测任务过于简单，Base 已有较高准确率；
- 评测样本过少，准确率粒度太粗；
- 训练和评测分布不完全相同；
- 主题标签存在语义边界重叠；
- teacher-forcing loss 和 greedy decoding 任务准确率不是同一个指标。

## 10. DPO 原理和实现

### 10.1 DPO 解决什么问题

SFT 学习一个正向示范，DPO 学习：

~~~text
chosen 应该比 rejected 更可能
~~~

DPO 不需要先训练 reward model，也不需要单独运行 PPO。

### 10.2 response-only log-prob

给定 prompt x 和回答 y：

~~~text
log pi(y|x) = sum_t log pi(y_t | x, y_<t)
~~~

代码根据 prompt token 长度切分：

~~~python
logits = outputs.logits[:, prompt_len - 1:-1, :]
labels = input_ids[:, prompt_len:]
token_log_probs = torch.log_softmax(logits, dim=-1).gather(
    -1, labels.unsqueeze(-1)
).squeeze(-1)
return token_log_probs.sum()
~~~

prompt 对 chosen/rejected 相同，真正需要比较的是 response。只计算 response 可以避免 prompt 内容和长度污染偏好 margin。代码会给 response 追加 <|im_end|>，使结束 token 也参与概率计算。

### 10.3 DPO 公式

~~~text
chosen_margin   = log pi(chosen)   - log pi_ref(chosen)
rejected_margin = log pi(rejected) - log pi_ref(rejected)
preference_margin = chosen_margin - rejected_margin
L_DPO = -log sigmoid(beta * preference_margin)
~~~

当前 beta=0.1。beta 控制偏好目标尺度，学习率控制参数更新步长，二者不是同一个东西。

### 10.4 为什么 reference 用 Domain SFT

当前关系：

~~~text
policy:    Base + Domain SFT adapter -> trainable
reference: Base + Domain SFT adapter -> frozen
~~~

Domain SFT 已经具备领域格式和基础能力，使用它作为 reference，可以让 DPO 主要调整偏好，而不是把领域适应和偏好优化混在一起。如果使用 Base 作 reference，policy 和 reference 的能力差异更大，margin 会混入不必要的领域分布偏移。

### 10.5 DPO 训练流程

1. 读取 8 对偏好数据并切分 6/2；
2. 加载 Domain SFT reference；
3. 无梯度预计算 train/validation 的 reference chosen/rejected 分数；
4. 释放 reference 显存；
5. 重新加载 Domain SFT 作为可训练 policy；
6. 计算 response-only log-prob；
7. 计算 DPO loss；
8. 每条样本执行 AdamW 更新；
9. 记录 loss 和 reference-adjusted preference accuracy；
10. 保存 domain_dpo adapter。

配置和结果：

~~~text
beta = 0.1
learning rate = 5e-6
epochs = 3
loss: 0.6770 -> 0.4134
train preference accuracy: 83.33% -> 100%
validation preference accuracy: 100%
~~~

验证集只有 2 对，所以 100% 只能解释为 2/2，不能解释为稳定的偏好对齐能力。

## 11. GRPO 原理和实现

### 11.1 组内相对优化

同一个 prompt 采样多个回答：

~~~text
x -> y_1, y_2, y_3, y_4
~~~

得到 reward：

~~~text
r_1, r_2, r_3, r_4
~~~

组内 advantage：

~~~text
mean_G = mean(r_i)
std_G = std(r_i)
A_i = (r_i - mean_G) / (std_G + epsilon)
~~~

如果 std_G <= epsilon，代码返回全零 advantage，因为同组回答没有相对差异。

### 11.2 clipped objective

~~~text
ratio_i = exp(log pi_theta(y_i|x) - log pi_old(y_i|x))
clipped_ratio_i = clamp(ratio_i, 1-epsilon, 1+epsilon)
L = -mean(min(ratio_i*A_i, clipped_ratio_i*A_i))
~~~

clipping 防止一次策略更新幅度过大。当前实现是 PPO-style objective 的最小版本。

### 11.3 当前 smoke 配置

~~~text
policy: Domain DPO adapter
group size: 4
max new tokens: 8
temperature: 0.8
top_p: 0.95
steps: 4
learning rate: 1e-6
~~~

四个可验证乘法任务：

~~~text
17 * 23 -> 391
19 * 17 -> 323
37 * 26 -> 962
48 * 27 -> 1296
~~~

reward 不能简单写成 expected in response，否则回答 14 会被误判为 4。当前代码提取回答中的数字，并比较最后一个数字：

~~~python
numbers = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", response.strip())
return 1.0 if numbers and numbers[-1] == expected else 0.0
~~~

这仍是最小 verifier，不是完整的数学解析器。

### 11.4 实际 rollout 结果

| step | responses 概况 | rewards | reward std | 非零 advantage |
|---:|---|---|---:|---:|
| 1 | 全部生成 401 | [0,0,0,0] | 0 | 0 |
| 2 | 全部生成 323 | [1,1,1,1] | 0 | 0 |
| 3 | 全部生成 962 | [1,1,1,1] | 0 | 0 |
| 4 | 三个 1368，一个 1296 | [0,0,0,1] | 0.4330 | 4 |

第 4 组：

~~~text
mean reward = 0.25
advantages = [-0.5773, -0.5773, -0.5773, 1.7320]
loss = 0.004792
~~~

前三组 zero variance，按设计没有有效组内信号；第 4 组有正确/错误回答混合，才产生更新信号。

### 11.5 GRPO 的明确边界

当前是单卡教学型 smoke experiment，缺少：

- KL penalty 或 reference policy 约束；
- vLLM 等高吞吐 rollout engine；
- 分布式 rollout worker；
- 长序列高效 log-prob；
- 完整 token-level response mask；
- 多轮 rollout、更多 steps 和多 seed；
- 大规模 reward model 或人工评价。

当前 response_mask 是 [GROUP_SIZE, 1] 的全 1 mask，reward 是 response-level scalar。它验证了从 rollout 到更新的链路，不应被称为完整的大规模 GRPO trainer。

## 12. 评测原理和结果

### 12.1 为什么比较四个模型

只看 Domain DPO 无法判断变化来自哪个阶段，所以评测：

~~~text
Base / General SFT / Domain SFT / Domain DPO
~~~

推理使用 greedy decoding，不采样。结果写入 artifacts/evaluation/comparison.json，包含 task accuracy 和 confusion matrix。

### 12.2 实际结果

| 模型 | 情感 | 主题 |
|---|---:|---:|
| Base | 6/6 = 100% | 4/6 = 66.7% |
| General SFT | 6/6 = 100% | 4/6 = 66.7% |
| Domain SFT | 6/6 = 100% | 4/6 = 66.7% |
| Domain DPO | 6/6 = 100% | 4/6 = 66.7% |

准确结论是：

> 在当前 12 条闭集评测和 greedy decoding 设置下，没有观察到四个阶段之间的准确率差异。情感任务存在 ceiling effect，主题任务存在标签边界问题；当前结果足以验证评测流程，但不足以支持领域泛化结论。

训练 loss 和分类准确率不一致的原因可能是：

- Base 已经能解决简单情感题；
- 测试集太小，准确率粒度太粗；
- 训练和测试分布不完全一致；
- teacher-forcing token loss 和最终 greedy label 是不同指标；
- 领域标签边界仍不清晰。

## 13. FastAPI 推理服务

api_server.py 提供：

~~~text
GET  /health
POST /v1/chat/completions
~~~

启动时加载 Qwen base 和 domain_dpo adapter，拼接 system/user/assistant 消息，调用 model.generate，返回 OpenAI 风格的 choices 和 usage。

usage 记录 prompt token、completion token、total token 和生成耗时。接口可用不代表模型已经达到生产质量，项目曾观察到通用问题上的泛化拒答，这是后训练退化的一个 bad case。

## 14. 测试覆盖

当前测试结果：13/13 通过。

覆盖内容：

- JSON 数据格式和字段；
- 固定 seed 的 train/validation 切分；
- padding label mask；
- DPO loss 的方向；
- reference-adjusted preference accuracy；
- GRPO 组内标准化；
- zero variance 处理；
- response mask；
- clipped objective；
- reward 不把 14 误判成 4；
- General SFT -> Domain SFT -> DPO 的路径关系；
- ablation 配置和最佳验证 loss 选择。

## 15. 高频面试问题和标准回答

### Q1：为什么用 LoRA？

> 基座权重冻结，只训练低秩矩阵，减少显存和优化器状态，适合单卡反复做算法实验。代价是适配器容量有限，仍然可能过拟合，不能把结果等同于全参数微调。

### Q2：SFT 和 DPO 的区别？

> SFT 学习一个正向示范，目标是提高示范 response 的 likelihood；DPO 学习 chosen 相对 rejected 的偏好排序，并用 reference-adjusted margin 约束 policy。SFT 是正例监督，DPO 是偏好对的相对优化。

### Q3：为什么 DPO reference 使用 Domain SFT？

> policy 也从 Domain SFT 开始，reference 使用相同初始化能让 DPO 主要调整偏好，而不是把领域适应和偏好优化混在一起。reference 冻结，policy 训练。

### Q4：DPO 为什么只计算 response log-prob？

> prompt 对 chosen 和 rejected 相同，真正需要比较的是回答。只计算 response 可以避免 prompt 长度和内容污染偏好 margin。

### Q5：GRPO 为什么要同一个 prompt 采样多个回答？

> GRPO 使用组内相对 reward。同一 prompt 下比较可以减少不同 prompt 难度差异，多个回答才能计算 mean、std 和相对 advantage。

### Q6：如果 reward 全部相同怎么办？

> std 为零，项目返回零 advantage，因为组内没有学习排序信号。实际运行前三组就是这样，第 4 组 [0,0,0,1] 才产生非零 advantage。

### Q7：你的 GRPO 是完整实现吗？

> 不是大规模完整实现。项目已经实际运行真实 rollout、reward、组内 advantage、clipped policy loss、LoRA 更新和 checkpoint，但还没有 KL 约束、分布式 rollout、vLLM、高效长序列训练、多 seed 和大规模评测。我把它定位为单卡 smoke experiment。

### Q8：训练 loss 下降为什么准确率不变？

> loss 是 teacher-forcing 下的 token-level 指标，准确率是 greedy decoding 后的离散指标。当前测试集很小，情感任务有 ceiling effect，主题任务只有 6 条，所以不能从 loss 下降直接推出泛化提升。

### Q9：项目最大的局限是什么？

> 数据和测试集太小，没有多 seed；DPO 验证集只有 2 对；GRPO 是单卡 smoke；SFT 当前只做 padding mask，还没有 completion-only mask；评测没有公开 benchmark 级别的统计支撑。

### Q10：如果继续做，首先改什么？

> 先加入 completion-only SFT labels 和对应单测，再扩大独立测试集、增加 macro-F1 和多 seed；然后增加 GRPO prompt、group size、KL 约束和训练前后 pass rate，最后做公开论文或 benchmark 的严格复现。

## 16. 面试时不要过度包装

不要说：

- “我复现了完整 GRPO。”
- “DPO 让模型性能提升了 100%。”
- “验证集 100% 说明偏好对齐很好。”
- “Domain SFT 一定提升了泛化。”
- “这是大规模 benchmark。”

建议说：

- “我实现并运行了一个单卡 GRPO smoke experiment。”
- “DPO 训练 preference accuracy 从 83.33% 到 100%，但验证集只有 2 对。”
- “训练 loss 下降，但当前 12 条闭集评测没有观察到准确率差异。”
- “项目验证了后训练链路，也暴露了小数据、标签边界和 reward variance 问题。”

## 17. 项目能证明什么、不能证明什么

项目能够证明：

1. 能在有限显存上搭建多阶段后训练流程；
2. 理解 LoRA、SFT、DPO 和 GRPO 的基本机制；
3. 能自己实现核心张量计算而不是只调用黑盒接口；
4. 会设计 baseline、固定随机种子、记录 metrics 和做消融；
5. 能分析训练指标和泛化指标不一致；
6. 能把代码整理成可测试、可复现的仓库。

项目不能证明：

1. 已经掌握大规模分布式训练；
2. 已完成工业级 RLHF/GRPO；
3. 模型在公开 benchmark 上有竞争力；
4. 可以从小数据直接提出普适科学结论。

## 18. 推荐后续路线

### 第一优先级：completion-only SFT

将 system/user labels 设为 -100，只保留 assistant response labels，并测试 assistant 起点、截断和 padding 边界。

### 第二优先级：扩大评测

每类至少 100 条独立测试样本，增加 macro-F1、JSON schema exact match、hard negative、去重和错误案例分析。

### 第三优先级：多随机种子

使用 42、123、3407，报告均值、标准差和置信区间，而不是单次结果。

### 第四优先级：扩大 GRPO

增加 20-50 个可验证 prompt，每个 prompt 采样 8-16 个回答，比较 group size、reward variance、KL penalty 和训练前后 pass rate。

### 第五优先级：论文复现

选择公开后训练或 GRPO 论文，逐项写出目标函数、代码对应关系、单卡简化部分、无法复现的原因和额外消融。

## 19. 最终定位

最准确的项目定位是：

> 一个面向中文领域任务的研究型大模型后训练实验平台。从 Qwen2.5-1.5B-Instruct 出发，实现了 LoRA SFT、领域继续训练、response-only DPO、可测试的 GRPO 组内 advantage，以及单卡真实 rollout smoke experiment。项目重点是后训练机制理解、实验可复现性和失败分析，而不是声称已经完成大规模 RL 或取得公开 benchmark 领先结果。

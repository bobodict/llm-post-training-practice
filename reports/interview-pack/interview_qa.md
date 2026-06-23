# 面试拷问 Q&A

## Q1: 这个项目为什么匹配 B站大模型后训练的 JD？

**回答**：
这个项目直接覆盖 JD 的核心要求——SFT、DPO、GRPO 三个后训练方法。我用 LLaMA-Factory 完成了 SFT（通用指令微调 + B站领域数据微调），手动实现了 DPO 算法（偏好对齐），并精读了 EasyR1 中 GRPO 的 advantage 计算代码。三个方法从数据需求、训练方式、显存消耗三个维度形成了完整的理解框架。

面试官可以追问的深度：
- 你实际跑通了哪些？→ SFT 完整跑通，DPO 手动实现并验证梯度，GRPO 完成代码精读+笔记
- 为什么不跑完整的 GRPO？→ GRPO 需要多卡（7B LoRA 需 2×32GB），本地 8GB 无法支撑，但代码逻辑已理清

## Q2: 你在 Windows + 8GB 显存上遇到了什么问题？怎么解决的？

**回答**：
核心问题是 HuggingFace Trainer 在 Windows 上调用 trainer.train() 时直接 segfault（139），不管是 Trainer 还是 SFTTrainer 都会崩溃。排查过程：先确认模型加载正常 → 发现手动 PyTorch 训练步骤正常 → 定位到 accelerate 库与 Windows 的兼容性问题 → 放弃 Trainer，手动实现完整训练循环。

解决方案：单样本前向传播 → loss / GRAD_ACCUM → backward → 梯度累积到指定步数 → optimizer.step()。关键优化：tokenize 时 pad_to_max_length 控制在 256-512，batch_size=1，gradient_accumulation_steps=4，每 50 步手动 gc.collect() 和 empty_cache()。最终 VRAM 峰值仅 3.4GB，在 8GB 显卡上稳定运行。

这个排查过程体现了 GPU 内存管理、PyTorch 自动求导机制和训练 loop 的工程理解。

## Q3: 你的 DPO 实现是怎么做的？遇到了什么问题？

**回答**：
DPO 的核心创新是用偏好数据直接优化策略模型，不需要单独训练奖励模型。我按照论文公式分两步实现：

Phase 1：用 base model（参考模型）预计算每对偏好数据的 log probability，参考模型不参与训练，torch.no_grad() 下计算后保存到内存。

Phase 2：用 SFT 后的模型作为策略模型，对相同数据计算 log prob（需要梯度），然后套 DPO loss：-log(σ(β * (log(π_chosen/π_ref_chosen) - log(π_rejected/π_ref_rejected))))。

遇到的核心问题：PeftModel.from_pretrained() 加载的 LoRA 权重默认 requires_grad=False，导致 loss.backward() 时报 "element 0 of tensors does not require grad"。解决方案：手动遍历 `named_parameters()` 将含 `lora` 的参数设为 `requires_grad=True`，optimizer 用 `filter(lambda p: p.requires_grad, ...)` 确保只优化可训练参数。

训练结果：10 epoch 后 chosen-rejected log-prob 差距从 2.1 扩大到 5.1，DPO loss 从 0.687 降至 0.562。但评测发现 DPO 模型在 B站 情感分析任务上反而退化为 0% 准确率——因为 8 对数据太少、10 轮过多导致过拟合，模型变得过于保守。这是一个真实的工程教训。

## Q4: 你为什么选择 Qwen2.5-1.5B 而不是更大或更小的模型？

**回答**：
三个考虑：1）显存约束，1.5B fp16 仅 3GB，LoRA 额外开销 ~0.4GB，8GB 显存绰绰有余；2）中文能力，Qwen2.5 在中文指令理解和生成上表现好，适合 B站 中文场景；3）训练速度，500 样本 × 3 epoch 仅 7.5 分钟，快速迭代实验。

工业界会用更大的模型（7B-72B），但我们的实验目的是验证后训练方法本身——SFT 的数据构建、DPO 的偏好优化、GRPO 的 advantage 计算——这些方法在大模型上的原理相同，在小模型上跑通足以说明理解程度。

## Q5: 如果让你在 B站 真实的业务场景中做后训练，你会怎么设计？

**回答**：
第一步：数据构建。B站 有海量的弹幕、评论、用户行为数据。我会从高质量 UGC 内容中提取 instruction 数据（如视频分类、内容审核、弹幕情感分析、个性化推荐文案生成），这会比通用数据集更有业务价值。

第二步：SFT 基线。用高质量的 B站 场景数据做 SFT，建立基线模型，重点评测业务指标（如审核准确率、推荐点击率）。

第三步：DPO/RLHF 对齐。从用户反馈中构造偏好对（如哪些回复被点赞、哪些被举报），用 DPO 或 GRPO 做偏好对齐。GRPO 更适合 B站 因为可以用弹幕互动量作为天然的 reward 信号。

第四步：工程部署。用 vLLM 做推理加速，Ascend NPU 做国产化部署（B站 有腾讯云/华为云合作）。

## Q6: 如果没有完整跑完 GRPO 实验，你怎么在面试中讲清楚？

**回答**：
GRPO 我虽然没有在本地跑完整训练，但我精读了 EasyR1 的核心代码，可以讲清楚 5 步流程：Rollout（vLLM 生成 n 个 response）→ Reward（打分函数）→ Advantage（组内 Z-score: (score-mean)/std）→ PPO Loss（加权优化）→ Checkpoint。核心创新是省掉了 Critic 模型（省 50% 显存），用组内相对比较替代绝对价值评估。

如果继续做，我会用 EasyR1 的 LoRA + GRPO 示例（examples/qwen3_4b_math_grpo_lora.sh）在租用的云 GPU 上跑通完整实验，验证组内归一化的 advantage 是否比 DPO 的偏好对比更稳定。

## Q7: 你做的 B站-SFT 和 DPO 效果反而比 Base 差，你怎么看？

**回答**：
这恰恰是后训练中非常真实的挑战。Base 模型 80% 的情感分析准确率说明 Qwen2.5-1.5B 本身有不错的 zero-shot 能力。我的 B站-SFT 只用了 36 条领域数据（远少于 Base 的预训练数据量），出现了灾难性遗忘——模型过度适应小数据集，丢失了通用的推理能力。

DPO 效果更差的原因：1）8 对偏好数据量太小，DPO 论文推荐至少几百对；2）10 轮训练在 8 对数据上必然过拟合；3）参考模型用了 base 而非 SFT 模型，chosen/rejected 的 log-prob 差距不够大。

改进方向：1）扩充 B站 数据集到 500+ 条；2）用 SFT 模型作为 DPO 参考模型；3）早停 + 验证集监控过拟合；4）混合通用数据防止遗忘。这个问题如果有机会在 B站 实习，是每天都会面对的——我会带着这个教训去实习。

## Q8: 你了解 Ascend NPU 吗？为什么准备了这个？

**回答**：
B站的基础模型组在做国产化部署，JD 中也提到了推理优化。我阅读了 self-llm 项目中 Ascend NPU（昇腾 910B/C）的部署教程，了解了三种推理框架：MindIE（华为官方）、vLLM-ascend（社区适配）、sglang-ascend。核心区别在于：MindIE 需要专属 Docker 镜像和 NPU 驱动，但性能最优；vLLM-ascend 接口与 CUDA 版 vLLM 基本一致，迁移成本低。

性能优化方面，了解了 CPU 高性能模式（cpupower +3% 吞吐）和透明大页（稳定吞吐率）、W8A8 量化（msModelSlim）等 Ascend 特有的优化手段。这体现了对国产 AI 基础设施的关注，也是 B站 工程落地的重要方向。

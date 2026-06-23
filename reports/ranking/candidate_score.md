# 候选项目排序

- JD 来源：`reports\jd.txt`
- 主项目推荐：`LLaMA-Factory`，score=88.46
- 备选项目：`self-llm (开源大模型食用指南)`，score=79.81
- 分数说明：先计算 raw_score，再按 max_raw_score 归一化到 0-100。

| Rank | Name | Score | Raw | Max Raw | License | Stars | Last Commit | Runnable | Resources | Matched | Risks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LLaMA-Factory | 88.46 | 92 | 104 | Apache 2.0 | 45000 | 2026-06-15 | Web UI 一键启动，支持 QLoRA 在单张 24GB 消费级显卡上微调 7B 模型；有完整中文文档和 Docker 支持 | 本地单卡 RTX 3060/4060 (12GB) 可跑 QLoRA 微调 1.5B-3B 模型；CPU 可做推理演示 | SFT, LoRA微调, LLaMA-Factory框架, 大模型训练, 监督微调, DPO | 不直接覆盖 GRPO/DAPO 等 RL 算法，需配合 EasyR1 学习; 消费级显卡只能跑小模型 LoRA，与生产环境有差距; B站弹幕数据需要自行爬取和清洗 |
| 2 | self-llm (开源大模型食用指南) | 79.81 | 83 | 104 | Apache 2.0 | 12000 | 2026-06-20 | 所有教程基于 Linux 平台，面向初学者，有详细步骤和截图；已本地 clone | CPU 可做推理，单卡 GPU 可做微调；有 AutoDL 云 GPU 教程适配 | B站Index-1.9B模型, FastAPI部署, LoRA微调, LangChain接入, Ascend NPU部署, vLLM推理 | 本质是教程集合而非可魔改的代码项目，改造以补写教程为主; JD 要求的 GRPO/DAPO/蒸馏等 RL 算法不在此项目范围; 作为教程仓库，简历上需要包装为'为开源项目贡献教程+自研实验' |
| 3 | EasyR1 | 65.38 | 68 | 104 | Apache 2.0 | 5000 | 2026-06-10 | 提供 Docker 镜像和 shell 脚本一键启动训练；但实际训练需要多卡 GPU | GRPO LoRA 7B 需 2×32GB；GRPO 全量 7B 需 8×40GB；本地消费级显卡几乎无法运行完整训练 | GRPO, DAPO, 强化学习训练, RL后训练, EasyR1框架, 奖励函数设计 | GPU 要求高：GRPO LoRA 7B 需 2×32GB，本地几乎无法跑完整训练; 代码复杂：基于 veRL 的分布式架构，调试门槛高; 初学者可能只能做代码阅读+概念理解，无法跑完整实验 |

## 使用说明

- 这个脚本只根据显式字段打分；语义判断、JD 命中度和最终选择仍需 AI 助手/人工审阅。
- 不可运行、资源要求过高、风险说明过多的项目，除非非常贴 JD，否则不建议作为主项目。
- 推荐项目应尽快进入最小路径摸底、简历 4-5 行版本和面试 Q&A，而不是卡在完美复现。

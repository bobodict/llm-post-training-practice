# Technical Presentation Outline

请生成一份 6-8 页中文技术汇报，主题为“中文场景大模型后训练实践”，重点展示研究问题、算法实现、实验边界和后续计划，不使用求职广告式表达。

建议结构：

1. 研究问题：小规模领域数据如何影响任务适应与通用能力。
2. 实验路线：Base → General SFT → Domain SFT → Domain DPO。
3. SFT 实现：LoRA、padding mask、梯度累积、验证 loss。
4. DPO 实现：response-only log-prob、SFT reference、preference margin。
5. GRPO 核心：组内 reward 标准化、零方差保护、response mask。
6. 评测设计：固定 seed、greedy decoding、准确率与混淆矩阵。
7. 局限性：小数据、无大规模 RL、尚未做多卡和多次重复实验。
8. 后续计划：扩大数据、独立测试集、多 seed、GRPO rollout 和系统性能分析。

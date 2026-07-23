# Research Presentation Checklist

| 项目 | 当前状态 |
| --- | --- |
| 研究问题 | 小规模领域后训练中的适应与泛化权衡 |
| SFT | LoRA、验证 loss、padding mask、梯度累积 |
| DPO | response-only log-prob、SFT reference、验证 preference accuracy |
| GRPO | CPU 核心单测 + 单卡真实 rollout smoke、可验证 reward、clipped policy update、checkpoint |
| 评测 | greedy decoding、任务准确率、混淆矩阵、JSON 落盘 |
| 部署 | FastAPI 本地推理接口 |
| 数据边界 | 人工构造的小型演示语料，不冒充公开 benchmark |
| 诚实表述 | 明确区分已运行、已实现和后续计划 |

## 申请实验室前应能回答

- 为什么 padding label mask 会改变 loss 的解释？
- DPO 的 reference 应该怎样选择，为什么？
- GRPO 的组内标准化解决了什么问题？零方差如何处理？
- 当前实验为什么不能支持泛化结论？下一步如何补足？

# Research-Oriented Post-Training Practice

## Goal

将仓库从面向特定岗位的后训练练习，整理为可公开展示、可复现、适合实验室申请的中文场景大模型后训练实践。

## Scope

- 删除公司、平台和岗位导向的公开叙事，统一使用“中文场景”与“领域数据”。
- 将数据放入仓库自己的 `data/` 目录，并让脚本可从仓库根目录运行。
- 改进 SFT 数据处理：固定随机种子、训练/验证划分、attention mask、padding label mask。
- 改进 DPO：明确 reference model、使用 response token mask、记录训练曲线和验证指标。
- 改进评测：固定解码、严格分类标签、输出 JSON 结果和混淆矩阵。
- 增加 GRPO advantage 的最小可测试实现，明确它不等同于大规模 RL 训练。
- 重写 README、SETUP 和面试材料，区分“实际完成”“最小实现”和“后续计划”。

## Architecture

训练脚本继续采用 `Transformers + PEFT + PyTorch`，不引入大型新框架。仓库根目录作为工作目录，数据统一位于 `data/`，模型和适配器统一位于可配置的 `artifacts/` 下。评测脚本使用确定性生成并将结构化结果写入 `artifacts/evaluation/`，便于复查。

## Success Criteria

1. 仓库搜索不到旧平台名称或岗位专属表述。
2. 静态检查通过，JSON 数据可解析，脚本可从仓库根目录导入。
3. 不依赖缺失的外部目录即可完成数据准备和算法单元测试。
4. 文档不再声称已经完成 GRPO 大规模训练，也不把未提交的结果包装成可验证事实。
5. 代码能够清楚展示 SFT、DPO、评测和 GRPO advantage 四个独立技术点。

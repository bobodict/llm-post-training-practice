# Chinese Domain LLM Post-Training Practice

一个面向中文短文本任务的大模型后训练实验仓库。项目使用 Qwen2.5-1.5B 和 LoRA，在有限显存环境下研究 SFT、DPO、确定性评测以及 GRPO 优势估计的核心实现。

## 研究问题

小规模领域数据进行后训练时，模型的任务适应能力、通用能力保持和偏好对齐效果之间如何平衡？本项目围绕以下对照展开：

```text
Base model -> General SFT -> Domain SFT -> Domain DPO
```

## 已实现内容

- **LoRA SFT**：手写 PyTorch 训练循环，支持梯度累积、验证集 loss 和 padding label mask。
- **中文领域数据**：包含情感判断、主题分类、摘要和短文本生成等任务，数据文件位于 `data/`。
- **DPO**：手写 response-only log-prob、reference-adjusted margin 和 DPO loss，记录训练集/验证集 preference accuracy。
- **评测**：固定随机种子，使用 greedy decoding，输出任务准确率、混淆矩阵和 JSON 结果。
- **GRPO 核心**：实现可单元测试的组内 reward 标准化，不宣称完成大规模分布式 RL 训练。
- **部署**：提供 FastAPI + OpenAI 风格接口的 LoRA 推理服务。

## 项目状态

| 模块 | 状态 |
| --- | --- |
| 通用中文数据 SFT | 代码完成，需本地模型权重后运行 |
| 中文领域 SFT | 代码完成，需本地模型权重后运行 |
| DPO | 最小可运行实现，需先完成通用 SFT |
| GRPO | advantage 核心实现与 CPU 单元测试完成 |
| 7B 以上模型、多卡 RL | 未包含 |

仓库不提交模型权重和训练产物。实际训练后，checkpoint 写入 `artifacts/checkpoints/`，指标写入 `artifacts/metrics/`，评测结果写入 `artifacts/evaluation/`。

## 实际运行记录

当前环境已实际跑通 General SFT、Domain SFT、Domain DPO 和四模型评测。训练指标与限制分析见 [reports/experiment_results.md](reports/experiment_results.md)。本次 12 条闭集评测中，四个模型情感判断均为 `6/6`，主题分类均为 `4/6`；该结果没有显示后训练带来可观测提升。

## 目录结构

```text
data/
├── general_zh_demo.json       # 通用中文指令数据
├── domain_zh.json             # 中文短文本领域数据
└── dpo_domain_demo.json       # 偏好对数据
scripts/
├── config.py                  # 路径、随机种子、数据工具
├── train_windows.py           # 通用中文 SFT
├── train_domain.py            # 领域 SFT
├── train_dpo.py               # 手写 DPO
├── dpo_core.py                # DPO 数学核心
├── grpo_core.py               # GRPO advantage 核心
├── eval_models.py             # 确定性评测
└── api_server.py              # FastAPI 推理服务
tests/
└── test_data_and_algorithms.py
reports/interview-pack/        # 项目讲解材料
```

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

准备 Qwen2.5-1.5B-Instruct 权重，并设置环境变量：

```powershell
$env:MODEL_PATH = "D:\models\Qwen2.5-1.5B-Instruct"
```

从仓库根目录运行：

```bash
python -m scripts.train_windows
python -m scripts.train_domain
python -m scripts.train_dpo
python -m scripts.eval_models
python -m scripts.api_server
```

默认配置面向单卡 8GB 显存。若要用于更严谨的研究结论，应继续扩充数据、增加独立测试集，并报告多次运行的均值和方差。

## 技术栈

Python 3.11、PyTorch、Transformers、PEFT、FastAPI。训练代码参考 LLaMA-Factory 的数据和模型生态，但核心实验逻辑由本仓库独立实现。

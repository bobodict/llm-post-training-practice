# 环境配置指南

## 硬件要求
- NVIDIA GPU，≥8GB VRAM（RTX 3060/4060 即可）
- 推荐 16GB+ 内存

## 1. 安装依赖

```bash
# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# LLaMA-Factory（核心框架）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 本项目脚本依赖
pip install peft transformers datasets fastapi uvicorn
```

## 2. 下载模型

```bash
# 从 ModelScope 下载（国内快）
cd LLaMA-Factory
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')"
```

## 3. 运行脚本

```bash
# 将本项目 scripts/ 下的 .py 和 .json 文件复制到 LLaMA-Factory 目录
cp scripts/*.py scripts/*.json LLaMA-Factory/

# 基线 SFT 训练（500 样本, 3 epoch, ~7.5 分钟）
cd LLaMA-Factory
python train_windows.py

# B站领域数据 SFT
python train_bilibili.py

# DPO 偏好对齐
python train_dpo.py

# 4 模型对比评测
python eval_models.py

# API 推理服务（启动后访问 http://localhost:8000）
python api_server.py
```

## 4. GRPO 代码精读（可选）

```bash
git clone https://github.com/hiyouga/EasyR1.git
# 核心文件: verl/trainer/core_algos.py (compute_grpo_outcome_advantage)
```

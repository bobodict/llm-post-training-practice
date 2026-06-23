# LLaMA-Factory 项目摸底报告

## 基本信息

| 字段 | 值 |
| --- | --- |
| repo_path | D:\shushu-projects\LLaMA-Factory |
| generated_at | 2026-06-23T07:01:18.298385+00:00 |
| file_count_scanned | 553 |
| approx_total_bytes | 13472122 |

## 语言和文件类型

| 语言 | 文件数 |
| --- | --- |
| Python | 291 |
| Other | 156 |
| YAML | 100 |
| Shell | 4 |
| TOML | 1 |
| JavaScript | 1 |

## 依赖和环境线索

- `Makefile`
- `docker/docker-cuda/Dockerfile`
- `docker/docker-cuda/docker-compose.yml`
- `docker/docker-npu/Dockerfile`
- `docker/docker-npu/docker-compose.yml`
- `docker/docker-rocm/Dockerfile`
- `docker/docker-rocm/docker-compose.yml`
- `docs/Makefile`
- `docs/requirements.txt`
- `pyproject.toml`

## README

- `README.md`
- `README_zh.md`
- `data/README.md`
- `data/README_zh.md`
- `docker/docker-cuda/README.md`
- `examples/README.md`
- `examples/README_zh.md`

## 核心链路线索

| 类别 | 命中文件数 | 代表路径 |
| --- | --- | --- |
| api_backend | 9 | assets/sponsors/serpapi.svg<br>scripts/api_example/test_image.py<br>scripts/api_example/test_toolcall.py<br>src/api.py<br>src/llamafactory/api/__init__.py<br>src/llamafactory/api/app.py<br>src/llamafactory/api/chat.py<br>src/llamafactory/api/common.py |
| async_jobs | 1 | src/llamafactory/v1/plugins/trainer_plugins/lr_scheduler.py |
| config | 40 | .github/ISSUE_TEMPLATE/config.yml<br>.pre-commit-config.yaml<br>data/v1_dpo_demo.yaml<br>data/v1_sft_demo.yaml<br>examples/accelerate/fsdp2_config.yaml<br>examples/accelerate/fsdp2_config_qwen35.yaml<br>examples/accelerate/fsdp2_config_qwen35_moe.yaml<br>examples/accelerate/fsdp_config.yaml |
| data_pipeline | 7 | data/dataset_info.json<br>examples/train_lora/qwen3_preprocess.yaml<br>src/llamafactory/data/loader.py<br>src/llamafactory/model/loader.py<br>src/llamafactory/v1/plugins/data_plugins/loader.py<br>tests/data/test_loader.py<br>tests_v1/core/test_model_loader.py |
| database_state | 40 | data/identity.json<br>docs/en/dev-guide/core/model-engine.md<br>docs/en/dev-guide/plugins/model-plugins/initialization.md<br>docs/en/dev-guide/plugins/model-plugins/kernels.md<br>docs/en/dev-guide/plugins/model-plugins/rendering.md<br>docs/en/hyperparameters/model-argument.md<br>docs/zh/dev-guide/core/model-engine.md<br>docs/zh/dev-guide/plugins/model-plugins/initialization.md |
| devops_deploy | 28 | .dockerignore<br>.github/workflows/docker.yml<br>.github/workflows/docs.yml<br>.github/workflows/label_issue.yml<br>.github/workflows/publish.yml<br>.github/workflows/tests.yml<br>.github/workflows/tests_cuda.yml<br>.github/workflows/tests_npu.yml |
| evaluation | 40 | .github/workflows/tests.yml<br>.github/workflows/tests_cuda.yml<br>.github/workflows/tests_npu.yml<br>examples/extras/nlg_eval/llama3_lora_predict.yaml<br>requirements/metrics.txt<br>scripts/api_example/test_image.py<br>scripts/api_example/test_toolcall.py<br>scripts/eval_bleu_rouge.py |
| frontend_mobile | 9 | src/llamafactory/webui/components/__init__.py<br>src/llamafactory/webui/components/chatbot.py<br>src/llamafactory/webui/components/data.py<br>src/llamafactory/webui/components/eval.py<br>src/llamafactory/webui/components/export.py<br>src/llamafactory/webui/components/footer.py<br>src/llamafactory/webui/components/infer.py<br>src/llamafactory/webui/components/top.py |
| inference_demo | 40 | assets/sponsors/serpapi.svg<br>data/alpaca_en_demo.json<br>data/alpaca_zh_demo.json<br>data/c4_demo.jsonl<br>data/dpo_en_demo.json<br>data/dpo_zh_demo.json<br>data/glaive_toolcall_en_demo.json<br>data/glaive_toolcall_zh_demo.json |
| model | 40 | docs/en/dev-guide/core/model-engine.md<br>docs/en/dev-guide/plugins/model-plugins/initialization.md<br>docs/en/dev-guide/plugins/model-plugins/kernels.md<br>docs/en/dev-guide/plugins/model-plugins/rendering.md<br>docs/en/hyperparameters/model-argument.md<br>docs/zh/dev-guide/core/model-engine.md<br>docs/zh/dev-guide/plugins/model-plugins/initialization.md<br>docs/zh/dev-guide/plugins/model-plugins/kernels.md |
| testing_quality | 40 | .github/workflows/tests.yml<br>.github/workflows/tests_cuda.yml<br>.github/workflows/tests_npu.yml<br>scripts/api_example/test_image.py<br>scripts/api_example/test_toolcall.py<br>src/llamafactory/train/test_utils.py<br>src/llamafactory/v1/utils/pytest.py<br>tests/check_license.py |
| training | 40 | docs/en/dev-guide/core/trainer.md<br>docs/en/hyperparameters/training-argument.md<br>docs/en/training/dpo.md<br>docs/en/training/sft.md<br>docs/zh/dev-guide/core/trainer.md<br>docs/zh/hyperparameters/training-argument.md<br>docs/zh/training/dpo.md<br>docs/zh/training/sft.md |

## Notebook / Docker / Test 线索

### Notebooks
- 无

### Docker
- `.dockerignore`
- `.github/workflows/docker.yml`
- `docker/docker-cuda/Dockerfile`
- `docker/docker-cuda/Dockerfile.base`
- `docker/docker-cuda/Dockerfile.megatron`
- `docker/docker-cuda/README.md`
- `docker/docker-cuda/docker-compose.yml`
- `docker/docker-npu/Dockerfile`
- `docker/docker-npu/docker-compose.yml`
- `docker/docker-rocm/Dockerfile`
- `docker/docker-rocm/docker-compose.yml`

### Tests
- `.github/workflows/tests.yml`
- `.github/workflows/tests_cuda.yml`
- `.github/workflows/tests_npu.yml`
- `scripts/api_example/test_image.py`
- `scripts/api_example/test_toolcall.py`
- `src/llamafactory/train/test_utils.py`
- `src/llamafactory/v1/utils/pytest.py`
- `tests/check_license.py`
- `tests/conftest.py`
- `tests/data/processor/test_feedback.py`
- `tests/data/processor/test_pairwise.py`
- `tests/data/processor/test_processor_utils.py`
- `tests/data/processor/test_supervised.py`
- `tests/data/processor/test_unsupervised.py`
- `tests/data/test_collator.py`
- `tests/data/test_converter.py`
- `tests/data/test_formatter.py`
- `tests/data/test_loader.py`
- `tests/data/test_mm_plugin.py`
- `tests/data/test_template.py`
- `tests/e2e/test_chat.py`
- `tests/e2e/test_sglang.py`
- `tests/e2e/test_train.py`
- `tests/eval/test_eval_template.py`
- `tests/model/model_utils/test_add_tokens.py`
- `tests/model/model_utils/test_attention.py`
- `tests/model/model_utils/test_checkpointing.py`
- `tests/model/model_utils/test_embedding.py`
- `tests/model/model_utils/test_misc.py`
- `tests/model/model_utils/test_packing.py`

## 潜在数据/状态/模型/资源路径

- `assets/logo.png`
- `assets/sponsors/serpapi.svg`
- `assets/sponsors/warp.jpg`
- `assets/thirdparty/colab.svg`
- `assets/thirdparty/discord.svg`
- `assets/thirdparty/dsw.svg`
- `assets/thirdparty/lab4ai.svg`
- `assets/thirdparty/online.svg`
- `data/README.md`
- `data/README_zh.md`
- `data/alpaca_en_demo.json`
- `data/alpaca_zh_demo.json`
- `data/c4_demo.jsonl`
- `data/dataset_info.json`
- `data/dpo_en_demo.json`
- `data/dpo_zh_demo.json`
- `data/glaive_toolcall_en_demo.json`
- `data/glaive_toolcall_zh_demo.json`
- `data/identity.json`
- `data/kto_en_demo.json`
- `data/mllm_audio_demo.json`
- `data/mllm_demo.json`
- `data/mllm_demo_data/1.jpg`
- `data/mllm_demo_data/1.mp3`
- `data/mllm_demo_data/1.mp4`
- `data/mllm_demo_data/2.avi`
- `data/mllm_demo_data/2.jpg`
- `data/mllm_demo_data/2.wav`
- `data/mllm_demo_data/3.flac`
- `data/mllm_demo_data/3.jpg`

## 目录树摘要

```text
LLaMA-Factory/
  .ai/
  .claude/
  .github/
  assets/
  data/
  docker/
  docs/
  examples/
  requirements/
  scripts/
  src/
  tests/
  tests_v1/
  .dockerignore
  .env.local
  .gitattributes
  .gitignore
  .pre-commit-config.yaml
  CITATION.cff
  CLAUDE.md
  LICENSE
  MANIFEST.in
  Makefile
  README.md
  README_zh.md
  pyproject.toml
    CLAUDE.md
    skills/
      llamafactory-sft/
    ISSUE_TEMPLATE/
    workflows/
    CODE_OF_CONDUCT.md
    CONTRIBUTING.md
    PULL_REQUEST_TEMPLATE.md
    SECURITY.md
    copilot-instructions.md
    instructions-v0.md
    instructions-v1.md
      1-bug-report.yml
      2-feature-request.yml
      config.yml
      docker.yml
      docs.yml
      label_issue.yml
      publish.yml
      tests.yml
      tests_cuda.yml
      tests_npu.yml
    sponsors/
    thirdparty/
    logo.png
      serpapi.svg
      warp.jpg
      colab.svg
      discord.svg
      dsw.svg
      lab4ai.svg
      online.svg
    mllm_demo_data/
    README.md
    README_zh.md
    alpaca_en_demo.json
    alpaca_zh_demo.json
    c4_demo.jsonl
    dataset_info.json
    dpo_en_demo.json
    dpo_zh_demo.json
    glaive_toolcall_en_demo.json
    glaive_toolcall_zh_demo.json
    identity.json
    kto_en_demo.json
    mllm_audio_demo.json
    mllm_demo.json
    mllm_video_audio_demo.json
    mllm_video_demo.json
    reason_tool_use_demo_50.jsonl
    v1_dpo_demo.jsonl
    v1_dpo_demo.yaml
    v1_sft_demo.jsonl
    v1_sft_demo.yaml
    wiki_demo.txt
      1.jpg
      1.mp3
      1.mp4
      2.avi
      2.jpg
      2.wav
      3.flac
      3.jpg
      3.mp4
      4.mp3
      4.mp4
    docker-cuda/
    docker-npu/
    docker-rocm/
      Dockerfile
      Dockerfile.base
      Dockerfile.megatron
      README.md
      docker-compose.yml
      Dockerfile
      docker-compose.yml
      Dockerfile
      docker-compose.yml
    _static/
    en/
    zh/
    Makefile
    conf.py
    make.bat
    requirements.txt
      css/
      js/
      advanced/
      data-preparation/
      dev-guide/
      hyperparameters/
      inference/
      training/
      conf.py
      getting-started.md
      index.rst
      installation.md
      llamaboard-web-ui.md
      advanced/
      data-preparation/
      dev-guide/
      hyperparameters/
      inference/
      training/
      conf.py
      getting-started.md
      index.rst
      installation.md
      llamaboard-web-ui.md
    accelerate/
    ascend/
    deepspeed/
    extras/
    inference/
    ktransformers/
    megatron/
    merge_lora/
    train_full/
    train_lora/
    train_qlora/
    v1/
    README.md
    README_zh.md
      fsdp2_config.yaml
      fsdp2_config_qwen35.yaml
      fsdp2_config_qwen35_moe.yaml
      fsdp_config.yaml
      fsdp_config_multiple_nodes.yaml
      fsdp_config_offload.yaml
      qwen3_5_full_sft_fsdp2.yaml
      qwen3_5moe_lora_sft_fsdp2.yaml
      qwen3_full_sft_fsdp2.yaml
      qwen3moe_full_sft_fsdp.yaml
      qwen3vlmoe_full_sft_fsdp2.yaml
      qwen3vlmoe_lora_sft_fsdp.yaml
      ds_z0_config.json
      ds_z2_autotp_config.json
      ds_z2_config.json
      ds_z2_offload_config.json
      ds_z3_config.json
      ds_z3_fp8_config.json
      ds_z3_offload_config.json
      adam_mini/
      apollo/
      asft/
      badam/
      dft/
      eaft/
      fp8/
      fsdp_qlora/
      galore/
      llama_pro/
      loraplus/
      mod/
      multi_tokens/
      muon/
      nlg_eval/
      oft/
      pissa/
      qoft/
      qwen3.yaml
      qwen3_full_sft.yaml
      qwen3_lora_sft.yaml
      qwen3vl.yaml
      accelerate/
      train_lora/
      qwen2_vl_full.yaml
      qwen3_moe_full.yaml
      qwen3_full_sft.yaml
      qwen3_gptq.yaml
      qwen3_lora_sft.yaml
      qwen3vl_lora_sft.yaml
      qwen3_full_sft.yaml
      qwen3vl_full_sft.yaml
      qwen3_lora_dpo.yaml
      qwen3_lora_kto.yaml
      qwen3_lora_pretrain.yaml
      qwen3_lora_reward.yaml
      qwen3_lora_sft.sh
      qwen3_lora_sft.yaml
      qwen3_lora_sft_ds3.yaml
      qwen3_lora_sft_ray.yaml
      qwen3_preprocess.yaml
      qwen3vl_lora_dpo.yaml
      qwen3vl_lora_sft.yaml
      llama3_lora_sft_aqlm.yaml
      llama3_lora_sft_awq.yaml
      llama3_lora_sft_gptq.yaml
      qwen3_lora_sft_bnb_npu.yaml
      qwen3_lora_sft_otfq.yaml
      train_batching_strategy/
      train_freeze/
      train_full/
      train_lora/
      train_qlora/
    adam-mini.txt
    apollo.txt
    aqlm.txt
    badam.txt
    bitsandbytes.txt
    deepspeed.txt
    dev.txt
    eetq.txt
    fp8-te.txt
    fp8.txt
    galore.txt
    gptq.txt
    hqq.txt
    ktransformers.txt
    liger-kernel.txt
    metrics.txt
    minicpm-v.txt
    npu.txt
    openmind.txt
    sglang.txt
    swanlab.txt
    triton_ascend.txt
    vllm.txt
    api_example/
    convert_ckpt/
    stat_utils/
    bench_qwen.py
    dcp2hf.py
```

## 下一步人工确认

- 找到最小可运行命令：API、页面、CLI、worker、测试、训练或 demo 至少一个。
- 确认依赖、环境变量、数据库/数据文件、端口和外部服务。
- 确认 baseline/demo 是否能在本地、Docker、云服务器或 GPU 环境上跑通。
- 确认自己要做的面试亮点：改造点、demo、测试、报告或实验计划。

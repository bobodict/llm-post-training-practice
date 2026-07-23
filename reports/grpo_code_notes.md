# GRPO Advantage Notes

## 1. Core idea

GRPO uses several sampled responses for the same prompt and compares them within a group. It does not train a separate value critic in the basic formulation, but it still requires rollout, reward calculation, policy loss, checkpointing, and usually a distributed runtime for practical experiments.

## 2. This repository's implementation

`scripts/grpo_core.py` implements only the mathematical core:

1. Sum token-level rewards over valid response tokens.
2. Group responses by prompt id.
3. Compute each group's mean and population standard deviation.
4. Normalize scores within the group.
5. Broadcast the normalized advantage over valid response tokens.
6. Return zero advantage for a group with zero reward variance.

The implementation is intentionally independent of a model or GPU so that the algorithm can be tested on CPU.

## 3. Relation to a full training system

```text
prompt -> rollout -> reward -> group advantage -> policy loss -> checkpoint
```

In addition to the CPU-tested core, `scripts/grpo_smoke.py` runs a small single-GPU experiment with a real Qwen rollout, an exact-final-number reward, group-relative advantages, a PPO-style clipped objective, LoRA optimization, metric logging, and checkpoint saving. It is deliberately a smoke experiment: it does not claim a complete multi-GPU GRPO reproduction, KL-regularized production trainer, or benchmark result.

The latest run is recorded in `artifacts/metrics/grpo_smoke.json`. Three groups had zero reward variance and therefore correctly produced zero advantage. The fourth group had rewards `[0, 0, 0, 1]`, producing non-zero advantages and one policy update with loss `0.0048`.

## 4. Questions to discuss

- Why compare responses within the same prompt rather than normalize all rewards together?
- What happens when all responses in a group receive the same reward?
- How does group size trade off reward variance against rollout cost?
- Which failure modes can come from a biased or sparse reward function?

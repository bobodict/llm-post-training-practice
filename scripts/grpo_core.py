"""Minimal GRPO advantage computation, independent of distributed training."""

from collections import defaultdict

import torch


def compute_group_advantages(token_level_rewards, response_mask, group_ids, eps=1e-6):
    """Normalize response scores within each prompt group and mask padding tokens."""
    if token_level_rewards.ndim == 1:
        token_level_rewards = token_level_rewards.unsqueeze(-1)
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError("response_mask must have the same shape as token_level_rewards")
    if len(group_ids) != token_level_rewards.shape[0]:
        raise ValueError("group_ids must contain one id per response")

    mask = response_mask.to(dtype=token_level_rewards.dtype)
    scores = (token_level_rewards * mask).sum(dim=-1)
    grouped = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)

    advantages = torch.zeros_like(scores)
    for indices in grouped.values():
        group_scores = scores[indices]
        mean = group_scores.mean()
        std = group_scores.std(unbiased=False)
        if std <= eps:
            normalized = torch.zeros_like(group_scores)
        else:
            normalized = (group_scores - mean) / (std + eps)
        advantages[indices] = normalized

    return advantages.unsqueeze(-1) * mask


def clipped_grpo_loss(log_probs, old_log_probs, advantages, clip_eps=0.2):
    """Compute the PPO-style clipped objective used by a minimal GRPO update."""
    ratio = torch.exp(log_probs - old_log_probs.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    return -surrogate.mean()

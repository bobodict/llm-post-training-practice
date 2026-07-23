"""Small, framework-independent DPO helpers used by training and tests."""

import torch
import torch.nn.functional as F


def dpo_loss(chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """Compute the mean DPO loss from response-level log probabilities."""
    chosen_margin = chosen_logps - reference_chosen_logps
    rejected_margin = rejected_logps - reference_rejected_logps
    preference_margin = chosen_margin - rejected_margin
    return -F.logsigmoid(beta * preference_margin).mean()


def preference_accuracy(chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps):
    """Return the fraction of examples whose reference-adjusted margin is positive."""
    chosen_margin = chosen_logps - reference_chosen_logps
    rejected_margin = rejected_logps - reference_rejected_logps
    return float((chosen_margin > rejected_margin).float().mean().item())

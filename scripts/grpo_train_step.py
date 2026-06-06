import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Literal
from scripts.grpo import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    compute_rollout_rewards, 
    compute_group_normalized_rewards, 
    compute_policy_gradient_loss,
    aggregate_loss_across_microbatch,
)

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    only_good: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    original_len_rollout_responses = len(rollout_responses)
    if loss_normalization == "constant" and normalization_constant is None:
        normalization_constant = original_len_rollout_responses
    # compute rewards
    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    # compute advantages
    advantages, advantage_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer, only_good)
    keep = advantages != 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=device), {
            "reward_mean": reward_metadata["reward_mean"],
            "format_reward_mean": reward_metadata["format_reward_mean"],
            "answer_reward_mean": reward_metadata["answer_reward_mean"],
            "loss": 0.0,
            "grad_norm": 0.0,
            "token_entropy": 0.0,
            "clip_fraction": 0.0,
        }

    kept = keep.nonzero(as_tuple=True)[0]
    advantages = advantages[kept]
    if old_log_probs is not None:
        old_log_probs = old_log_probs[kept].to(device)
    repeated_prompts = [repeated_prompts[i] for i in kept.tolist()]
    rollout_responses = [rollout_responses[i] for i in kept.tolist()]
    len_rollout_responses = len(rollout_responses)
    # tokenize
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]
    advantages = advantages.to(device)
    microbatch_size = original_len_rollout_responses // gradient_accumulation_steps
    total_loss = 0.0
    total_entropy = 0.0
    total_clip_frac = 0.0
    # gradient accumulation
    for i in range(0, len_rollout_responses, microbatch_size):
        inputs_microbatch = input_ids[i:i+microbatch_size].to(device)
        labels_microbatch = labels[i:i+microbatch_size].to(device)
        mask_microbatch = response_mask[i:i+microbatch_size].to(device)
        advantages_microbatch = advantages[i:i+microbatch_size].to(device)
        old_log_probs_microbatch = (old_log_probs[i:i+microbatch_size] if old_log_probs is not None else None)
        response_log_probs = get_response_log_probs(model, inputs_microbatch, labels_microbatch, return_token_entropy=True)
        token_entropy = response_log_probs["token_entropy"]
        # compute the policy gradient loss
        policy_gradient_loss, pg_metadata = compute_policy_gradient_loss(advantages_microbatch, response_log_probs["log_probs"], importance_reweighting_method, old_log_probs_microbatch, cliprange, mask_microbatch)
        microbatch_loss = aggregate_loss_across_microbatch(policy_gradient_loss, mask_microbatch, loss_normalization, normalization_constant)

        if loss_normalization == "sequence": # average
            loss = microbatch_loss * len(inputs_microbatch) / original_len_rollout_responses
        elif loss_normalization == "constant":
            loss = microbatch_loss
        loss.backward()
        total_loss += float(loss.item())
        mask_sum = mask_microbatch.sum()
        if mask_sum > 0:
            total_entropy += float(((token_entropy * mask_microbatch).sum() / mask_sum).item()) * len(inputs_microbatch) / len_rollout_responses
        if "clip_fraction" in pg_metadata:
            total_clip_frac += float(pg_metadata["clip_fraction"].item()) * len(inputs_microbatch) / len_rollout_responses

    # clip
    if max_grad_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
    else:
        grad_norm = float(torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), ord=2))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata = {
        "reward_mean": reward_metadata["reward_mean"],
        "format_reward_mean": reward_metadata["format_reward_mean"],
        "answer_reward_mean": reward_metadata["answer_reward_mean"],
        "advantage_mean": advantage_metadata["advantage_mean"],
        "advantage_std": advantage_metadata["advantage_std"],
        "raw_reward_std": advantage_metadata["raw_reward_std"],
        "loss": total_loss,
        "grad_norm": grad_norm,
        "token_entropy": total_entropy,
        "clip_fraction": total_clip_frac,
    }

    log_loss = torch.tensor(total_loss, device=device, dtype=torch.float32)
    return log_loss, metadata

import torch
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel
import torch.nn.functional as F
from typing import Callable
from typing import Literal

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    token_list = []
    mask_list = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tok = tokenizer(prompt, add_special_tokens=False)["input_ids"] # tokenize prompt
        output_tok = tokenizer(output, add_special_tokens=False)["input_ids"] # tokenize output
        token = prompt_tok + output_tok # concatenate
        token_list.append(token)
        mask = [int(j >= len(prompt_tok)) for j in range(1, len(token))]
        mask_list.append(mask)

    new_len = max(len(t) for t in token_list) - 1
    input_ids = torch.full((len(prompt_strs), new_len), tokenizer.pad_token_id)
    labels = torch.full((len(prompt_strs), new_len), tokenizer.pad_token_id)
    response_mask = torch.zeros((len(prompt_strs), new_len))

    # populate lists
    for i, token in enumerate(token_list):
        cur_len = min(len(token), new_len)
        input_ids[i, :cur_len] = torch.tensor(token[:cur_len])
        labels[i, :len(token) - 1] = torch.tensor(token[1:])
        response_mask[i, :len(token) - 1] = torch.tensor(mask_list[i])

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    logits = model(input_ids, use_cache=False).logits
    log_probs_all = F.log_softmax(logits, dim=-1) # log probabilities
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        with torch.no_grad():
            token_entropy = -(F.softmax(logits, dim=-1) * log_probs_all).sum(dim=-1) # token entropy
        return {"log_probs": log_probs, "token_entropy": token_entropy}
    return {"log_probs": log_probs}

def compute_rollout_rewards(reward_fn: Callable[[str, str], dict[str, float]], rollout_responses: list[str], repeated_ground_truths: list[str]) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    format_rewards = []
    answer_rewards = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        info = reward_fn(response, truth) # get information
        rewards.append(info["reward"])
        format_rewards.append(info["format_reward"])
        answer_rewards.append(info["answer_reward"])
    rewards = torch.tensor(rewards)
    metadata = {
        "reward_mean": float(sum(rewards) / len(rewards)),
        "format_reward_mean": float(sum(format_rewards) / len(format_rewards)),
        "answer_reward_mean": float(sum(answer_rewards) / len(answer_rewards)),
    }
    return rewards, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor, 
    group_size: int, 
    baseline: Literal["mean", "none"] = "mean", 
    advantage_eps: float = 1e-6, 
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    only_good: bool = False,
):
    groups = raw_rewards.reshape(-1, group_size)
    if baseline == "mean": # subtract mean
        advantages = groups - groups.mean(dim=1, keepdim=True)
    elif baseline == "none": # no baseline
        advantages = groups

    if advantage_normalizer == "std": # divide by std
        advantages = advantages / (groups.std(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean": # divide by mean
        advantages = advantages / (groups.mean(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "none": # no normalization
        pass
    if only_good: # only keep positive advantages
        advantages = advantages.clamp(min=0)
    advantages = advantages.reshape(-1)
    metadata = {
        "raw_reward_mean": float(raw_rewards.mean()),
        "raw_reward_std": float(raw_rewards.std()),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
    }
    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = raw_rewards_or_advantages
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1) # dimension stuff

    if importance_reweighting_method == "none":
        loss = -advantages * policy_log_probs # no importance reweighting
        return loss, {}

    ratio = torch.exp(policy_log_probs - old_log_probs)
    if importance_reweighting_method == "noclip": # no clipping
        loss = -ratio * advantages
        metadata = {
            "importance_ratio_mean": ratio.mean(),
            "importance_ratio_max": ratio.max(),
            "importance_ratio_min": ratio.min(),
        }
    elif importance_reweighting_method == "grpo": # PPO/GRPO-style
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
        metadata = {
            "importance_ratio_mean": ratio.mean(),
            "importance_ratio_max": ratio.max(),
            "importance_ratio_min": ratio.min(),
            "clip_fraction": ratio.ne(clipped_ratio).float().mean(),
        }
    elif importance_reweighting_method == "gspo":
        mask = response_mask.to(policy_log_probs.dtype)
        log_diff = policy_log_probs - old_log_probs
        new_ratio = torch.exp((log_diff * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True))
        clipped_ratio = torch.clamp(new_ratio, 1.0 - cliprange, 1.0 + cliprange)
        loss = -torch.minimum(new_ratio * advantages, clipped_ratio * advantages)
        loss = loss.expand_as(policy_log_probs) # dimension stuff
        metadata = {
            "importance_ratio_mean": new_ratio.mean(),
            "importance_ratio_max": new_ratio.max(),
            "importance_ratio_min": new_ratio.min(),
            "clip_fraction": (new_ratio != clipped_ratio).float().mean(),
        }

    return loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    mask = mask.to(per_token_policy_gradient_loss.dtype)
    if loss_normalization == "sequence":
        # average loss over each sequence and average over sequences
        loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / mask.sum(dim=1)
        loss = loss.mean()
    elif loss_normalization == "constant": # just divide
        loss = (per_token_policy_gradient_loss * mask).sum() / normalization_constant
    return loss
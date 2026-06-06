import argparse
import json
import random
from pathlib import Path
import torch
import wandb
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from scripts.grpo_train_step import grpo_train_step
from scripts.grpo import tokenize_prompt_and_output, get_response_log_probs


def get_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def evaluate(vllm_server, val_data, template, batch_size, max_tokens):
    prompts = [template.format(question=ex["question"]) for ex in val_data]
    gts = [ex["answer"].split("####")[-1].strip() for ex in val_data]
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": 0,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    completions = vllm_server.generate_completions(prompts, sampling_params, batch_size=batch_size)
    rewards = []
    format_rewards = []
    lengths = []
    # compute rewards
    for completion, gt in zip(completions, gts):
        info = r1_zero_reward_fn(completion.text, gt)
        rewards.append(info["reward"])
        format_rewards.append(info["format_reward"])
        lengths.append(len(completion.token_ids))
    return {
        "val_reward_mean": sum(rewards) / len(rewards),
        "val_format_reward_mean": sum(format_rewards) / len(format_rewards),
        "val_avg_response_len": sum(lengths) / len(lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--prompt-path", default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--train-path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_max_tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--baseline", default="mean")
    parser.add_argument("--advantage-normalizer", default="std")
    parser.add_argument("--loss-normalization", default="sequence")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--importance-reweighting-method", default="none")
    parser.add_argument("--cliprange", type=float, default=0.1)
    parser.add_argument("--only-good", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompt_text = Path(args.prompt_path).stem
    wandb_run_name = args.run_name if args.run_name is not None else f"grpo-{prompt_text}-lr{args.learning_rate:g}-seed{args.seed}"
    wandb.init(
        project="cs336-a5",
        name=wandb_run_name,
        id=wandb.util.generate_id(),
        config=vars(args),
        resume="never",
    )

    # get data
    train_data = get_data(args.train_path)[: args.n_train_examples]
    val_data = get_data(args.val_path)[: args.n_val_examples]
    
    template = Path(args.prompt_path).read_text()
    model, tokenizer = get_model_and_tokenizer(args.model_name, "cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)

    vllm_server = VLLMServer(model_id=args.model_name, gpu=1, seed=args.seed)
    vllm_server.start()
    vllm_server.init_weight_sync(policy_device="cuda:0")

    n_prompts_per_batch = args.rollout_batch_size // args.group_size
    try:
        # train!!
        for step in range(args.num_rollout_steps):
            batch_examples = random.sample(train_data, n_prompts_per_batch) # sample batch
            prompts = [template.format(question=ex["question"]) for ex in batch_examples]
            ground_truths = [ex["answer"].split("####")[-1].strip() for ex in batch_examples]
            vllm_server.sync_policy_weights(model)

            train_sampling_params = {
                "temperature": args.sampling_temperature,
                "top_p": 1.0,
                "max_tokens": args.sampling_max_tokens,
                "n": args.group_size,
                "seed": args.seed + step,
                "stop": ["</answer>"],
                "include_stop_str_in_output": True,
            }
            # generate rollouts
            completions = vllm_server.generate_completions(prompts, train_sampling_params, batch_size=n_prompts_per_batch) # generate rollouts
            rollout_responses = [completion.text for completion in completions]
            repeated_prompts = [p for p in prompts for _ in range(args.group_size)]
            repeated_ground_truths = [gt for gt in ground_truths for _ in range(args.group_size)]
            step_old_log_prob_chunks = None
            # compute log probs
            if args.importance_reweighting_method != "none":
                step_old_log_prob_chunks = []
                for start in range(0, len(rollout_responses), args.train_batch_size):
                    tokenized = tokenize_prompt_and_output(
                        repeated_prompts[start:start + args.train_batch_size],
                        rollout_responses[start:start + args.train_batch_size],
                        tokenizer,
                    )
                    with torch.no_grad():
                        step_old_log_prob_chunks.append(get_response_log_probs(model, tokenized["input_ids"].to("cuda:0"), tokenized["labels"].to("cuda:0"))["log_probs"].cpu()) # get log probs

            train_info = {}
            # chunk due to error
            for chunk_idx, start in enumerate(range(0, len(rollout_responses), args.train_batch_size)):
                chunk_prompts = repeated_prompts[start:start + args.train_batch_size]
                chunk_responses = rollout_responses[start:start + args.train_batch_size]
                chunk_ground_truths = repeated_ground_truths[start:start + args.train_batch_size]
                chunk_old_log_probs = step_old_log_prob_chunks[chunk_idx] if step_old_log_prob_chunks is not None else None

                _, train_info = grpo_train_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm,
                    reward_fn=r1_zero_reward_fn,
                    repeated_prompts=chunk_prompts,
                    rollout_responses=chunk_responses,
                    repeated_ground_truths=chunk_ground_truths,
                    group_size=args.group_size,
                    importance_reweighting_method=args.importance_reweighting_method,
                    old_log_probs=chunk_old_log_probs,
                    cliprange=args.cliprange,
                    baseline=args.baseline,
                    advantage_normalizer=args.advantage_normalizer,
                    loss_normalization=args.loss_normalization,
                    only_good=args.only_good,
                )

            logs = {"step": step, **train_info}
            # evaluate
            if step % 10 == 0:
                vllm_server.sync_policy_weights(model)
                val_info = evaluate(vllm_server, val_data, template, args.rollout_batch_size, args.sampling_max_tokens)
                logs.update(val_info)

            # add info to wandb
            if step % 40 == 0:
                add_info = wandb.Table(columns=["prompt", "response", "ground_truth"])
                for x, y, z in zip(repeated_prompts[:8], rollout_responses[:8], repeated_ground_truths[:8]):
                    add_info.add_data(x, y, z)
                logs["train_rollouts"] = add_info

            # log and clean up
            wandb.log(logs, step=step)
            print(step, logs, flush=True)
            del completions, rollout_responses, step_old_log_prob_chunks
            torch.cuda.empty_cache()

        model.save_pretrained(f"outputs/grpo_{args.seed}")
        tokenizer.save_pretrained(f"outputs/grpo_{args.seed}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
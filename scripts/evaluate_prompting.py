import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# assign a category to the reward
def categories(format_reward, correctness_reward):
    if format_reward == 1 and correctness_reward == 1:
        return "format1_and_correctness1"
    if format_reward == 1 and correctness_reward == 0:
        return "format1_correctness0"
    return "format0_correctness0"

def evaluate_prompt(server, dataset, prompt_name, prompt_path, output_dir, batch_size):
    template = Path(prompt_path).read_text()
    prompts = []
    for ex in dataset:
        # format the prompt with the question
        prompts.append(template.format(question=ex["question"]))

    # set the sampling parameters
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 0,
    }

    if prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True
        reward_fn = r1_zero_reward_fn
    else:
        # use the question only reward function
        reward_fn = question_only_reward_fn

    completions = server.generate_completions(prompts, sampling_params, batch_size)
    counts = Counter()
    store_by_cat = defaultdict(list)
    rows = []
    for i, (ex, prompt, completion) in enumerate(zip(dataset, prompts, completions)):
        response = completion.text
        right_answer = ex["answer"].split("####")[-1].strip()
        out = reward_fn(response, right_answer) # get reward
        category = categories(out["format_reward"], out["answer_reward"])
        counts[category] += 1
        row = {
            "idx": i,
            "question": ex["question"],
            "right_answer": right_answer,
            "prompt": prompt,
            "response": response,
            "finish_reason": completion.finish_reason,
            "format_reward": out["format_reward"],
            "answer_reward": out["answer_reward"],
            "total_reward": out["reward"],
            "category": category,
        }
        rows.append(row)
        if len(store_by_cat[category]) < 15: store_by_cat[category].append(row)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/f"{prompt_name}_results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with open(output_dir / f"{prompt_name}_examples.json", "w") as f:
        json.dump(store_by_cat, f, indent=2)

    return {
        "prompt_name": prompt_name,
        "num_examples": len(dataset),
        "accuracy": counts["format1_and_correctness1"] / len(dataset),
        "counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/prompting_baselines")
    args = parser.parse_args()

    data_path = Path(f"data/gsm8k/{args.split}.jsonl")
    with open(data_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    output_dir = Path(args.output_dir)
    prompt_paths = {
        "question_only": "cs336_alignment/prompts/question_only.prompt",
        "r1_zero": "cs336_alignment/prompts/r1_zero.prompt",
        "r1_zero_three_shot": "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
    }
    # start the vllm server
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=args.gpu, seed=0)
    server.start()
    summary = []
    for prompt_name, prompt_path in prompt_paths.items():
        print(prompt_name)
        # evaluate the prompt
        metrics = evaluate_prompt(server, dataset, prompt_name, prompt_path, output_dir, args.batch_size)
        summary.append(metrics)
        print(json.dumps(metrics, indent=2))

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
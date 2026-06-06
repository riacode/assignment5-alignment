from __future__ import annotations

import argparse
import runpy
import sys
from typing import NamedTuple

from cs336_alignment.modal_utils import GPU, MAX_CONTAINERS, app, image, wandb_secret
R1_ZERO_PROMPT = "cs336_alignment/prompts/r1_zero.prompt"
QUESTION_ONLY_PROMPT = "cs336_alignment/prompts/question_only.prompt"
R1_ZERO_THREE_SHOT_PROMPT = "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"

VARIATIONS = {
    "GRPO_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
    },
    "Dr_GRPO": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
    },
    "RFT": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
    },
    "MaxRL": {
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
    },
    "OnlyGood_GRPO": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
    },
}

OFFPOLICY_VARIATIONS = {
    "offpolicy_naive": {
        "importance_reweighting_method": "none",
    },
    "offpolicy_noclip": {
        "importance_reweighting_method": "noclip",
    },
    "offpolicy_clip": {
        "importance_reweighting_method": "grpo",
    },
    "offpolicy_gspo": {
        "importance_reweighting_method": "gspo",
    },
}

DEFAULT_LEARNING_RATES = (1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4)

class Job(NamedTuple):
    seed: int
    num_rollout_steps: int
    learning_rate: float
    prompt_path: str

class VariantJob(NamedTuple):
    variation: str
    seed: int
    num_rollout_steps: int
    learning_rate: float
    prompt_path: str
    baseline: str
    advantage_normalizer: str
    loss_normalization: str


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]

def build_variant_jobs(seeds, num_rollout_steps, learning_rate, variations, prompt_path):
    jobs = []
    for variation in variations:
        config = VARIATIONS[variation]
        for seed in seeds:
            jobs.append(
                VariantJob(
                    variation=variation,
                    seed=seed,
                    num_rollout_steps=num_rollout_steps,
                    learning_rate=learning_rate,
                    prompt_path=prompt_path,
                    baseline=config["baseline"],
                    advantage_normalizer=config["advantage_normalizer"],
                    loss_normalization=config["loss_normalization"],
                )
            )
    return jobs

def build_lr_sweep_jobs(seeds, num_rollout_steps, learning_rates, prompt_path):
    return [Job(seed, num_rollout_steps, learning_rate, prompt_path)
        for learning_rate in learning_rates
        for seed in seeds
    ]


def build_prompt_ablation_jobs(seeds, num_rollout_steps, learning_rate, include_r1_zero):
    prompt_paths = [QUESTION_ONLY_PROMPT, R1_ZERO_THREE_SHOT_PROMPT]
    if include_r1_zero:
        prompt_paths.append(R1_ZERO_PROMPT)
    return [
        Job(seed, num_rollout_steps, learning_rate, prompt_path)
        for prompt_path in prompt_paths
        for seed in seeds
    ]


def dedupe_jobs(jobs: list[Job]) -> list[Job]:
    seen: set[Job] = set()
    unique: list[Job] = []
    for job in jobs:
        if job not in seen:
            seen.add(job)
            unique.append(job)
    return unique


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 10,
    max_containers=MAX_CONTAINERS,
    secrets=[wandb_secret],
)
def run_grpo_train(seed, num_rollout_steps, learning_rate, prompt_path):
    sys.argv = [
        "scripts/grpo_train.py",
        "--seed", str(seed),
        "--num-rollout-steps", str(num_rollout_steps),
        "--learning-rate", str(learning_rate),
        "--prompt-path", prompt_path,
    ]
    runpy.run_path("scripts/grpo_train.py", run_name="__main__")
    prompt_text = prompt_path.rsplit("/", 1)[-1].removesuffix(".prompt")
    return f"{prompt_text}-lr{learning_rate:g}-seed{seed}"


@app.local_entrypoint()
def main(*argv: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--prompt-path", default=R1_ZERO_PROMPT)
    parser.add_argument("--learning-rates", default="1e-6,3e-6,5e-6,1e-5,3e-5,5e-5,1e-4")
    parser.add_argument("--ablation-learning-rate", type=float, default=1e-5)
    parser.add_argument("--sweep-all", action="store_true")
    parser.add_argument("--sweep-lr", action="store_true", help="Submit learning-rate sweep only.")
    parser.add_argument("--sweep-prompts", action="store_true", help="Submit prompt ablation only.")
    parser.add_argument("--all_variants", action="store_true")
    parser.add_argument("--variations", default="GRPO_constant,Dr_GRPO,RFT,MaxRL")
    parser.add_argument("--all_offpolicy", action="store_true")
    parser.add_argument("--offpolicy-variations", default="offpolicy_naive,offpolicy_noclip,offpolicy_clip,offpolicy_gspo")
    args = parser.parse_args(list(argv))

    seeds = parse_csv_ints(args.seeds)
    sweep_lr = args.sweep_lr or args.sweep_all
    sweep_prompts = args.sweep_prompts or args.sweep_all
    if sweep_lr or sweep_prompts:
        jobs: list[Job] = []
        if sweep_lr:
            learning_rates = parse_csv_floats(args.learning_rates)
            jobs.extend(build_lr_sweep_jobs(seeds=seeds, num_rollout_steps=args.num_rollout_steps, learning_rates=learning_rates))
        if sweep_prompts:
            jobs.extend(build_prompt_ablation_jobs(seeds=seeds, num_rollout_steps=args.num_rollout_steps, learning_rate=args.ablation_learning_rate))
        jobs = dedupe_jobs(jobs)
        starmap_args = [(job.seed, job.num_rollout_steps, job.learning_rate, job.prompt_path) for job in jobs]
        for result in run_grpo_train.starmap(starmap_args):
            print("Done")
        return
    
    if args.all_variants:
        variations = [v.strip() for v in args.variations.split(",")]
        jobs = build_variant_jobs(
            seeds=seeds,
            num_rollout_steps=args.num_rollout_steps,
            learning_rate=args.learning_rate,
            variations=variations,
            prompt_path=args.prompt_path,
        )

        starmap_args = [
            (
                job.variation,
                job.seed,
                job.num_rollout_steps,
                job.learning_rate,
                job.prompt_path,
                job.baseline,
                job.advantage_normalizer,
                job.loss_normalization,
            )
            for job in jobs
        ]

        for result in run_grpo_variant_train.starmap(starmap_args):
            print("Done")
        return

    if args.all_offpolicy:
        variations = [v.strip() for v in args.offpolicy_variations.split(",") if v.strip()]
        starmap_args = [
            (variation, seed, args.num_rollout_steps, args.learning_rate, args.prompt_path, OFFPOLICY_VARIATIONS[variation]["importance_reweighting_method"])
            for variation in variations
            for seed in seeds
        ]

        for result in run_grpo_offpolicy_train.starmap(starmap_args):
            print("Done")
        return

    starmap_args = [(seed, args.num_rollout_steps, args.learning_rate, args.prompt_path) for seed in seeds]
    for result in run_grpo_train.starmap(starmap_args):
        print("Done")

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 10,
    max_containers=MAX_CONTAINERS,
    secrets=[wandb_secret],
)
def run_grpo_variant_train(variation, seed, num_rollout_steps, learning_rate, prompt_path, baseline, advantage_normalizer, loss_normalization):
    sys.argv = [
        "scripts/grpo_train.py",
        "--seed", str(seed),
        "--num-rollout-steps", str(num_rollout_steps),
        "--learning-rate", str(learning_rate),
        "--prompt-path", prompt_path,
        "--baseline", baseline,
        "--advantage-normalizer", advantage_normalizer,
        "--loss-normalization", loss_normalization,
        "--run-name", f"{variation}-seed-{seed}",
    ]
    if variation == "OnlyGood_GRPO":
        sys.argv.append("--only-good")
    runpy.run_path("scripts/grpo_train.py", run_name="__main__")
    return f"{variation}-lr{learning_rate:g}-seed{seed}"

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 10,
    max_containers=MAX_CONTAINERS,
    secrets=[wandb_secret],
)
def run_grpo_offpolicy_train(variation, seed, num_rollout_steps, learning_rate, prompt_path, importance_reweighting_method):
    sys.argv = [
        "scripts/grpo_train.py",
        "--seed", str(seed),
        "--num-rollout-steps", str(num_rollout_steps),
        "--learning-rate", str(learning_rate),
        "--prompt-path", prompt_path,
        "--train-batch-size", "8",
        "--gradient-accumulation-steps", "1",
        "--importance-reweighting-method", importance_reweighting_method,
        "--cliprange", "0.1",
        "--run-name", f"{variation}-seed-{seed}",
    ]

    runpy.run_path("scripts/grpo_train.py", run_name="__main__")
    return f"{variation}-lr{learning_rate:g}-seed{seed}"
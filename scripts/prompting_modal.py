from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from cs336_alignment.modal_utils import GPU, RUN_TIMEOUT_SECONDS, app, image, wandb_secret
OUTPUT_DIR = "outputs/prompting_baselines"

@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    secrets=[wandb_secret],
)
def run_prompting_baselines(split="test", batch_size=32, limit=None):
    argv = [
        "scripts/evaluate_prompting.py",
        "--split", split,
        "--batch-size", str(batch_size),
    ]
    sys.argv = argv
    runpy.run_path("scripts/evaluate_prompting.py", run_name="__main__")
    output_dir = Path(OUTPUT_DIR)
    return {
        path.name: path.read_text()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }

@app.local_entrypoint()
def main(*argv: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(list(argv))
    files = run_prompting_baselines.remote(split=args.split, batch_size=args.batch_size, limit=args.limit)
    local_dir = Path(OUTPUT_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        (local_dir / name).write_text(contents)

    print(f"saved to {local_dir.resolve()}", flush=True)

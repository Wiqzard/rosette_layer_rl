#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run analysis + experiments end-to-end.")
    p.add_argument("--dataset-steps", type=int, default=8000)
    p.add_argument("--episodes", type=int, default=350)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    _run(
        [
            "python",
            str(ROOT / "scripts" / "run_layer_analysis.py"),
            "--dataset-steps",
            str(args.dataset_steps),
            "--grid-size",
            str(args.grid_size),
            "--history",
            str(args.history),
            "--max-steps",
            str(args.max_steps),
            "--out-dir",
            str(args.out_dir),
        ]
    )

    _run(
        [
            "python",
            str(ROOT / "scripts" / "run_experiments.py"),
            "--episodes",
            str(args.episodes),
            "--seeds",
            str(args.seeds),
            "--grid-size",
            str(args.grid_size),
            "--history",
            str(args.history),
            "--max-steps",
            str(args.max_steps),
            "--batch-size",
            str(args.batch_size),
            "--out-dir",
            str(args.out_dir),
        ]
    )


if __name__ == "__main__":
    main()

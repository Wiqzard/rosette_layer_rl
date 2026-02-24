#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_rosetta_layer.analysis import analyze_layers, collect_random_dataset
from rl_rosetta_layer.envs import GridVideoEnv
from rl_rosetta_layer.layers import build_candidate_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer quality analysis for frozen-layer RL.")
    p.add_argument("--dataset-steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = GridVideoEnv(
        grid_size=args.grid_size,
        history=args.history,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    layers = build_candidate_layers(
        obs_dim=env.obs_dim,
        grid_size=args.grid_size,
        history=args.history,
        seed=7,
    )

    dataset = collect_random_dataset(env, steps=args.dataset_steps, seed=args.seed)
    rows = analyze_layers(layers.values(), dataset=dataset, seed=args.seed)

    out_csv = args.out_dir / "layer_analysis.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "reward_r2",
                "transition_r2",
                "effective_rank",
                "composite_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_layer = rows[0]["layer"]
    (args.out_dir / "selected_layer.txt").write_text(f"{best_layer}\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Selected best layer: {best_layer}")
    print("Top scores:")
    for row in rows:
        print(
            f"  {row['layer']:<22} score={row['composite_score']:.4f} "
            f"reward_r2={row['reward_r2']:.4f} transition_r2={row['transition_r2']:.4f}"
        )


if __name__ == "__main__":
    main()

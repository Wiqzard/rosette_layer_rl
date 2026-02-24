#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_rosetta_layer.agent import TrainConfig, train_single_layer
from rl_rosetta_layer.envs import GridVideoEnv
from rl_rosetta_layer.layers import build_candidate_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run layer-comparison RL experiments.")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--episodes", type=int, default=350)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer names, or 'all'.",
    )
    return p.parse_args()


def _read_selected_layer(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text if text else None


def _parse_seeds(seed_text: str) -> List[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def _layer_subset(all_layers: Dict[str, object], layer_arg: str, selected_path: Path) -> List[str]:
    if layer_arg == "all":
        return list(all_layers.keys())

    if layer_arg == "selected":
        selected = _read_selected_layer(selected_path)
        if selected is None:
            raise ValueError(f"{selected_path} not found or empty; run layer analysis first.")
        if selected not in all_layers:
            raise ValueError(f"Selected layer '{selected}' not in candidate layers.")
        return [selected]

    names = [x.strip() for x in layer_arg.split(",") if x.strip()]
    missing = [x for x in names if x not in all_layers]
    if missing:
        raise ValueError(f"Unknown layer(s): {missing}")
    return names


def _summary_stats(rows: List[dict], layer: str) -> dict:
    layer_rows = [r for r in rows if r["layer"] == layer]
    if not layer_rows:
        return {
            "layer": layer,
            "mean_last_50": np.nan,
            "std_last_50": np.nan,
            "mean_auc": np.nan,
        }

    by_seed: Dict[int, List[float]] = {}
    for row in layer_rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row["return"]))

    last50_seed_means = []
    auc_seed_means = []
    for _, returns in by_seed.items():
        arr = np.asarray(returns, dtype=np.float32)
        last50_seed_means.append(float(np.mean(arr[-50:])))
        auc_seed_means.append(float(np.mean(arr)))

    return {
        "layer": layer,
        "mean_last_50": float(np.mean(last50_seed_means)),
        "std_last_50": float(np.std(last50_seed_means)),
        "mean_auc": float(np.mean(auc_seed_means)),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env_probe = GridVideoEnv(
        grid_size=args.grid_size,
        history=args.history,
        max_steps=args.max_steps,
        seed=0,
    )
    all_layers = build_candidate_layers(
        obs_dim=env_probe.obs_dim,
        grid_size=args.grid_size,
        history=args.history,
        seed=7,
    )
    selected_layers = _layer_subset(all_layers, args.layers, args.out_dir / "selected_layer.txt")

    seeds = _parse_seeds(args.seeds)
    config = TrainConfig(
        episodes=args.episodes,
        batch_size=args.batch_size,
    )

    rows = []
    for layer_name in selected_layers:
        layer = all_layers[layer_name]
        print(f"Training layer: {layer_name}")
        for seed in seeds:
            result = train_single_layer(
                layer=layer,
                seed=seed,
                env_kwargs={
                    "grid_size": args.grid_size,
                    "history": args.history,
                    "max_steps": args.max_steps,
                },
                config=config,
            )

            returns = result["episode_returns"]
            for ep, ret in enumerate(returns):
                rows.append(
                    {
                        "layer": layer_name,
                        "seed": int(seed),
                        "episode": int(ep),
                        "return": float(ret),
                    }
                )
            print(
                f"  seed={seed:<3d} mean_last_50={float(np.mean(returns[-50:])):.4f} "
                f"mean_all={float(np.mean(returns)):.4f}"
            )

    returns_csv = args.out_dir / "experiment_returns.csv"
    with returns_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "seed", "episode", "return"])
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [_summary_stats(rows, layer_name) for layer_name in selected_layers]
    summary_rows.sort(key=lambda r: r["mean_last_50"], reverse=True)

    summary_csv = args.out_dir / "experiment_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "mean_last_50", "std_last_50", "mean_auc"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    best = summary_rows[0]["layer"] if summary_rows else "N/A"
    print(f"Wrote: {returns_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Best by mean_last_50: {best}")


if __name__ == "__main__":
    main()

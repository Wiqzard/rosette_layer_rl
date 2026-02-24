# Frozen-Layer RL Prototype

This repository contains a runnable prototype to test the hypothesis:

> A single frozen layer from a pretrained video model, wrapped with tiny pre/post adapters,
> can outperform other layer choices on RL tasks.

The current implementation is a **numpy-only benchmark** (no torch/gym dependency) so it runs in minimal environments.
It includes:

- Candidate frozen layers (`random`, `structured`, `bottleneck`)
- Layer selection analysis (reward predictivity, transition predictivity, effective rank)
- RL training comparison across layers
- CSV outputs for paper figures/tables
- A dummy paper template (`paper/dummy_paper.md`)

## Run Order (Python Commands)

Run these in order:

```bash
# 1) Layer analysis (chooses best candidate layer)
python scripts/run_layer_analysis.py --dataset-steps 8000 --seed 0

# 2) RL experiments across candidate layers
python scripts/run_experiments.py --episodes 350 --seeds 0,1,2 --batch-size 64

# 3) Export results table for paper markdown
python scripts/export_markdown_tables.py
```

Optional single command for steps 1+2:

```bash
python scripts/run_all.py --dataset-steps 8000 --episodes 350 --seeds 0,1,2
python scripts/export_markdown_tables.py
```

## Key Outputs

- `outputs/layer_analysis.csv`
- `outputs/selected_layer.txt`
- `outputs/experiment_returns.csv`
- `outputs/experiment_summary.csv`
- `outputs/experiment_summary_table.md`

## Project Layout

- `src/rl_rosetta_layer/envs.py`: video-like grid RL environment
- `src/rl_rosetta_layer/layers.py`: frozen layer definitions
- `src/rl_rosetta_layer/analysis.py`: layer quality scoring
- `src/rl_rosetta_layer/agent.py`: tiny adapters + Q-learning loop
- `scripts/run_layer_analysis.py`: choose best candidate layer
- `scripts/run_experiments.py`: train/evaluate all layers
- `paper/dummy_paper.md`: paper skeleton with placeholders

## Extending to Real Video Models

To use CogVideo/HunyuanVideo layers:

1. Replace or augment `build_candidate_layers()` with wrappers around real model layers.
2. Keep the same interface (`forward`, `backward`, `input_dim`, `output_dim`).
3. Re-run `run_layer_analysis.py` to pick a promising layer before full RL training.

The scoring code is model-agnostic and designed to stay unchanged.

## TODO List

- [ ] Replace synthetic layers in `src/rl_rosetta_layer/layers.py` with wrappers around CogVideo/HunyuanVideo layers.
- [ ] Define exact candidate layer indices/module names for each backbone.
- [ ] Re-run `scripts/run_layer_analysis.py` for each backbone and save outputs per model.
- [ ] Re-run `scripts/run_experiments.py` with fixed seeds and identical budgets.
- [ ] Add at least one stronger baseline (full small MLP and/or random frozen layer bank).
- [ ] Add statistical tests (bootstrap CIs or paired seed-level tests).
- [ ] Copy `outputs/layer_analysis.csv` + `outputs/experiment_summary_table.md` into `paper/dummy_paper.md`.
- [ ] Fill citation placeholders in Related Work.
- [ ] Replace placeholder claims/numbers in Abstract, Results, and Conclusion.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .envs import GridVideoEnv
from .layers import FrozenLayer


@dataclass
class TransitionDataset:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    state: np.ndarray
    next_state: np.ndarray
    dones: np.ndarray
    num_actions: int


def collect_random_dataset(env: GridVideoEnv, steps: int = 8000, seed: int = 0) -> TransitionDataset:
    rng = np.random.default_rng(seed)

    obs_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    rewards: List[float] = []
    actions: List[int] = []
    states: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    dones: List[float] = []

    obs, info = env.reset(seed=seed)
    state = info.state
    for _ in range(steps):
        action = int(rng.integers(0, env.num_actions))
        next_obs, reward, done, next_info = env.step(action)

        obs_list.append(obs)
        next_obs_list.append(next_obs)
        rewards.append(float(reward))
        actions.append(action)
        states.append(state)
        next_states.append(next_info.state)
        dones.append(1.0 if done else 0.0)

        if done:
            obs, info = env.reset()
            state = info.state
        else:
            obs = next_obs
            state = next_info.state

    return TransitionDataset(
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_obs=np.asarray(next_obs_list, dtype=np.float32),
        state=np.asarray(states, dtype=np.float32),
        next_state=np.asarray(next_states, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        num_actions=env.num_actions,
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    xtx = x_train.T @ x_train
    dim = xtx.shape[0]
    w = np.linalg.solve(xtx + reg * np.eye(dim, dtype=np.float32), x_train.T @ y_train)
    return x_test @ w


def _effective_rank(x: np.ndarray) -> float:
    x_center = x - np.mean(x, axis=0, keepdims=True)
    cov = (x_center.T @ x_center) / max(1, x_center.shape[0] - 1)
    s = np.linalg.svd(cov, compute_uv=False)
    total = float(np.sum(s))
    if total <= 1e-12:
        return 0.0
    p = np.clip(s / total, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p))
    rank = float(np.exp(entropy))
    return rank / float(x.shape[1])


def _normalize_r2(score: float) -> float:
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def score_layer(
    layer: FrozenLayer,
    dataset: TransitionDataset,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = dataset.obs.shape[0]
    perm = rng.permutation(n)
    split = int(0.8 * n)
    idx_train = perm[:split]
    idx_test = perm[split:]

    phi = layer.forward(dataset.obs)
    phi_next = layer.forward(dataset.next_obs)

    phi_train = phi[idx_train]
    phi_test = phi[idx_test]

    # Reward linear probe.
    y_reward_train = dataset.rewards[idx_train][:, None]
    y_reward_test = dataset.rewards[idx_test][:, None]
    reward_pred = _ridge_fit_predict(phi_train, y_reward_train, phi_test)
    reward_r2 = _r2(y_reward_test, reward_pred)

    # Transition probe: predict next layer feature from current layer feature + action.
    act_onehot = np.eye(dataset.num_actions, dtype=np.float32)[dataset.actions]
    trans_x = np.concatenate([phi, act_onehot], axis=1)
    trans_x_train = trans_x[idx_train]
    trans_x_test = trans_x[idx_test]
    trans_y_train = phi_next[idx_train]
    trans_y_test = phi_next[idx_test]

    trans_pred = _ridge_fit_predict(trans_x_train, trans_y_train, trans_x_test)
    transition_r2 = _r2(trans_y_test, trans_pred)

    rank_score = _effective_rank(phi_train)

    composite = (
        0.45 * _normalize_r2(reward_r2)
        + 0.45 * _normalize_r2(transition_r2)
        + 0.10 * rank_score
    )

    return {
        "layer": layer.name,
        "reward_r2": float(reward_r2),
        "transition_r2": float(transition_r2),
        "effective_rank": float(rank_score),
        "composite_score": float(composite),
    }


def analyze_layers(
    layers: Iterable[FrozenLayer],
    dataset: TransitionDataset,
    seed: int = 0,
) -> List[Dict[str, float]]:
    rows = [score_layer(layer, dataset, seed=seed) for layer in layers]
    rows.sort(key=lambda row: row["composite_score"], reverse=True)
    return rows

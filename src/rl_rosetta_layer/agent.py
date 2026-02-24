from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .envs import GridVideoEnv
from .layers import FrozenLayer


class TinyInputAdapter:
    def __init__(self, dim: int, rng: np.random.Generator, init_scale: float = 0.01) -> None:
        self.W = (np.eye(dim, dtype=np.float32) + init_scale * rng.standard_normal((dim, dim))).astype(
            np.float32
        )
        self.b = np.zeros((dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.W + self.b).astype(np.float32)

    def backward(self, x: np.ndarray, grad_out: np.ndarray, lr: float) -> np.ndarray:
        pre = x @ self.W + self.b
        y = np.tanh(pre)
        grad_pre = grad_out * (1.0 - y * y)

        bsz = x.shape[0]
        grad_W = (x.T @ grad_pre) / bsz
        grad_b = np.mean(grad_pre, axis=0)
        grad_x = grad_pre @ self.W.T

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_x.astype(np.float32)


class TinyOutputAdapter:
    def __init__(self, latent_dim: int, num_actions: int, rng: np.random.Generator) -> None:
        self.W = (0.05 * rng.standard_normal((latent_dim, num_actions))).astype(np.float32)
        self.b = np.zeros((num_actions,), dtype=np.float32)

    def forward(self, z: np.ndarray) -> np.ndarray:
        return (z @ self.W + self.b).astype(np.float32)

    def backward(self, z: np.ndarray, grad_q: np.ndarray, lr: float) -> np.ndarray:
        bsz = z.shape[0]
        grad_W = (z.T @ grad_q) / bsz
        grad_b = np.mean(grad_q, axis=0)
        grad_z = grad_q @ self.W.T

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_z.astype(np.float32)


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int = 20000) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        idx = rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


@dataclass
class TrainConfig:
    episodes: int = 400
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_steps: int = 500
    gamma: float = 0.98
    lr_pre: float = 5e-4
    lr_post: float = 3e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 12000


class LayerWrappedQAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        layer: FrozenLayer,
        rng: np.random.Generator,
        gamma: float,
        lr_pre: float,
        lr_post: float,
    ) -> None:
        self.num_actions = num_actions
        self.layer = layer
        self.gamma = gamma
        self.lr_pre = lr_pre
        self.lr_post = lr_post

        self.input_adapter = TinyInputAdapter(obs_dim, rng)
        self.output_adapter = TinyOutputAdapter(layer.output_dim, num_actions, rng)

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.input_adapter.forward(obs)
        z = self.layer.forward(x)
        q = self.output_adapter.forward(z)
        return x, z, q

    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if float(rng.random()) < epsilon:
            return int(rng.integers(0, self.num_actions))
        _, _, q = self._forward(obs[None, :])
        return int(np.argmax(q[0]))

    def train_batch(self, batch: Dict[str, np.ndarray]) -> float:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        x, z, q = self._forward(obs)
        with np.errstate(over="ignore"):
            _, _, q_next = self._forward(next_obs)

        next_best = np.max(q_next, axis=1)
        target = rewards + self.gamma * (1.0 - dones) * next_best

        pred = q[np.arange(q.shape[0]), actions]
        td_error = pred - target
        loss = float(0.5 * np.mean(td_error * td_error))

        grad_q = np.zeros_like(q, dtype=np.float32)
        grad_q[np.arange(q.shape[0]), actions] = td_error.astype(np.float32)

        grad_z = self.output_adapter.backward(z, grad_q, lr=self.lr_post)
        grad_x = self.layer.backward(x, grad_z)
        _ = self.input_adapter.backward(obs, grad_x, lr=self.lr_pre)

        return loss


def _epsilon_at(step: int, config: TrainConfig) -> float:
    if step >= config.eps_decay_steps:
        return config.eps_end
    frac = step / max(1, config.eps_decay_steps)
    return float(config.eps_start + frac * (config.eps_end - config.eps_start))


def train_single_layer(
    layer: FrozenLayer,
    seed: int,
    env_kwargs: Dict[str, int],
    config: TrainConfig,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = GridVideoEnv(**env_kwargs, seed=seed)
    agent = LayerWrappedQAgent(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        layer=layer,
        rng=rng,
        gamma=config.gamma,
        lr_pre=config.lr_pre,
        lr_post=config.lr_post,
    )
    replay = ReplayBuffer(obs_dim=env.obs_dim, capacity=config.replay_capacity)

    episode_returns = np.zeros((config.episodes,), dtype=np.float32)
    episode_lengths = np.zeros((config.episodes,), dtype=np.int32)
    losses = []

    global_step = 0
    for ep in range(config.episodes):
        obs, _ = env.reset(seed=seed * 1009 + ep)
        ep_ret = 0.0
        ep_len = 0

        done = False
        while not done:
            epsilon = _epsilon_at(global_step, config)
            action = agent.act(obs, epsilon=epsilon, rng=rng)
            next_obs, reward, done, _ = env.step(action)

            replay.add(obs, action, reward, next_obs, done)

            if len(replay) >= max(config.batch_size, config.warmup_steps):
                batch = replay.sample(config.batch_size, rng)
                loss = agent.train_batch(batch)
                losses.append(loss)

            obs = next_obs
            ep_ret += reward
            ep_len += 1
            global_step += 1

            if ep_len >= env.max_steps:
                break

        episode_returns[ep] = ep_ret
        episode_lengths[ep] = ep_len

    return {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "mean_loss": np.array([float(np.mean(losses)) if losses else np.nan], dtype=np.float32),
    }

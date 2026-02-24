from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


Action = int
Position = Tuple[int, int]


@dataclass
class StepInfo:
    state: np.ndarray
    success: bool


class GridVideoEnv:
    """A tiny video-like RL environment with image-sequence observations.

    Observation: (history, 2, grid, grid)
    - channel 0: agent occupancy map
    - channel 1: goal occupancy map
    """

    def __init__(
        self,
        grid_size: int = 8,
        history: int = 2,
        max_steps: int = 60,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.history = history
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self._rng = np.random.default_rng(seed)

        self._agent: Position = (0, 0)
        self._goal: Position = (0, 0)
        self._step = 0
        self._frames: list[np.ndarray] = []

    @property
    def obs_shape(self) -> Tuple[int, int, int, int]:
        return (self.history, 2, self.grid_size, self.grid_size)

    @property
    def obs_dim(self) -> int:
        h, c, g1, g2 = self.obs_shape
        return h * c * g1 * g2

    @property
    def num_actions(self) -> int:
        # stay, up, down, left, right
        return 5

    def _sample_position(self) -> Position:
        return (
            int(self._rng.integers(0, self.grid_size)),
            int(self._rng.integers(0, self.grid_size)),
        )

    def _state_vector(self) -> np.ndarray:
        denom = max(1, self.grid_size - 1)
        ax, ay = self._agent
        gx, gy = self._goal
        return np.array([ax / denom, ay / denom, gx / denom, gy / denom], dtype=np.float32)

    def _build_frame(self) -> np.ndarray:
        frame = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        ax, ay = self._agent
        gx, gy = self._goal
        frame[0, ay, ax] = 1.0
        frame[1, gy, gx] = 1.0
        return frame

    def _get_obs(self) -> np.ndarray:
        obs = np.stack(self._frames, axis=0)
        return obs.reshape(-1).astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, StepInfo]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._agent = self._sample_position()
        self._goal = self._sample_position()
        while self._goal == self._agent:
            self._goal = self._sample_position()

        self._step = 0
        first_frame = self._build_frame()
        self._frames = [first_frame.copy() for _ in range(self.history)]
        return self._get_obs(), StepInfo(state=self._state_vector(), success=False)

    def sample_action(self) -> Action:
        return int(self._rng.integers(0, self.num_actions))

    def _apply_action(self, action: Action) -> None:
        ax, ay = self._agent
        if action == 1:
            ay -= 1
        elif action == 2:
            ay += 1
        elif action == 3:
            ax -= 1
        elif action == 4:
            ax += 1

        ax = int(np.clip(ax, 0, self.grid_size - 1))
        ay = int(np.clip(ay, 0, self.grid_size - 1))
        self._agent = (ax, ay)

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, StepInfo]:
        self._step += 1
        self._apply_action(action)

        success = self._agent == self._goal
        reward = self.goal_reward if success else self.step_penalty
        done = success or self._step >= self.max_steps

        self._frames.pop(0)
        self._frames.append(self._build_frame())

        info = StepInfo(state=self._state_vector(), success=success)
        return self._get_obs(), float(reward), bool(done), info


def action_to_delta(action: Action) -> Tuple[int, int]:
    mapping: Dict[int, Tuple[int, int]] = {
        0: (0, 0),
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
    }
    return mapping.get(int(action), (0, 0))

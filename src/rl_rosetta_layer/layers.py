from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LayerSpec:
    name: str
    input_dim: int
    output_dim: int


class FrozenLayer:
    def __init__(self, spec: LayerSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def output_dim(self) -> int:
        return self.spec.output_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        """Gradient wrt layer input (weights remain frozen)."""
        raise NotImplementedError


class RandomTanhLayer(FrozenLayer):
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        scale: float = 0.25,
    ) -> None:
        super().__init__(LayerSpec(name=name, input_dim=input_dim, output_dim=output_dim))
        self.W = (rng.standard_normal((input_dim, output_dim)).astype(np.float32) * scale)
        self.b = (rng.standard_normal((output_dim,)).astype(np.float32) * scale)

    def forward(self, x: np.ndarray) -> np.ndarray:
        pre = x @ self.W + self.b
        return np.tanh(pre).astype(np.float32)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        pre = x @ self.W + self.b
        tanh_pre = np.tanh(pre)
        grad_pre = grad_out * (1.0 - tanh_pre * tanh_pre)
        return (grad_pre @ self.W.T).astype(np.float32)


class StructuredSpatialLayer(FrozenLayer):
    """Task-aligned frozen layer that extracts geometry from the final frame."""

    def __init__(self, name: str, obs_dim: int, grid_size: int, history: int) -> None:
        super().__init__(LayerSpec(name=name, input_dim=obs_dim, output_dim=8))
        self.grid_size = grid_size
        self.history = history

        xs = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        self.x_coords = xg.reshape(-1).astype(np.float32)
        self.y_coords = yg.reshape(-1).astype(np.float32)

    def _unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bsz = x.shape[0]
        obs = x.reshape(bsz, self.history, 2, self.grid_size, self.grid_size)
        final = obs[:, -1]
        agent_map = final[:, 0].reshape(bsz, -1)
        goal_map = final[:, 1].reshape(bsz, -1)
        return obs, agent_map, goal_map

    def forward(self, x: np.ndarray) -> np.ndarray:
        _, a, g = self._unpack(x)
        ax = a @ self.x_coords
        ay = a @ self.y_coords
        gx = g @ self.x_coords
        gy = g @ self.y_coords

        dx = ax - gx
        dy = ay - gy
        z = np.stack(
            [
                ax,
                ay,
                gx,
                gy,
                dx,
                dy,
                dx * dx,
                dy * dy,
            ],
            axis=1,
        )
        return z.astype(np.float32)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        obs, a, g = self._unpack(x)

        ax = a @ self.x_coords
        ay = a @ self.y_coords
        gx = g @ self.x_coords
        gy = g @ self.y_coords
        dx = ax - gx
        dy = ay - gy

        g_ax = grad_out[:, 0] + grad_out[:, 4] + 2.0 * dx * grad_out[:, 6]
        g_ay = grad_out[:, 1] + grad_out[:, 5] + 2.0 * dy * grad_out[:, 7]
        g_gx = grad_out[:, 2] - grad_out[:, 4] - 2.0 * dx * grad_out[:, 6]
        g_gy = grad_out[:, 3] - grad_out[:, 5] - 2.0 * dy * grad_out[:, 7]

        grad_agent = (
            g_ax[:, None] * self.x_coords[None, :] + g_ay[:, None] * self.y_coords[None, :]
        )
        grad_goal = (
            g_gx[:, None] * self.x_coords[None, :] + g_gy[:, None] * self.y_coords[None, :]
        )

        grad_obs = np.zeros_like(obs, dtype=np.float32)
        grad_obs[:, -1, 0] = grad_agent.reshape(-1, self.grid_size, self.grid_size)
        grad_obs[:, -1, 1] = grad_goal.reshape(-1, self.grid_size, self.grid_size)
        return grad_obs.reshape(x.shape).astype(np.float32)


def build_candidate_layers(
    obs_dim: int,
    grid_size: int,
    history: int,
    seed: int = 7,
) -> Dict[str, FrozenLayer]:
    rng = np.random.default_rng(seed)
    return {
        "layer_random_wide": RandomTanhLayer(
            name="layer_random_wide",
            input_dim=obs_dim,
            output_dim=128,
            rng=rng,
            scale=0.20,
        ),
        "layer_structured_mid": StructuredSpatialLayer(
            name="layer_structured_mid",
            obs_dim=obs_dim,
            grid_size=grid_size,
            history=history,
        ),
        "layer_random_narrow": RandomTanhLayer(
            name="layer_random_narrow",
            input_dim=obs_dim,
            output_dim=16,
            rng=rng,
            scale=0.30,
        ),
        "layer_bottleneck": RandomTanhLayer(
            name="layer_bottleneck",
            input_dim=obs_dim,
            output_dim=4,
            rng=rng,
            scale=0.35,
        ),
    }

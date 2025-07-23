# Factory Layout Placement with Reinforcement Learning
"""
FactoryLayoutEnv: A Gymnasium-compatible environment for training RL agents to
place rectangular equipment blocks inside a rectangular factory floor.

Key design points
-----------------
1. **Classes & equipment sizes** – Each equipment belongs to a class.  All
   items within a class share identical (w, h) dimensions.
2. **Obstacle mask** – The factory may contain cells where placement is
   impossible.  These can be injected via an obstacle mask when the
   environment is created or later via ``add_obstacle``.
3. **Reward structure** – Multi‑objective: maximise utilisation (area
   covered), encourage clustering of same‑class items, and reward alignment
   (same rows / columns).  All terms are configurable via ``reward_cfg``.
4. **Visualisation** – Built‑in ``render`` method plots the current layout
   with distinct colours per class (obstacles rendered black).

Dependencies
------------
- gymnasium
- numpy
- matplotlib
- stable-baselines3 (for PPO training example)

Usage
-----
```bash
pip install gymnasium stable-baselines3 matplotlib
```

```python
from factory_layout_rl import FactoryLayoutEnv
import stable_baselines3 as sb3

# Define block classes: class_id -> (width, height)
class_dims = {1: (2, 2), 2: (3, 1), 3: (1, 1)}
class_counts = {1: 10, 2: 6, 3: 20}

env = FactoryLayoutEnv(width=20, height=10,
                       class_dims=class_dims,
                       class_counts=class_counts)
model = sb3.PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Evaluate
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.render()
```
"""
from __future__ import annotations

import itertools
from typing import Dict, Tuple, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class FactoryLayoutEnv(gym.Env):
    """Gymnasium environment for facility layout optimisation.

    The layout is discretised into *cells* of unit size. Each equipment class
    occupies a (w, h) rectangle measured in cells.  During an episode the agent
    receives a stream of items to place (drawn from *class_counts*).  The agent
    selects a *grid coordinate* for each item; orientation is fixed (no
    rotation) because all blocks are rectangles with class‑defined width &
    height.

    Observation
    -----------
    A flattened integer grid of shape ``(height * width,)`` whose values are:
        0           – empty, placeable cell
        -1          – obstacle cell (unplaceable)
        1..N        – class id occupying the cell

    Action
    ------
    Discrete ``height * width``.  The action ``a`` is mapped to
    ``(row = a // width, col = a % width)``.

    Reward (dense)
    --------------
    ``reward = R_area + R_cluster + R_align``
        * *R_area*   – +area_weight * placed_area
        * *R_cluster*– -cluster_weight * mean_pairwise_distance(same‑class)
        * *R_align*  – +align_weight if the new block shares a row or column
                        with any existing same‑class block
    Invalid placements yield *-invalid_penalty* and the step is skipped.

    Episode termination
    -------------------
    Episode ends when every item is placed *or* the agent reaches the maximum
    number of placement attempts (sum(class_counts) * 2 by default).
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        width: int,
        height: int,
        class_dims: Dict[int, Tuple[int, int]],
        class_counts: Dict[int, int],
        obstacle_mask: Optional[np.ndarray] = None,
        reward_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        assert width > 0 and height > 0, "Factory dimensions must be positive"
        self.W = width
        self.H = height
        self.class_dims = class_dims
        self.class_counts_init = class_counts
        self.n_classes = len(class_dims)

        self.obstacle_mask = (
            obstacle_mask.copy()
            if obstacle_mask is not None
            else np.zeros((self.H, self.W), dtype=np.int8)
        )
        assert self.obstacle_mask.shape == (self.H, self.W)

        # Reward hyper‑parameters
        default_reward_cfg = {
            "area_weight": 1.0,
            "cluster_weight": 0.1,
            "align_weight": 0.2,
            "invalid_penalty": 1.0,
        }
        self.reward_cfg = {**default_reward_cfg, **(reward_cfg or {})}

        # Spaces
        self.observation_space = spaces.Box(
            low=-1,
            high=max(class_dims.keys()),
            shape=(self.H * self.W,),
            dtype=np.int16,
        )
        self.action_space = spaces.Discrete(self.H * self.W)

        # Dynamic state (set in reset())
        self.grid: np.ndarray  # shape (H, W)
        self.remaining_items: List[int]
        self.attempt_limit: int
        self.reset()

    # ---------------------------------------------------------------------
    # Environment API implementation
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs):  # noqa: D401
        super().reset(seed=seed)
        self.grid = np.where(self.obstacle_mask == 1, -1, 0).astype(np.int16)
        # Expand counts into a list of class ids and shuffle
        self.remaining_items = list(
            itertools.chain.from_iterable(
                [[cls] * cnt for cls, cnt in self.class_counts_init.items()]
            )
        )
        self.np_random.shuffle(self.remaining_items)
        self.attempts = 0
        self.attempt_limit = len(self.remaining_items) * 2  # heuristic
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action: int):  # noqa: D401
        if not self.remaining_items:
            raise RuntimeError("No remaining items; call reset() first.")

        row, col = divmod(action, self.W)
        cls = self.remaining_items[0]
        w, h = self.class_dims[cls]

        self.attempts += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Check placement feasibility
        if self._can_place(row, col, w, h):
            self._place(row, col, w, h, cls)
            self.remaining_items.pop(0)
            reward += self.reward_cfg["area_weight"] * (w * h)
            reward += self.reward_cfg["cluster_weight"] * (
                -self._mean_pairwise_dist(cls)
            )
            if self._shares_row_or_col(row, col, w, h, cls):
                reward += self.reward_cfg["align_weight"]

        else:  # invalid placement
            reward -= self.reward_cfg["invalid_penalty"]

        # Check termination conditions
        if not self.remaining_items:
            terminated = True
        if self.attempts >= self.attempt_limit:
            truncated = True

        observation = self._get_obs()
        info = {}
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return self.grid.flatten().copy()

    def _can_place(self, r: int, c: int, w: int, h: int) -> bool:
        if r < 0 or c < 0 or r + h > self.H or c + w > self.W:
            return False
        sub = self.grid[r : r + h, c : c + w]
        return np.all(sub == 0)  # all empty & placeable

    def _place(self, r: int, c: int, w: int, h: int, cls: int) -> None:
        self.grid[r : r + h, c : c + w] = cls

    def _mean_pairwise_dist(self, cls: int) -> float:
        ys, xs = np.where(self.grid == cls)
        if len(xs) <= 1:
            return 0.0
        coords = np.column_stack((ys, xs))
        # Compute pairwise Manhattan distances efficiently
        diffs = np.abs(coords[:, None, :] - coords[None, :, :])
        dists = diffs.sum(axis=-1)
        return dists[np.triu_indices(len(coords), k=1)].mean()

    def _shares_row_or_col(self, r: int, c: int, w: int, h: int, cls: int) -> bool:
        ys, xs = np.where(self.grid == cls)
        if len(xs) == 0:
            return False
        # New block occupies rows [r, r+h) & cols [c, c+w)
        rows_new = set(range(r, r + h))
        cols_new = set(range(c, c + w))
        return any(y in rows_new for y in ys) or any(x in cols_new for x in xs)

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    def add_obstacle(self, top: int, left: int, w: int, h: int) -> None:
        """Mark a (w, h) rectangle as an obstacle (unplaceable)."""
        self.obstacle_mask[top : top + h, left : left + w] = 1
        if hasattr(self, "grid"):
            self.grid[top : top + h, left : left + w] = -1

    def render(self, mode="human"):
        cmap = ListedColormap([
            "#ffffff",  # 0 empty
            "#1f77b4",  # 1
            "#2ca02c",  # 2
            "#ff7f0e",  # 3
            "#d62728",  # 4
            "#9467bd",  # 5
            "#8c564b",  # 6
            "#e377c2",  # 7
            "#7f7f7f",  # 8
            "#bcbd22",  # 9
        ])
        show_grid = self.grid.copy()
        show_grid[show_grid == -1] = 9  # obstacles as index 9 (dark gray)
        plt.figure(figsize=(self.W / 2, self.H / 2))
        plt.imshow(show_grid, cmap=cmap, origin="upper")
        plt.xticks(np.arange(-0.5, self.W, 1), [])
        plt.yticks(np.arange(-0.5, self.H, 1), [])
        plt.grid(color="black", linestyle="-", linewidth=0.5)
        plt.title("Factory Layout – step {}".format(self.attempts))
        plt.tight_layout()
        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            import io
            from PIL import Image

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return np.asarray(Image.open(buf))
        else:
            raise NotImplementedError


# ---------------------------------------------------------------------------
# Quick sanity check (run as module)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class_dims = {1: (2, 2), 2: (3, 1), 3: (1, 1)}
    class_counts = {1: 8, 2: 5, 3: 15}
    env = FactoryLayoutEnv(20, 10, class_dims, class_counts)

    try:
        import stable_baselines3 as sb3

        model = sb3.PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10_000)

        obs, _ = env.reset()
        done, tr = False, False
        while not (done or tr):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = env.step(action)
        env.render()
    except ImportError:
        print("stable-baselines3 not installed – running one random episode")
        obs, _ = env.reset()
        while env.remaining_items:
            action = env.action_space.sample()
            obs, reward, done, tr, _ = env.step(action)
            if done or tr:
                break
        env.render()

from __future__ import annotations

from typing import Protocol

import torch

from envs.action_space import ActionSpace


class Agent(Protocol):
    """Evaluate action_space for the given state."""

    def policy(self, *, obs: dict, action_space: ActionSpace) -> torch.Tensor:
        """Return float32 [N] non-negative policy scores/probabilities (not necessarily normalized)."""

    def select_action(self, *, obs: dict, action_space: ActionSpace) -> int:
        """Return an action index in [0, N)."""

    def value(self, *, obs: dict, action_space: ActionSpace) -> float:
        """Return a scalar leaf value estimate for MCTS (higher should be better)."""

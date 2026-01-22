from __future__ import annotations

from typing import Protocol

import torch

from envs.env import FactoryLayoutEnv
from actionspace.candidate_set import CandidateSet


class Agent(Protocol):
    """Evaluate candidates for the given state."""

    def policy(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> torch.Tensor:
        """Return float32 [N] non-negative policy scores/probabilities (not necessarily normalized)."""

    def select_action(self, *, env: FactoryLayoutEnv, obs: dict, candidates: CandidateSet) -> int:
        """Return an action index in [0, N)."""


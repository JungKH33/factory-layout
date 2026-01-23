from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import gymnasium as gym
import torch

from envs.env import FactoryLayoutEnv


class BaseWrapper(gym.Env, ABC):
    """Base wrapper for a FactoryLayoutEnv.

    Wrapper responsibilities:
    - Own an action-space adapter (Discrete(N))
    - Provide `action_mask` (valid-action mask) and any policy-specific observation fields
    - Decode discrete `action` to a concrete placement `(x_bl, y_bl, rot)`
    """

    metadata = {"render_modes": []}

    def __init__(self, *, engine: FactoryLayoutEnv):
        super().__init__()
        self.engine = engine
        self.device = engine.device
        self.mask: Optional[torch.Tensor] = None

    @abstractmethod
    def create_mask(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _build_obs(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        _obs_core, info = self.engine.reset(seed=seed, options=options)
        self.mask = self.create_mask()
        return self._build_obs(), info


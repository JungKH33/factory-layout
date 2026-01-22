from __future__ import annotations

# NOTE: 레거시 보관용 파일입니다. 실행/호환성은 신경 쓰지 않습니다.
# (원래 `agents/greedy.py`에 섞여 있던 env_old 기반 Greedy 구현)

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from envs.env_old import Candidate, FactoryLayoutEnvOld
from old.policies.topk_selector import TopKConfig, TopKSelector

Point = Tuple[float, float]


@dataclass(frozen=True)
class GreedyConfig:
    topk_config: TopKConfig
    prior_temperature: float = 1.0


class GreedyAgent:
    """Greedy agent that reuses TopK candidates and picks best score."""

    def __init__(self, config: GreedyConfig):
        self.config = config
        self._selector = TopKSelector(config.topk_config)

    def get_candidates(self, env: FactoryLayoutEnvOld) -> Tuple[List[Candidate], np.ndarray]:
        gid = self._next_gid(env)
        if gid is None:
            return [], np.zeros((0,), dtype=np.int8)
        return self._selector.generate(env, gid)

    def select_action(
        self,
        env: FactoryLayoutEnvOld,
        candidates: List[Candidate],
        mask: np.ndarray,
    ) -> int:
        if not candidates:
            return 0
        valid_indices = [i for i, m in enumerate(mask) if m == 1]
        if not valid_indices:
            return 0
        scored = [
            (env.estimate_delta_obj(candidates[i].group_id, candidates[i].x, candidates[i].y, candidates[i].rot), i)
            for i in valid_indices
        ]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def get_action_priors(
        self,
        env: FactoryLayoutEnvOld,
        candidates: List[Candidate],
        mask: np.ndarray,
    ) -> np.ndarray:
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        priors = np.zeros((len(candidates),), dtype=np.float32)
        valid_indices = [i for i, m in enumerate(mask) if m == 1]
        if not valid_indices:
            return priors
        scores = np.array(
            [env.estimate_delta_obj(candidates[i].group_id, candidates[i].x, candidates[i].y, candidates[i].rot) for i in valid_indices],
            dtype=np.float32,
        )
        temp = self.config.prior_temperature if self.config.prior_temperature > 0 else 1.0
        logits = -scores / temp
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs_sum = float(np.sum(probs))
        if probs_sum <= 0:
            probs = np.full_like(probs, 1.0 / len(valid_indices))
        else:
            probs = probs / probs_sum
        for idx, p in zip(valid_indices, probs):
            priors[idx] = p
        return priors

    def get_selector_state(self) -> object:
        return self._selector.get_state()

    def set_selector_state(self, state: object) -> None:
        self._selector.set_state(state)

    def _next_gid(self, env: FactoryLayoutEnvOld) -> Union[int, str, None]:
        return env.remaining[0] if env.remaining else None


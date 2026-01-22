from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from envs.env_old import Candidate, FactoryLayoutEnvOld
from old.policies.mcts import BaseAgent


@dataclass(frozen=True)
class BeamConfig:
    """Configuration for Beam Search."""

    beam_width: int = 5
    depth: int = 3


class BeamSearchAgent(BaseAgent):
    """Legacy env_old-based beam search agent."""

    def __init__(self, base_agent: BaseAgent, config: BeamConfig):
        self.base_agent = base_agent
        self.config = config

    def get_candidates(self, env: FactoryLayoutEnvOld) -> Tuple[List[Candidate], np.ndarray]:
        return self.base_agent.get_candidates(env)

    def select_action(self, env: FactoryLayoutEnvOld, candidates: List[Candidate], mask: np.ndarray) -> int:
        if not candidates or int(np.sum(mask)) == 0:
            return 0

        root_snapshot = env.get_snapshot()
        selector_state = self.base_agent.get_selector_state()

        valid_indices = [i for i, m in enumerate(mask) if int(m) == 1]
        beams: List[Tuple[int, float, Dict[str, object]]] = []

        for idx in valid_indices:
            env.set_snapshot(root_snapshot)
            env.set_candidates(candidates, mask)
            _, reward, terminated, truncated, _ = env.step(idx)
            beams.append((idx, float(reward), env.get_snapshot()))

        for _d in range(1, self.config.depth):
            new_candidates: List[Tuple[int, float, Dict[str, object]]] = []
            for first_action, current_reward, snapshot in beams:
                env.set_snapshot(snapshot)
                if len(env.remaining) == 0:
                    new_candidates.append((first_action, current_reward, snapshot))
                    continue
                next_cands, next_mask = self.base_agent.get_candidates(env)
                if not next_cands or int(np.sum(next_mask)) == 0:
                    new_candidates.append((first_action, current_reward, snapshot))
                    continue
                valid_next = [i for i, m in enumerate(next_mask) if int(m) == 1]
                for n_idx in valid_next:
                    env.set_snapshot(snapshot)
                    env.set_candidates(next_cands, next_mask)
                    _, r, terminated, truncated, _ = env.step(n_idx)
                    new_candidates.append((first_action, current_reward + float(r), env.get_snapshot()))

            if not new_candidates:
                break
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = new_candidates[: self.config.beam_width]

        env.set_snapshot(root_snapshot)
        self.base_agent.set_selector_state(selector_state)

        if not beams:
            return valid_indices[0] if valid_indices else 0
        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[0][0]

    def get_action_priors(self, env: FactoryLayoutEnvOld, candidates: List[Candidate], mask: np.ndarray) -> np.ndarray:
        return self.base_agent.get_action_priors(env, candidates, mask)

    def get_selector_state(self) -> object:
        return self.base_agent.get_selector_state()

    def set_selector_state(self, state: object) -> None:
        self.base_agent.set_selector_state(state)


from __future__ import annotations

from typing import Protocol

from envs.env import FactoryLayoutEnv

from agents.base import Agent
from actionspace.base import CandidateSelector
from actionspace.candidate_set import CandidateSet


class SearchStrategy(Protocol):
    """Choose an action index among candidates for the current state."""

    def select(
        self,
        *,
        env: FactoryLayoutEnv,
        obs: dict,
        agent: Agent,
        selector: CandidateSelector,
        root_candidates: CandidateSet,
        root_selector_state: object,
    ) -> int: ...


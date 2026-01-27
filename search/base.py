from __future__ import annotations

from typing import Protocol

from envs.wrappers.base import BaseWrapper

from agents.base import Agent
from envs.wrappers.candidate_set import CandidateSet


class SearchStrategy(Protocol):
    """Choose an action index among candidates for the current state."""

    def select(
        self,
        *,
        env: BaseWrapper,
        obs: dict,
        agent: Agent,
        root_candidates: CandidateSet,
    ) -> int: ...


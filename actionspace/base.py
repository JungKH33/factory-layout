from __future__ import annotations

from typing import Protocol

from envs.env import FactoryLayoutEnv

from .candidate_set import CandidateSet


class CandidateSelector(Protocol):
    """Build candidates for the *current* env state."""

    def build(self, env: FactoryLayoutEnv) -> CandidateSet: ...

    def get_state(self) -> object:  # RNG etc.
        return None

    def set_state(self, state: object) -> None:
        return None


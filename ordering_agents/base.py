from __future__ import annotations

from typing import Any, Protocol

from envs.env import FactoryLayoutEnv


class OrderingAgent(Protocol):
    """Reorder env.get_state().remaining before wrapper candidate generation."""

    def reorder(self, *, env: FactoryLayoutEnv, obs: dict[str, Any]) -> None:
        """Mutate env.get_state().remaining through env APIs (e.g., env.reorder_remaining)."""

